import asyncio
import aiohttp
from loguru import logger
from .config import Config
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider
from .providers.local_db import LocalDatabaseProvider

class DataManager:
    def __init__(self):
        self.db = DBManager()
        
        # 根据配置选择数据源
        if Config.DATA_SOURCE == "local_db":
            self.provider = LocalDatabaseProvider()
            logger.info("使用本地数据库 (CB_HISTORY) 作为数据源")
        else:
            self.provider = BirdeyeProvider()
            logger.info("使用 Birdeye API 作为数据源")
        
        # 保留兼容性引用
        self.birdeye = self.provider if isinstance(self.provider, BirdeyeProvider) else BirdeyeProvider()
        self.dexscreener = DexScreenerProvider()
        
    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.db.close()
        # 关闭本地数据库连接
        if hasattr(self.provider, 'close'):
            self.provider.close()

    async def pipeline_sync_daily(self):
        logger.info("Step 1: Discovering trending tokens...")
        
        # 根据数据源调整 limit
        if Config.DATA_SOURCE == "local_db":
            limit = 500  # 本地数据库不受 API 限制
        else:
            limit = 500 if Config.BIRDEYE_IS_PAID else 100
            
        candidates = await self.provider.get_trending_tokens(limit=limit)
        
        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for t in candidates:
            liq = t.get('liquidity', 0) or 0
            fdv = t.get('fdv', 0) or 0
            
            # 本地数据库模式下跳过流动性过滤
            if Config.DATA_SOURCE != "local_db":
                if liq < Config.MIN_LIQUIDITY_USD: continue
                if fdv < Config.MIN_FDV: continue
                if fdv > Config.MAX_FDV: continue
            
            selected_tokens.append(t)
            
        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        
        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        # 确定 chain 值
        chain = "CB" if Config.DATA_SOURCE == "local_db" else Config.CHAIN
        db_tokens = [(t['address'], t['symbol'], t['name'], t.get('decimals', 0), chain) for t in selected_tokens]
        await self.db.upsert_tokens(db_tokens)

        logger.info(f"Step 4: Fetching OHLCV for {len(selected_tokens)} tokens...")
        
        # 使用 provider 的 headers（如果存在）
        headers = getattr(self.provider, 'headers', {})
        
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = []
            for t in selected_tokens:
                tasks.append(self.provider.get_token_history(session, t['address']))
            
            batch_size = 20
            total_candles = 0
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                results = await asyncio.gather(*batch)
                
                records = [item for sublist in results if sublist for item in sublist]
                
                # 批量写入
                await self.db.batch_insert_ohlcv(records)
                total_candles += len(records)
                logger.info(f"Processed batch {i}/{len(tasks)}. Inserted {len(records)} candles.")
                
        logger.success(f"Pipeline complete. Total candles stored: {total_candles}")
