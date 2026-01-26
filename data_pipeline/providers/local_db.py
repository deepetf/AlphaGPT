"""
LocalDatabaseProvider - 从 CB_HISTORY MySQL 数据库读取可转债数据

复用 local_libs/LidoDBClass.py 的数据库访问模式
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger

from .base import DataProvider
from ..config import Config


class LocalDatabaseProvider(DataProvider):
    """从本地 CB_HISTORY 数据库读取可转债数据"""
    
    def __init__(self):
        self.engine = create_engine(Config.CB_DB_DSN)
        self.headers = {}  # 兼容 BirdeyeProvider 接口
        
    async def get_trending_tokens(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        从 CB_DATA 表获取最新交易日的可转债列表
        
        模拟 API 返回格式以保持接口兼容
        """
        query = """
        SELECT code as address, 
               name as symbol, 
               name as name,
               0 as decimals,
               COALESCE(close * remain_cap, 0) as liquidity,
               COALESCE(close * remain_cap, 0) as fdv
        FROM CB_DATA 
        WHERE trade_date = (SELECT MAX(trade_date) FROM CB_DATA)
        ORDER BY remain_cap DESC
        LIMIT :limit
        """
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params={"limit": limit})
                
            logger.info(f"从本地数据库获取 {len(df)} 个可转债")
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"获取可转债列表失败: {e}")
            return []
    
    async def get_token_history(
        self, 
        session, 
        address: str, 
        days: int = 30
    ) -> Optional[List[tuple]]:
        """
        从 CB_DATA 表获取指定可转债的历史数据
        
        返回格式与 BirdeyeProvider 一致:
        (datetime, address, open, high, low, close, volume, liquidity, fdv, source)
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        query = """
        SELECT trade_date, code, open, high, low, close, 
               COALESCE(amount, 0) as volume, 
               COALESCE(remain_cap, 0) as liquidity, 
               COALESCE(close * remain_cap, 0) as fdv
        FROM CB_DATA 
        WHERE code = :code AND trade_date >= :start_date
        ORDER BY trade_date ASC
        """
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(query), 
                    conn, 
                    params={"code": address, "start_date": start_date}
                )
            
            if df.empty:
                return None
                
            # 转换为与 BirdeyeProvider 一致的 tuple 格式
            formatted = []
            for _, row in df.iterrows():
                formatted.append((
                    row['trade_date'],
                    address,
                    float(row['open']) if pd.notna(row['open']) else 0.0,
                    float(row['high']) if pd.notna(row['high']) else 0.0,
                    float(row['low']) if pd.notna(row['low']) else 0.0,
                    float(row['close']) if pd.notna(row['close']) else 0.0,
                    float(row['volume']) if pd.notna(row['volume']) else 0.0,
                    float(row['liquidity']) if pd.notna(row['liquidity']) else 0.0,
                    float(row['fdv']) if pd.notna(row['fdv']) else 0.0,
                    'CB_HISTORY'  # source
                ))
            return formatted
        except Exception as e:
            logger.error(f"获取 {address} 历史数据失败: {e}")
            return None
    
    def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("本地数据库连接已关闭")
