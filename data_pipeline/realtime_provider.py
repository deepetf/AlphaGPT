"""
RealtimeDataProvider - 实时数据提供者

整合 Mini QMT (xtquant.xtdata) 和本地 SQL 数据库，
为模拟盘提供因子计算所需的实时行情和 CB 特性数据。
"""
import logging
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine, text

from data_pipeline.config import Config
from model_core.config import ModelConfig

logger = logging.getLogger(__name__)


class RealtimeDataProvider:
    """
    实时数据提供者
    
    职责:
    1. 从 Mini QMT 获取实时行情 (open, high, close, vol)
    2. 从本地 SQL 数据库获取 CB 特性数据 (pure_value, prem, etc.)
    3. 合并数据并构建因子计算所需的 feat_tensor
    """
    
    def __init__(self, sql_engine=None):
        """
        初始化数据提供者
        
        Args:
            sql_engine: SQLAlchemy 引擎，如果为 None 则使用默认配置创建
        """
        self.sql_engine = sql_engine or create_engine(Config.CB_DB_DSN)
        self._xtdata = None  # 延迟加载
        
    @property
    def xtdata(self):
        """延迟加载 xtdata 模块"""
        if self._xtdata is None:
            try:
                from xtquant import xtdata
                self._xtdata = xtdata
                logger.info("xtquant.xtdata 模块加载成功")
            except ImportError:
                logger.warning("无法加载 xtquant.xtdata，将使用模拟数据模式")
                self._xtdata = None
        return self._xtdata
    
    # =========================================================================
    # Mini QMT 行情接口
    # =========================================================================
    
    def download_history_data(self, code_list: List[str], period: str = '1d'):
        """
        下载/更新历史数据 (用于因子计算)
        
        Args:
            code_list: 标的代码列表 (如 ['123001.SZ', '127050.SZ'])
            period: 周期 ('1d', '1m', etc.)
        """
        if self.xtdata is None:
            logger.warning("xtdata 不可用，跳过历史数据下载")
            return
            
        logger.info(f"开始下载 {len(code_list)} 个标的的历史数据...")
        for code in code_list:
            self.xtdata.download_history_data(code, period=period, incrementally=True)
        logger.info("历史数据下载完成")
    
    def subscribe_quotes(self, code_list: List[str], period: str = '1d'):
        """
        订阅实时行情
        
        Args:
            code_list: 标的代码列表
            period: 周期
        """
        if self.xtdata is None:
            logger.warning("xtdata 不可用，跳过行情订阅")
            return
            
        for code in code_list:
            self.xtdata.subscribe_quote(code, period=period, count=-1)
        logger.info(f"已订阅 {len(code_list)} 个标的的实时行情")
    
    def get_realtime_quotes(self, code_list: List[str], period: str = '1d') -> pd.DataFrame:
        """
        从 Mini QMT 获取实时行情 (open, high, close, vol)
        
        Args:
            code_list: 标的代码列表
            period: 周期
            
        Returns:
            DataFrame with columns: [code, open, high, low, close, vol, amount]
        """
        if self.xtdata is None:
            logger.warning("xtdata 不可用，返回空 DataFrame")
            return pd.DataFrame()
        
        # 获取快照
        data = self.xtdata.get_market_data_ex([], code_list, period=period)
        
        # 转换为统一格式
        return self._format_qmt_data(data, code_list)
    
    def _format_qmt_data(self, data: Dict, code_list: List[str]) -> pd.DataFrame:
        """
        将 QMT 返回的数据格式化为统一的 DataFrame
        
        Args:
            data: xtdata.get_market_data_ex 返回的字典
            code_list: 标的代码列表
            
        Returns:
            DataFrame with columns: [code, trade_date, open, high, low, close, vol, amount]
        """
        rows = []
        for code in code_list:
            if code not in data:
                continue
            df = data[code]
            if df is None or df.empty:
                continue
            # 取最后一行 (最新数据)
            last_row = df.iloc[-1]
            rows.append({
                'code': code,
                'trade_date': df.index[-1] if hasattr(df.index[-1], 'strftime') else str(df.index[-1]),
                'open': float(last_row.get('open', 0)),
                'high': float(last_row.get('high', 0)),
                'low': float(last_row.get('low', 0)),
                'close': float(last_row.get('close', 0)),
                'vol': float(last_row.get('volume', 0)),
                'amount': float(last_row.get('amount', 0)),
            })
        return pd.DataFrame(rows)
    
    # =========================================================================
    # 本地 SQL 数据库接口
    # =========================================================================
    
    def get_cb_code_list(self) -> List[str]:
        """
        从数据库获取最新交易日的全部可转债代码列表
        
        Returns:
            List of CB codes
        """
        query = """
        SELECT DISTINCT code 
        FROM CB_DATA 
        WHERE trade_date = (SELECT MAX(trade_date) FROM CB_DATA)
        ORDER BY code
        """
        with self.sql_engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df['code'].tolist()
    
    def get_cb_features(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        从本地 SQL 数据库获取 CB 特性数据
        
        Args:
            date: 交易日期 (YYYY-MM-DD)，如果为 None 则使用最新日期
            
        Returns:
            DataFrame with CB characteristic columns
        """
        # 构建字段列表
        columns = ['code', 'name', 'trade_date']
        for internal_name, db_col, _ in ModelConfig.BASIC_FACTORS:
            if db_col not in columns:
                columns.append(db_col)
        
        columns_str = ', '.join(columns)
        
        if date:
            query = f"""
            SELECT {columns_str}
            FROM CB_DATA 
            WHERE trade_date = :date
            """
            params = {"date": date}
        else:
            query = f"""
            SELECT {columns_str}
            FROM CB_DATA 
            WHERE trade_date = (SELECT MAX(trade_date) FROM CB_DATA)
            """
            params = {}
        
        with self.sql_engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        logger.info(f"从 SQL 数据库获取 {len(df)} 条 CB 特性数据")
        return df
    
    def get_prev_close(self, date: str) -> Dict[str, float]:
        """
        获取 T-1 日的收盘价 (用于止盈计算)
        
        Args:
            date: 当前交易日期 (YYYY-MM-DD)
            
        Returns:
            Dict[code, prev_close]
        """
        query = """
        SELECT code, close
        FROM CB_DATA
        WHERE trade_date = (
            SELECT MAX(trade_date) FROM CB_DATA WHERE trade_date < :date
        )
        """
        with self.sql_engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"date": date})
        
        if df.empty:
            logger.warning(f"未找到 {date} 之前的交易日数据")
            return {}
        
        result = dict(zip(df['code'], df['close']))
        logger.info(f"获取 T-1 收盘价: {len(result)} 条")
        return result
    
    def get_trading_days_before(self, date: str, n: int) -> List[str]:
        """
        获取 date 及之前的 n 个交易日
        
        Args:
            date: 目标日期 (YYYY-MM-DD)
            n: 需要的交易日数量
            
        Returns:
            交易日列表 (按日期升序排列, 最后一个是 date)
        """
        query = """
        SELECT DISTINCT trade_date
        FROM CB_DATA
        WHERE trade_date <= :date
        ORDER BY trade_date DESC
        LIMIT :n
        """
        with self.sql_engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"date": date, "n": n})
        
        if df.empty:
            return []
        
        # 按升序返回
        dates = sorted(df['trade_date'].astype(str).tolist())
        return dates
    
    def get_cb_features_multi_days(self, dates: List[str]) -> pd.DataFrame:
        """
        获取多个交易日的 CB 特性数据
        
        Args:
            dates: 交易日列表 (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: code, name, trade_date, 及各特征列
        """
        if not dates:
            return pd.DataFrame()
        
        # 构建字段列表
        columns = ['code', 'name', 'trade_date']
        for internal_name, db_col, _ in ModelConfig.BASIC_FACTORS:
            if db_col not in columns:
                columns.append(db_col)
        
        columns_str = ', '.join(columns)
        
        # 使用 IN 子句查询多日数据
        placeholders = ', '.join([f":date_{i}" for i in range(len(dates))])
        query = f"""
        SELECT {columns_str}
        FROM CB_DATA 
        WHERE trade_date IN ({placeholders})
        ORDER BY trade_date, code
        """
        
        params = {f"date_{i}": d for i, d in enumerate(dates)}
        
        with self.sql_engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        logger.info(f"从 SQL 获取 {len(dates)} 天, {len(df)} 条 CB 数据")
        return df
    
    def build_feat_tensor_with_history(
        self, 
        date: str,
        realtime_quotes: pd.DataFrame,
        window: int = 5,
        strict_date_mode: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        构建包含历史窗口的特征张量 (支持 TS_* 时序算子)
        
        Args:
            date: 目标日期 (YYYY-MM-DD)
            realtime_quotes: 实时行情 DataFrame (用于覆盖当日数据)
            window: 历史窗口大小 (默认 5 天，支持 TS_MEAN5/TS_STD5)
            strict_date_mode: 严格日期模式
            
        Returns:
            feat_tensor: [Time=window, Assets, Features] 格式的张量
            asset_list: 资产代码列表 (与张量的 Assets 维度对应)
        """
        # 1. 获取历史交易日
        trading_days = self.get_trading_days_before(date, window)
        if len(trading_days) < window:
            logger.warning(f"历史数据不足: 需要 {window} 天，实际 {len(trading_days)} 天")
        
        if not trading_days:
            return torch.zeros((window, 0, len(ModelConfig.INPUT_FEATURES))), []
        
        # 2. 获取多日 CB 数据
        multi_day_df = self.get_cb_features_multi_days(trading_days)
        if multi_day_df.empty:
            return torch.zeros((window, 0, len(ModelConfig.INPUT_FEATURES))), []
        
        # 3. 获取所有日期都有的资产 (取交集)
        common_codes = None
        for d in trading_days:
            day_codes = set(multi_day_df[multi_day_df['trade_date'].astype(str) == d]['code'])
            if common_codes is None:
                common_codes = day_codes
            else:
                common_codes = common_codes & day_codes
        
        common_codes = sorted(list(common_codes)) if common_codes else []
        if not common_codes:
            logger.warning("没有在所有日期都存在的资产")
            return torch.zeros((window, 0, len(ModelConfig.INPUT_FEATURES))), []
        
        # 4. 如果有实时数据，验证日期并准备覆盖
        qmt_data = {}
        if not realtime_quotes.empty:
            qmt_date = str(realtime_quotes.iloc[0].get('trade_date', ''))[:10]
            if qmt_date and qmt_date != trading_days[-1]:
                msg = f"QMT 日期 ({qmt_date}) != 目标日期 ({trading_days[-1]})"
                if strict_date_mode:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
            
            # 构建 QMT 数据字典 {code: {col: value}}
            for _, row in realtime_quotes.iterrows():
                code = row['code']
                if code in common_codes:
                    qmt_data[code] = {
                        'open': float(row.get('open', 0)),
                        'high': float(row.get('high', 0)),
                        'low': float(row.get('low', 0)),
                        'close': float(row.get('close', 0)),
                        'vol': float(row.get('vol', 0)),
                    }
        
        # 5. 构建 [Time, Assets, Features] 张量
        factor_map = {internal: db_col for internal, db_col, _ in ModelConfig.BASIC_FACTORS}
        n_days = len(trading_days)
        n_assets = len(common_codes)
        n_features = len(ModelConfig.INPUT_FEATURES)
        
        feat_array = np.zeros((n_days, n_assets, n_features), dtype=np.float32)
        
        for t_idx, trade_date in enumerate(trading_days):
            day_df = multi_day_df[multi_day_df['trade_date'].astype(str) == trade_date]
            day_df = day_df.set_index('code')
            
            is_last_day = (t_idx == n_days - 1)
            
            for a_idx, code in enumerate(common_codes):
                if code not in day_df.index:
                    continue
                    
                row = day_df.loc[code]
                
                for f_idx, feature_name in enumerate(ModelConfig.INPUT_FEATURES):
                    db_col = factor_map.get(feature_name)
                    if db_col is None:
                        continue
                    
                    # 获取值
                    value = float(row.get(db_col, 0)) if db_col in day_df.columns else 0.0
                    
                    # 最后一天用 QMT 数据覆盖 (如有)
                    if is_last_day and code in qmt_data:
                        qmt_col = db_col.lower()
                        if qmt_col in qmt_data[code] and qmt_data[code][qmt_col] > 0:
                            value = qmt_data[code][qmt_col]
                    
                    feat_array[t_idx, a_idx, f_idx] = value
        
        # 处理 NaN
        feat_array = np.nan_to_num(feat_array, nan=0.0)
        feat_tensor = torch.tensor(feat_array, dtype=torch.float32, device=ModelConfig.DEVICE)
        
        logger.info(f"Built feat_tensor with history: {feat_tensor.shape} (days={n_days}, assets={n_assets}, features={n_features})")
        return feat_tensor, common_codes
    
    # =========================================================================
    # 数据整合接口
    # =========================================================================
    
    def build_feat_tensor(
        self, 
        realtime_quotes: pd.DataFrame, 
        cb_features: pd.DataFrame,
        strict_date_mode: bool = False
    ) -> torch.Tensor:
        """
        合并实时行情与 CB 特性数据，构建因子计算所需的 feat_tensor
        
        Args:
            realtime_quotes: 实时行情 DataFrame
            cb_features: CB 特性 DataFrame
            strict_date_mode: 若为 True，日期不一致时抛出异常；否则仅警告
        """
        # 结果 DataFrame
        merged = cb_features.copy()
        
        # 1. 日期对齐与实时数据合并
        if not realtime_quotes.empty:
            # 校验日期
            qmt_date = str(realtime_quotes.iloc[0]['trade_date'])[:10]
            sql_date = str(cb_features.iloc[0]['trade_date'])[:10]
            if qmt_date != sql_date:
                msg = f"数据日期不匹配: QMT={qmt_date}, SQL={sql_date}"
                if strict_date_mode:
                    raise ValueError(msg + " (strict_date_mode=True)")
                else:
                    logger.warning(msg + ". 请检查同步状态。")

            # 合并实时行情 (覆盖 SQL 中的基础价格列)
            merged = merged.merge(
                realtime_quotes[['code', 'open', 'high', 'low', 'close', 'vol']],
                on='code',
                how='left',
                suffixes=('_sql', '_qmt')
            )
            
            # 优先使用 QMT 数据
            for col in ['open', 'high', 'low', 'close', 'vol']:
                qmt_col = f'{col}_qmt'
                sql_col = f'{col}_sql'
                if qmt_col in merged.columns:
                    merged[col] = merged[qmt_col].fillna(merged.get(sql_col, 0))
                    merged.drop(columns=[qmt_col, sql_col], inplace=True, errors='ignore')
        
        # 2. 构建特征张量 (严格按 INPUT_FEATURES 顺序, 与 StackVM 对齐)
        # 创建 InternalName -> db_col 映射
        factor_map = {internal: db_col for internal, db_col, _ in ModelConfig.BASIC_FACTORS}
        
        feat_list = []
        for feature_name in ModelConfig.INPUT_FEATURES:
            # 从 BASIC_FACTORS 查找对应的数据库列名
            db_col = factor_map.get(feature_name)
            if db_col is None:
                logger.warning(f"Feature '{feature_name}' not found in BASIC_FACTORS, using 0 filling.")
                feat_list.append(np.zeros(len(merged)))
                continue
            
            # 健壮性处理: 处理可能因别名导致的列名变化
            actual_col = db_col
            if actual_col not in merged.columns:
                # 尝试查找 internal_name (在 merged 中可能已改为 internal_name)
                if feature_name in merged.columns:
                    actual_col = feature_name
                else:
                    logger.warning(f"Feature column '{db_col}' for '{feature_name}' not found, using 0 filling.")
                    feat_list.append(np.zeros(len(merged)))
                    continue
            
            values = merged[actual_col].values.astype(np.float32)
            # 填充 NaN
            values = np.nan_to_num(values, nan=0.0)
            feat_list.append(values)
        
        # [Features, Assets] -> [Assets, Features]
        feat_array = np.stack(feat_list, axis=0).T
        feat_tensor = torch.tensor(feat_array, dtype=torch.float32, device=ModelConfig.DEVICE)
        
        logger.info(f"Built feat_tensor: {feat_tensor.shape} (features: {len(ModelConfig.INPUT_FEATURES)})")
        return feat_tensor

    
    def get_asset_list(self, cb_features: pd.DataFrame) -> List[str]:
        """
        获取资产代码列表 (与 feat_tensor 行顺序一致)
        
        Args:
            cb_features: CB 特性 DataFrame
            
        Returns:
            List of asset codes
        """
        return cb_features['code'].tolist()
    
    def get_names_dict(self, cb_features: pd.DataFrame) -> Dict[str, str]:
        """
        获取 code -> name 映射字典
        
        Args:
            cb_features: CB 特性 DataFrame
            
        Returns:
            Dict mapping code to name
        """
        return dict(zip(cb_features['code'], cb_features['name']))
    
    def close(self):
        """关闭数据库连接"""
        if self.sql_engine:
            self.sql_engine.dispose()
            logger.info("SQL 数据库连接已关闭")
