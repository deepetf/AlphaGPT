"""
RealtimeDataProvider

整合 Mini QMT (xtquant.xtdata) 与本地 SQL 数据库，
为模拟盘提供因子计算所需的实时行情和可转债特征数据。
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
    实时数据提供者。

    职责:
    1. 从 Mini QMT 获取实时行情 (open/high/low/close/vol)
    2. 从本地 SQL 数据库获取可转债特征数据
    3. 合并数据并构建因子计算所需的 feat_tensor
    """
    
    def __init__(self, sql_engine=None):
        """
        鍒濆鍖栨暟鎹彁渚涜€?
        
        Args:
            sql_engine: SQLAlchemy 寮曟搸锛屽鏋滀负 None 鍒欎娇鐢ㄩ粯璁ら厤缃垱寤?
        """
        self.sql_engine = sql_engine or create_engine(Config.CB_DB_DSN)
        self._xtdata = None  # 寤惰繜鍔犺浇
        
    @property
    def xtdata(self):
        """寤惰繜鍔犺浇 xtdata 妯″潡"""
        if self._xtdata is None:
            try:
                from xtquant import xtdata
                self._xtdata = xtdata
                logger.info("xtquant.xtdata loaded successfully")
            except ImportError:
                logger.warning("Failed to load xtquant.xtdata, fallback to mock mode")
                self._xtdata = None
        return self._xtdata
    
    # =========================================================================
    # Mini QMT 琛屾儏鎺ュ彛
    # =========================================================================
    
    def download_history_data(self, code_list: List[str], period: str = '1d'):
        """
        涓嬭浇/鏇存柊鍘嗗彶鏁版嵁 (鐢ㄤ簬鍥犲瓙璁＄畻)
        
        Args:
            code_list: 鏍囩殑浠ｇ爜鍒楄〃 (濡?['123001.SZ', '127050.SZ'])
            period: 鍛ㄦ湡 ('1d', '1m', etc.)
        """
        if self.xtdata is None:
            logger.warning("xtdata unavailable, skip history download")
            return
            
        logger.info(f"Start downloading history for {len(code_list)} symbols...")
        for code in code_list:
            self.xtdata.download_history_data(code, period=period, incrementally=True)
        logger.info("History download finished")
    
    def subscribe_quotes(self, code_list: List[str], period: str = '1d'):
        """
        璁㈤槄瀹炴椂琛屾儏
        
        Args:
            code_list: 鏍囩殑浠ｇ爜鍒楄〃
            period: 鍛ㄦ湡
        """
        if self.xtdata is None:
            logger.warning("xtdata unavailable, skip quote subscription")
            return
            
        for code in code_list:
            self.xtdata.subscribe_quote(code, period=period, count=-1)
        logger.info(f"Subscribed realtime quotes for {len(code_list)} symbols")
    
    def get_realtime_quotes(self, code_list: List[str], period: str = '1d') -> pd.DataFrame:
        """
        浠?Mini QMT 鑾峰彇瀹炴椂琛屾儏 (open, high, close, vol)
        
        Args:
            code_list: 鏍囩殑浠ｇ爜鍒楄〃
            period: 鍛ㄦ湡
            
        Returns:
            DataFrame with columns: [code, open, high, low, close, vol, amount]
        """
        if self.xtdata is None:
            logger.warning("xtdata unavailable, returning empty DataFrame")
            return pd.DataFrame()
        
        # 鑾峰彇蹇収
        data = self.xtdata.get_market_data_ex([], code_list, period=period)
        
        # 杞崲涓虹粺涓€鏍煎紡
        return self._format_qmt_data(data, code_list)
    def get_realtime_quotes_dummy(self, code_list: List[str], date: Optional[str] = None) -> pd.DataFrame:
        """
        Dummy 实时行情实现（联调占位）：
        从 SQL 读取指定交易日 OHLCV 模拟当前快照。
        """
        trade_date = date
        if trade_date is None:
            query_date = "SELECT MAX(trade_date) AS trade_date FROM CB_DATA"
            with self.sql_engine.connect() as conn:
                latest = pd.read_sql(text(query_date), conn)
            if latest.empty or pd.isna(latest.iloc[0]["trade_date"]):
                logger.warning("Dummy realtime quotes: no trade_date found in CB_DATA")
                return pd.DataFrame()
            trade_date = str(latest.iloc[0]["trade_date"])[:10]

        query = """
        SELECT code, trade_date, open, high, low, close, vol, amount
        FROM CB_DATA
        WHERE trade_date = :trade_date
        """
        with self.sql_engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"trade_date": trade_date})

        if df.empty:
            logger.warning(f"Dummy realtime quotes: no rows for trade_date={trade_date}")
            return df

        if code_list:
            df = df[df["code"].isin(set(code_list))].copy()

        logger.info(f"Dummy realtime quotes loaded: trade_date={trade_date}, rows={len(df)}")
        return df
    
    def _format_qmt_data(self, data: Dict, code_list: List[str]) -> pd.DataFrame:
        """
        灏?QMT 杩斿洖鐨勬暟鎹牸寮忓寲涓虹粺涓€鐨?DataFrame
        
        Args:
            data: xtdata.get_market_data_ex 杩斿洖鐨勫瓧鍏?
            code_list: 鏍囩殑浠ｇ爜鍒楄〃
            
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
            # 鍙栨渶鍚庝竴琛?(鏈€鏂版暟鎹?
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
    # 鏈湴 SQL 鏁版嵁搴撴帴鍙?
    # =========================================================================
    
    def get_cb_code_list(self) -> List[str]:
        """
        浠庢暟鎹簱鑾峰彇鏈€鏂颁氦鏄撴棩鐨勫叏閮ㄥ彲杞€轰唬鐮佸垪琛?
        
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
        浠庢湰鍦?SQL 鏁版嵁搴撹幏鍙?CB 鐗规€ф暟鎹?
        
        Args:
            date: 浜ゆ槗鏃ユ湡 (YYYY-MM-DD)锛屽鏋滀负 None 鍒欎娇鐢ㄦ渶鏂版棩鏈?
            
        Returns:
            DataFrame with CB characteristic columns
        """
        # 鏋勫缓瀛楁鍒楄〃
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
        
        logger.info(f"Loaded {len(df)} CB feature rows from SQL")
        return df
    
    def get_prev_close(self, date: str) -> Dict[str, float]:
        """
        鑾峰彇 T-1 鏃ョ殑鏀剁洏浠?(鐢ㄤ簬姝㈢泩璁＄畻)
        
        Args:
            date: 褰撳墠浜ゆ槗鏃ユ湡 (YYYY-MM-DD)
            
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
            logger.warning(f"No previous trading date found before {date}")
            return {}
        
        result = dict(zip(df['code'], df['close']))
        logger.info(f"Loaded {len(result)} T-1 close prices")
        return result
    
    def get_trading_days_before(self, date: str, n: int) -> List[str]:
        """
        鑾峰彇 date 鍙婁箣鍓嶇殑 n 涓氦鏄撴棩
        
        Args:
            date: 鐩爣鏃ユ湡 (YYYY-MM-DD)
            n: 闇€瑕佺殑浜ゆ槗鏃ユ暟閲?
            
        Returns:
            浜ゆ槗鏃ュ垪琛?(鎸夋棩鏈熷崌搴忔帓鍒? 鏈€鍚庝竴涓槸 date)
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
        
        # 鎸夊崌搴忚繑鍥?
        dates = sorted(df['trade_date'].astype(str).tolist())
        return dates
    
    def get_cb_features_multi_days(self, dates: List[str]) -> pd.DataFrame:
        """
        鑾峰彇澶氫釜浜ゆ槗鏃ョ殑 CB 鐗规€ф暟鎹?
        
        Args:
            dates: 浜ゆ槗鏃ュ垪琛?(YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: code, name, trade_date, 鍙婂悇鐗瑰緛鍒?
        """
        if not dates:
            return pd.DataFrame()
        
        # 鏋勫缓瀛楁鍒楄〃
        columns = ['code', 'name', 'trade_date']
        for internal_name, db_col, _ in ModelConfig.BASIC_FACTORS:
            if db_col not in columns:
                columns.append(db_col)
        
        columns_str = ', '.join(columns)
        
        # 浣跨敤 IN 瀛愬彞鏌ヨ澶氭棩鏁版嵁
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
        
        logger.info(f"Loaded multi-day SQL data: days={len(dates)}, rows={len(df)}")
        return df
    
    def build_feat_tensor_with_history(
        self, 
        date: str,
        realtime_quotes: pd.DataFrame,
        window: int = 5,
        strict_date_mode: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        鏋勫缓鍖呭惈鍘嗗彶绐楀彛鐨勭壒寰佸紶閲?(鏀寔 TS_* 鏃跺簭绠楀瓙)
        
        Args:
            date: 鐩爣鏃ユ湡 (YYYY-MM-DD)
            realtime_quotes: 瀹炴椂琛屾儏 DataFrame (鐢ㄤ簬瑕嗙洊褰撴棩鏁版嵁)
            window: 鍘嗗彶绐楀彛澶у皬 (榛樿 5 澶╋紝鏀寔 TS_MEAN5/TS_STD5)
            strict_date_mode: 涓ユ牸鏃ユ湡妯″紡
            
        Returns:
            feat_tensor: [Time=window, Assets, Features] 鏍煎紡鐨勫紶閲?
            asset_list: 璧勪骇浠ｇ爜鍒楄〃 (涓庡紶閲忕殑 Assets 缁村害瀵瑰簲)
        """
        # 1. 鑾峰彇鍘嗗彶浜ゆ槗鏃?
        trading_days = self.get_trading_days_before(date, window)
        if len(trading_days) < window:
            logger.warning(f"Insufficient history window: required={window}, actual={len(trading_days)}")
        
        if not trading_days:
            return torch.zeros((window, 0, len(ModelConfig.INPUT_FEATURES))), []
        
        # 2. 鑾峰彇澶氭棩 CB 鏁版嵁
        multi_day_df = self.get_cb_features_multi_days(trading_days)
        if multi_day_df.empty:
            return torch.zeros((window, 0, len(ModelConfig.INPUT_FEATURES))), []
        
        # 3. 鑾峰彇鎵€鏈夋棩鏈熼兘鏈夌殑璧勪骇 (鍙栦氦闆?
        common_codes = None
        for d in trading_days:
            day_codes = set(multi_day_df[multi_day_df['trade_date'].astype(str) == d]['code'])
            if common_codes is None:
                common_codes = day_codes
            else:
                common_codes = common_codes & day_codes
        
        common_codes = sorted(list(common_codes)) if common_codes else []
        if not common_codes:
            logger.warning("No common assets across all history days")
            return torch.zeros((window, 0, len(ModelConfig.INPUT_FEATURES))), []
        
        # 4. 濡傛灉鏈夊疄鏃舵暟鎹紝楠岃瘉鏃ユ湡骞跺噯澶囪鐩?
        qmt_data = {}
        if not realtime_quotes.empty:
            qmt_date = str(realtime_quotes.iloc[0].get('trade_date', ''))[:10]
            if qmt_date and qmt_date != trading_days[-1]:
                msg = f"QMT date ({qmt_date}) != target date ({trading_days[-1]})"
                if strict_date_mode:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
            
            # 鏋勫缓 QMT 鏁版嵁瀛楀吀 {code: {col: value}}
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
        
        # 5. 鏋勫缓 [Time, Assets, Features] 寮犻噺
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
                    
                    # 鑾峰彇鍊?
                    value = float(row.get(db_col, 0)) if db_col in day_df.columns else 0.0
                    
                    # 鏈€鍚庝竴澶╃敤 QMT 鏁版嵁瑕嗙洊 (濡傛湁)
                    if is_last_day and code in qmt_data:
                        qmt_col = db_col.lower()
                        if qmt_col in qmt_data[code] and qmt_data[code][qmt_col] > 0:
                            value = qmt_data[code][qmt_col]
                    
                    feat_array[t_idx, a_idx, f_idx] = value
        
        # 澶勭悊 NaN
        feat_array = np.nan_to_num(feat_array, nan=0.0)
        feat_tensor = torch.tensor(feat_array, dtype=torch.float32, device=ModelConfig.DEVICE)
        
        logger.info(f"Built feat_tensor with history: {feat_tensor.shape} (days={n_days}, assets={n_assets}, features={n_features})")
        return feat_tensor, common_codes
    
    # =========================================================================
    # 鏁版嵁鏁村悎鎺ュ彛
    # =========================================================================
    
    def build_feat_tensor(
        self, 
        realtime_quotes: pd.DataFrame, 
        cb_features: pd.DataFrame,
        strict_date_mode: bool = False
    ) -> torch.Tensor:
        """
        鍚堝苟瀹炴椂琛屾儏涓?CB 鐗规€ф暟鎹紝鏋勫缓鍥犲瓙璁＄畻鎵€闇€鐨?feat_tensor
        
        Args:
            realtime_quotes: 瀹炴椂琛屾儏 DataFrame
            cb_features: CB 鐗规€?DataFrame
            strict_date_mode: 鑻ヤ负 True锛屾棩鏈熶笉涓€鑷存椂鎶涘嚭寮傚父锛涘惁鍒欎粎璀﹀憡
        """
        # 缁撴灉 DataFrame
        merged = cb_features.copy()
        
        # 1. 鏃ユ湡瀵归綈涓庡疄鏃舵暟鎹悎骞?
        if not realtime_quotes.empty:
            # 鏍￠獙鏃ユ湡
            qmt_date = str(realtime_quotes.iloc[0]['trade_date'])[:10]
            sql_date = str(cb_features.iloc[0]['trade_date'])[:10]
            if qmt_date != sql_date:
                msg = f"Data date mismatch: QMT={qmt_date}, SQL={sql_date}"
                if strict_date_mode:
                    raise ValueError(msg + " (strict_date_mode=True)")
                else:
                    logger.warning(msg + ". Please check data synchronization state.")

            # 鍚堝苟瀹炴椂琛屾儏 (瑕嗙洊 SQL 涓殑鍩虹浠锋牸鍒?
            merged = merged.merge(
                realtime_quotes[['code', 'open', 'high', 'low', 'close', 'vol']],
                on='code',
                how='left',
                suffixes=('_sql', '_qmt')
            )
            
            # 浼樺厛浣跨敤 QMT 鏁版嵁
            for col in ['open', 'high', 'low', 'close', 'vol']:
                qmt_col = f'{col}_qmt'
                sql_col = f'{col}_sql'
                if qmt_col in merged.columns:
                    merged[col] = merged[qmt_col].fillna(merged.get(sql_col, 0))
                    merged.drop(columns=[qmt_col, sql_col], inplace=True, errors='ignore')
        
        # 2. 鏋勫缓鐗瑰緛寮犻噺 (涓ユ牸鎸?INPUT_FEATURES 椤哄簭, 涓?StackVM 瀵归綈)
        # 鍒涘缓 InternalName -> db_col 鏄犲皠
        factor_map = {internal: db_col for internal, db_col, _ in ModelConfig.BASIC_FACTORS}
        
        feat_list = []
        for feature_name in ModelConfig.INPUT_FEATURES:
            # 浠?BASIC_FACTORS 鏌ユ壘瀵瑰簲鐨勬暟鎹簱鍒楀悕
            db_col = factor_map.get(feature_name)
            if db_col is None:
                logger.warning(f"Feature '{feature_name}' not found in BASIC_FACTORS, using 0 filling.")
                feat_list.append(np.zeros(len(merged)))
                continue
            
            # 鍋ュ．鎬у鐞? 澶勭悊鍙兘鍥犲埆鍚嶅鑷寸殑鍒楀悕鍙樺寲
            actual_col = db_col
            if actual_col not in merged.columns:
                # 灏濊瘯鏌ユ壘 internal_name (鍦?merged 涓彲鑳藉凡鏀逛负 internal_name)
                if feature_name in merged.columns:
                    actual_col = feature_name
                else:
                    logger.warning(f"Feature column '{db_col}' for '{feature_name}' not found, using 0 filling.")
                    feat_list.append(np.zeros(len(merged)))
                    continue
            
            values = merged[actual_col].values.astype(np.float32)
            # 濉厖 NaN
            values = np.nan_to_num(values, nan=0.0)
            feat_list.append(values)
        
        # [Features, Assets] -> [Assets, Features]
        feat_array = np.stack(feat_list, axis=0).T
        feat_tensor = torch.tensor(feat_array, dtype=torch.float32, device=ModelConfig.DEVICE)
        
        logger.info(f"Built feat_tensor: {feat_tensor.shape} (features: {len(ModelConfig.INPUT_FEATURES)})")
        return feat_tensor

    
    def get_asset_list(self, cb_features: pd.DataFrame) -> List[str]:
        """
        鑾峰彇璧勪骇浠ｇ爜鍒楄〃 (涓?feat_tensor 琛岄『搴忎竴鑷?
        
        Args:
            cb_features: CB 鐗规€?DataFrame
            
        Returns:
            List of asset codes
        """
        return cb_features['code'].tolist()
    
    def get_names_dict(self, cb_features: pd.DataFrame) -> Dict[str, str]:
        """
        鑾峰彇 code -> name 鏄犲皠瀛楀吀
        
        Args:
            cb_features: CB 鐗规€?DataFrame
            
        Returns:
            Dict mapping code to name
        """
        return dict(zip(cb_features['code'], cb_features['name']))
    
    def close(self):
        """关闭数据库连接。"""
        if self.sql_engine:
            self.sql_engine.dispose()
            logger.info("SQL database connection closed")



