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
from model_core.factors import FeatureEngineer
from model_core.features_registry import get_feature_spec, get_required_raw_feature_names

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
        从 Mini QMT 获取实时行情快照 (盘中 Tick 级数据)

        使用 xtdata.get_full_tick() 获取全推数据，字段映射:
        - lastPrice → close  (最新成交价，盘中实时更新)
        - open/high/low      (当日 OHLC)
        - volume → vol       (当日成交量)
        - amount             (当日成交额)
        - lastClose → pre_close (昨收价)
        - timetag → trade_date  (交易所时间戳)

        Args:
            code_list: 标的代码列表
            period: 保留参数以兼容旧签名，实际不使用

        Returns:
            DataFrame 列: [code, trade_date, open, high, low, close, vol, amount]
            与旧接口完全兼容，下游无需修改。
        """
        if self.xtdata is None:
            logger.warning("xtdata unavailable, returning empty DataFrame")
            return pd.DataFrame()

        if not code_list:
            return pd.DataFrame()

        # 获取全推 Tick 快照
        data = self.xtdata.get_full_tick(code_list)

        rows = []
        for code in code_list:
            tick = data.get(code, {})
            if not tick:
                continue

            last_price = tick.get('lastPrice', 0)
            # 跳过无有效价格的标的（可能未上市或已退市）
            if last_price <= 0:
                continue

            # 解析交易所时间戳为日期
            timetag = tick.get('timetag', '')
            if isinstance(timetag, (int, float)) and timetag > 0:
                # QMT timetag 通常是毫秒级时间戳
                try:
                    trade_date = pd.Timestamp(timetag, unit='ms').strftime('%Y-%m-%d')
                except Exception:
                    trade_date = datetime.now().strftime('%Y-%m-%d')
            elif isinstance(timetag, str) and len(timetag) >= 10:
                trade_date = datetime.strptime(timetag, "%Y%m%d %H:%M:%S").strftime("%Y-%m-%d")
            else:
                trade_date = datetime.now().strftime('%Y-%m-%d')

            rows.append({
                'code': code,
                'trade_date': trade_date,
                'open': float(tick.get('open', 0)),
                'high': float(tick.get('high', 0)),
                'low': float(tick.get('low', 0)),
                'close': float(last_price),           # lastPrice → close
                'vol': float(tick.get('volume', 0)),
                'amount': float(tick.get('amount', 0)),
            })

        if not rows:
            logger.warning("get_full_tick returned no valid ticks")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(
            f"Realtime tick snapshot: {len(df)} symbols, "
            f"trade_date={df['trade_date'].iloc[0]}"
        )
        return df

    def get_realtime_quotes_kline(self, code_list: List[str], period: str = '1d') -> pd.DataFrame:
        """
        [备用] 基于 K 线的行情获取（旧实现，盘中返回的是上一交易日数据）。
        仅在非交易时段或 get_full_tick 不可用时使用。

        Args:
            code_list: 标的代码列表
            period: 周期

        Returns:
            DataFrame with columns: [code, trade_date, open, high, low, close, vol, amount]
        """
        if self.xtdata is None:
            logger.warning("xtdata unavailable, returning empty DataFrame")
            return pd.DataFrame()

        data = self.xtdata.get_market_data_ex([], code_list, period=period)
        return self._format_qmt_data(data, code_list)

    def _format_qmt_data(self, data: Dict, code_list: List[str]) -> pd.DataFrame:
        """
        将 QMT get_market_data_ex 返回的数据格式化为统一 DataFrame

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

    def merge_live_ohlc_into_cb_features(
        self,
        cb_features: pd.DataFrame,
        realtime_quotes: pd.DataFrame,
        target_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Merge realtime OHLC into SQL cb_features for live mode.

        Rules:
        - Merge by `code`, keep SQL universe unchanged.
        - Only positive numeric realtime values override SQL values.
        - If realtime date mismatches target/sql date, log warning but continue.
        """
        if cb_features is None or cb_features.empty:
            return cb_features

        merged = cb_features.copy()
        if realtime_quotes is None or realtime_quotes.empty:
            return merged

        needed_cols = {"code", "open", "high", "low", "close"}
        if not needed_cols.issubset(set(realtime_quotes.columns)):
            missing = sorted(list(needed_cols.difference(set(realtime_quotes.columns))))
            logger.warning(f"Realtime quotes missing required columns, skip OHLC override: {missing}")
            return merged

        if target_date and "trade_date" in realtime_quotes.columns and not realtime_quotes.empty:
            raw_trade_date = str(realtime_quotes.iloc[0].get("trade_date", ""))
            qmt_date = pd.to_datetime(raw_trade_date, errors="coerce")
            qmt_date_str = qmt_date.strftime("%Y-%m-%d") if pd.notna(qmt_date) else raw_trade_date[:10]
            if qmt_date_str != target_date:
                logger.warning(
                    f"Realtime date mismatch: target={target_date}, realtime={qmt_date_str}"
                )

        realtime = realtime_quotes[["code", "open", "high", "low", "close"]].copy()
        for col in ["open", "high", "low", "close"]:
            realtime[col] = pd.to_numeric(realtime[col], errors="coerce")

        merged = merged.merge(realtime, on="code", how="left", suffixes=("_sql", "_rt"))

        stats = []
        for col in ["open", "high", "low", "close"]:
            rt_col = f"{col}_rt"
            sql_col = f"{col}_sql"
            if rt_col not in merged.columns:
                continue

            valid_rt = merged[rt_col].notna() & (merged[rt_col] > 0)
            override_count = int(valid_rt.sum())
            merged[col] = np.where(valid_rt, merged[rt_col], merged.get(sql_col))
            merged.drop(columns=[rt_col, sql_col], inplace=True, errors="ignore")
            stats.append(f"{col}={override_count}")

        if stats:
            logger.info(
                f"Live OHLC override done: rows={len(merged)}, " + ", ".join(stats)
            )
        return merged
    
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
    
    def _required_raw_features(self) -> List[str]:
        return list(get_required_raw_feature_names(ModelConfig.INPUT_FEATURES))

    def _get_raw_column_name(self, feature_name: str) -> Optional[str]:
        spec = get_feature_spec(feature_name)
        if spec is None or spec.kind != "raw":
            return None
        return spec.raw_column

    def _resolve_actual_raw_column(self, columns, feature_name: str, raw_column: Optional[str]) -> Optional[str]:
        if raw_column is not None and raw_column in columns:
            return raw_column
        if feature_name in columns:
            return feature_name
        return None

    def _build_raw_tensors_from_frame(self, frame: pd.DataFrame) -> Dict[str, torch.Tensor]:
        raw_tensors: Dict[str, torch.Tensor] = {}
        for feature_name in self._required_raw_features():
            raw_column = self._get_raw_column_name(feature_name)
            if raw_column is None:
                continue

            actual_col = self._resolve_actual_raw_column(frame.columns, feature_name, raw_column)
            if actual_col is None:
                logger.warning(
                    "Raw feature '%s' column '%s' not found in merged frame, using 0 filling.",
                    feature_name,
                    raw_column,
                )
                values = np.zeros(len(frame), dtype=np.float32)
            else:
                values = np.nan_to_num(frame[actual_col].values.astype(np.float32), nan=0.0)

            raw_tensors[feature_name] = torch.tensor(
                values.reshape(1, -1),
                dtype=torch.float32,
                device=ModelConfig.DEVICE,
            )

        return raw_tensors

    def _build_raw_tensors_from_history_panel(
        self,
        multi_day_df: pd.DataFrame,
        trading_days: List[str],
        common_codes: List[str],
        qmt_data: Dict[str, Dict[str, float]],
    ) -> Dict[str, torch.Tensor]:
        required_raw = self._required_raw_features()
        n_days = len(trading_days)
        n_assets = len(common_codes)
        raw_arrays = {
            feature_name: np.zeros((n_days, n_assets), dtype=np.float32)
            for feature_name in required_raw
        }
        qmt_override_map = {
            "OPEN": "open",
            "HIGH": "high",
            "CLOSE": "close",
            "VOL": "vol",
        }

        for t_idx, trade_date in enumerate(trading_days):
            day_df = multi_day_df[multi_day_df["trade_date"].astype(str) == trade_date]
            day_df = day_df.set_index("code")
            is_last_day = t_idx == n_days - 1

            for a_idx, code in enumerate(common_codes):
                if code not in day_df.index:
                    continue
                row = day_df.loc[code]

                for feature_name in required_raw:
                    raw_column = self._get_raw_column_name(feature_name)
                    if raw_column is None:
                        continue

                    actual_col = self._resolve_actual_raw_column(day_df.columns, feature_name, raw_column)
                    value = float(row.get(actual_col, 0.0)) if actual_col is not None else 0.0

                    qmt_key = qmt_override_map.get(feature_name)
                    if is_last_day and qmt_key and code in qmt_data and qmt_data[code].get(qmt_key, 0.0) > 0:
                        value = float(qmt_data[code][qmt_key])

                    raw_arrays[feature_name][t_idx, a_idx] = value

        return {
            feature_name: torch.tensor(values, dtype=torch.float32, device=ModelConfig.DEVICE)
            for feature_name, values in raw_arrays.items()
        }

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
        
        raw_tensors = self._build_raw_tensors_from_history_panel(
            multi_day_df=multi_day_df,
            trading_days=trading_days,
            common_codes=common_codes,
            qmt_data=qmt_data,
        )
        feat_tensor = FeatureEngineer.build_feature_tensor(
            raw_data=raw_tensors,
            feature_names=list(ModelConfig.INPUT_FEATURES),
            normalize=False,
            warmup_rows=0,
        )

        logger.info(
            f"Built feat_tensor with history: {feat_tensor.shape} "
            f"(days={len(trading_days)}, assets={len(common_codes)}, features={len(ModelConfig.INPUT_FEATURES)})"
        )
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
        
        raw_tensors = self._build_raw_tensors_from_frame(merged)
        feat_tensor = FeatureEngineer.build_feature_tensor(
            raw_data=raw_tensors,
            feature_names=list(ModelConfig.INPUT_FEATURES),
            normalize=False,
            warmup_rows=0,
        )[0]

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



