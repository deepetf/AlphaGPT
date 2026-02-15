"""
SQLStrictLoader

Load EOD data from SQL and build tensors with the same structure as CBDataLoader.
This loader is designed for strict replay alignment in sim_runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, inspect, text

from data_pipeline.config import Config
from model_core.config import ModelConfig, RobustConfig
from model_core.factors import FeatureEngineer


@dataclass
class _ColumnSpec:
    trade_date: str
    code: str
    name: Optional[str]
    factor_cols: Dict[str, str]  # internal_name -> sql_column


class SQLStrictLoader:
    """
    SQL EOD strict loader aligned with the CBDataLoader output contract.

    Exposed attributes:
    - raw_data_cache: Dict[str, Tensor[Time, Assets]]
    - feat_tensor: Tensor[Time, Assets, Features]
    - target_ret: Tensor[Time, Assets]
    - valid_mask: Tensor[Time, Assets]
    - split_idx: int
    - dates_list: List[str]
    - assets_list: List[str]
    - names_dict: Dict[str, str]
    """

    def __init__(
        self,
        sql_engine=None,
        table_name: str = "CB_DATA",
        start_date: str = "2022-08-01",
        end_date: Optional[str] = None,
    ):
        self.sql_engine = sql_engine or create_engine(Config.CB_DB_DSN)
        self.table_name = table_name
        self.start_date = start_date
        self.end_date = end_date

        self.raw_data_cache = None
        self.feat_tensor = None
        self.target_ret = None
        self.valid_mask = None
        self.split_idx = None

        self.dates_list: List[str] = []
        self.assets_list: List[str] = []
        self.names_dict: Dict[str, str] = {}

    def _resolve_columns(self) -> _ColumnSpec:
        inspector = inspect(self.sql_engine)
        columns = inspector.get_columns(self.table_name)
        if not columns:
            raise RuntimeError(f"No columns found in table: {self.table_name}")

        available = [c["name"] for c in columns]
        lower_map = {c.lower(): c for c in available}

        def resolve(required_name: str, allow_missing: bool = False) -> Optional[str]:
            if required_name in available:
                return required_name
            found = lower_map.get(required_name.lower())
            if found is not None:
                return found
            if allow_missing:
                return None
            raise RuntimeError(
                f"Required column '{required_name}' not found in {self.table_name}. "
                f"Available sample: {available[:20]}"
            )

        trade_date_col = resolve("trade_date")
        code_col = resolve("code")
        name_col = resolve("name", allow_missing=True)

        factor_cols: Dict[str, str] = {}
        for internal_name, db_col, _ in ModelConfig.BASIC_FACTORS:
            factor_cols[internal_name] = resolve(db_col)

        return _ColumnSpec(
            trade_date=trade_date_col,
            code=code_col,
            name=name_col,
            factor_cols=factor_cols,
        )

    def _load_sql_frame(self, cols: _ColumnSpec) -> pd.DataFrame:
        select_cols = [cols.trade_date, cols.code]
        if cols.name:
            select_cols.append(cols.name)
        select_cols.extend(cols.factor_cols.values())

        # Preserve order but deduplicate in case of same-name columns.
        seen = set()
        ordered_cols = []
        for c in select_cols:
            if c not in seen:
                seen.add(c)
                ordered_cols.append(c)

        cols_sql = ", ".join(ordered_cols)
        query = text(
            f"""
            SELECT {cols_sql}
            FROM {self.table_name}
            WHERE {cols.trade_date} >= :start_date
            AND (:end_date IS NULL OR {cols.trade_date} <= :end_date)
            ORDER BY {cols.trade_date}, {cols.code}
            """
        )

        with self.sql_engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                },
            )

        if df.empty:
            raise RuntimeError(
                "SQLStrictLoader got empty data from "
                f"{self.table_name} in range [{self.start_date}, {self.end_date or 'latest'}]"
            )

        # If SQL table has multiple rows for the same code/date, keep the last row.
        df = df.drop_duplicates(subset=[cols.trade_date, cols.code], keep="last")
        return df

    def load_data(self):
        cols = self._resolve_columns()
        df = self._load_sql_frame(cols)

        # Normalize key columns.
        df[cols.trade_date] = pd.to_datetime(df[cols.trade_date])
        df[cols.code] = df[cols.code].astype(str)

        # Pivot to [Time, Assets].
        assets = sorted(df[cols.code].unique().tolist())
        self.assets_list = assets

        pivot_df = df.pivot(index=cols.trade_date, columns=cols.code)
        pivot_df = pivot_df.sort_index(axis=0)

        self.dates_list = pivot_df.index.strftime("%Y-%m-%d").tolist()

        if cols.name and cols.name in df.columns:
            names = (
                df[[cols.code, cols.name]]
                .dropna(subset=[cols.code])
                .drop_duplicates(subset=[cols.code], keep="last")
            )
            names_map = dict(zip(names[cols.code].astype(str), names[cols.name].astype(str)))
            self.names_dict = {code: names_map.get(code, code) for code in assets}
        else:
            self.names_dict = {code: code for code in assets}

        raw_tensors: Dict[str, torch.Tensor] = {}
        for internal_name, _, fill_method in ModelConfig.BASIC_FACTORS:
            sql_col = cols.factor_cols[internal_name]
            sub_df = pivot_df[sql_col].copy()
            sub_df = sub_df.reindex(columns=assets)

            if fill_method == "ffill":
                sub_df = sub_df.ffill()
            elif fill_method == "zero":
                sub_df = sub_df.fillna(0.0)

            data_np = sub_df.fillna(0.0).values.astype(np.float32)
            raw_tensors[internal_name] = torch.tensor(data_np, device=ModelConfig.DEVICE)

        self.raw_data_cache = raw_tensors

        # Build valid mask exactly aligned with CBDataLoader.
        has_price = raw_tensors["CLOSE"] > 0
        is_trading = raw_tensors["VOL"] > 0
        not_expiring = raw_tensors["LEFT_YRS"] > 0.5
        self.valid_mask = has_price & is_trading & not_expiring

        # Build normalized features.
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # Build target return.
        close = raw_tensors["CLOSE"]
        ret_1d = (torch.roll(close, -1, dims=0) / (close + 1e-9)) - 1.0
        ret_1d[~self.valid_mask] = 0.0
        ret_1d[-1] = 0.0
        self.target_ret = ret_1d

        split_date = RobustConfig.TRAIN_TEST_SPLIT_DATE
        split_idx = 0
        used_default_split = False
        for i, d in enumerate(self.dates_list):
            if d >= split_date:
                split_idx = i
                break
        if split_idx == 0:
            split_idx = int(len(self.dates_list) * 0.8)
            used_default_split = True
        self.split_idx = split_idx

        if used_default_split:
            print(
                f"Warning: Split date '{split_date}' not found, using default 80% split"
            )
        print(
            f"SQL data ready. feat={self.feat_tensor.shape}, "
            f"valid={self.valid_mask.sum().item()}/{self.valid_mask.numel()}, "
            f"split_idx={self.split_idx}"
        )

    def close(self):
        if self.sql_engine is not None:
            self.sql_engine.dispose()
