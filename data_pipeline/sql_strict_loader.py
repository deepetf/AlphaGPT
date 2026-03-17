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
    factor_cols: Dict[str, Optional[str]]  # internal_name -> sql_column


class SQLStrictLoader:
    """
    SQL EOD strict loader aligned with the CBDataLoader output contract.

    Exposed attributes:
    - raw_data_cache: Dict[str, Tensor[Time, Assets]]
    - exec_raw_cache: Dict[str, Tensor[Time, Assets]]  # raw (no fill), for execution prices
    - feat_tensor: Tensor[Time, Assets, Features]
    - target_ret: Tensor[Time, Assets]
    - valid_mask: Tensor[Time, Assets]
    - present_mask: Tensor[Time, Assets]  # whether date-code exists in raw SQL rows
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
        warmup_anchor_date: Optional[str] = None,
    ):
        self.sql_engine = sql_engine or create_engine(Config.CB_DB_DSN)
        self.table_name = table_name
        self.start_date = start_date
        self.end_date = end_date
        self.warmup_anchor_date = warmup_anchor_date

        self.raw_data_cache = None
        self.exec_raw_cache = None
        self.feat_tensor = None
        self.target_ret = None
        self.valid_mask = None
        self.present_mask = None
        self.listed_mask = None
        self.data_mask = None
        self.tradable_mask = None
        self.cs_mask = None
        self.feature_valid_tensor = None
        self.split_idx = None

        self.dates_list: List[str] = []
        self.assets_list: List[str] = []
        self.names_dict: Dict[str, str] = {}

    def _resolve_warmup_rows(self) -> int:
        """Resolve warmup rows used by FeatureEngineer._robust_normalize."""
        if not self.dates_list:
            return 0
        if not self.warmup_anchor_date:
            return 0
        if self.warmup_anchor_date in self.dates_list:
            return self.dates_list.index(self.warmup_anchor_date)
        print(
            f"Warning: warmup_anchor_date '{self.warmup_anchor_date}' not in SQLStrictLoader dates "
            f"[{self.dates_list[0]}, {self.dates_list[-1]}], fallback warmup_rows=0"
        )
        return 0

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

        optional_raw = FeatureEngineer.get_optional_raw_feature_names()
        factor_cols: Dict[str, Optional[str]] = {}
        for internal_name, db_col, _ in ModelConfig.BASIC_FACTORS:
            allow_missing = internal_name in optional_raw
            factor_cols[internal_name] = resolve(db_col, allow_missing=allow_missing)

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
        select_cols.extend([c for c in cols.factor_cols.values() if c is not None])

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

        # Raw date-code presence mask (do not fill): used to align strict/live CS universe.
        present_df = (
            df.assign(__present__=1.0)
            .pivot(index=cols.trade_date, columns=cols.code, values="__present__")
            .sort_index(axis=0)
            .reindex(index=pivot_df.index, columns=assets)
            .fillna(0.0)
        )
        self.present_mask = torch.tensor(
            (present_df.values > 0.5),
            device=ModelConfig.DEVICE,
            dtype=torch.bool,
        )

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
        exec_raw_tensors: Dict[str, torch.Tensor] = {}
        for internal_name, _, fill_method in ModelConfig.BASIC_FACTORS:
            sql_col = cols.factor_cols[internal_name]
            if sql_col is None:
                print(
                    f"Warning: optional column for '{internal_name}' not found in SQL, skipping."
                )
                continue
            sub_df = pivot_df[sql_col].copy().reindex(columns=assets)
            sub_df_raw = sub_df.copy()

            if fill_method == "ffill":
                sub_df = sub_df.ffill()

            data_np = sub_df.values.astype(np.float32)
            raw_tensors[internal_name] = torch.tensor(data_np, device=ModelConfig.DEVICE)

            if internal_name in {"CLOSE", "OPEN", "HIGH"}:
                raw_np = sub_df_raw.values.astype(np.float32)
                exec_raw_tensors[internal_name] = torch.tensor(
                    raw_np, device=ModelConfig.DEVICE
                )

        self.raw_data_cache = raw_tensors
        self.exec_raw_cache = exec_raw_tensors

        close_raw_df = pivot_df[cols.factor_cols["CLOSE"]].copy().reindex(columns=assets)
        self.listed_mask = torch.tensor(
            close_raw_df.notna().values,
            device=ModelConfig.DEVICE,
            dtype=torch.bool,
        )
        self.data_mask = self.listed_mask & torch.isfinite(raw_tensors["CLOSE"])

        has_price = torch.isfinite(raw_tensors["CLOSE"]) & (raw_tensors["CLOSE"] > 0)
        is_trading = torch.isfinite(raw_tensors["VOL"]) & (raw_tensors["VOL"] > 0)
        not_expiring = torch.isfinite(raw_tensors["LEFT_YRS"]) & (raw_tensors["LEFT_YRS"] > 0.5)
        self.tradable_mask = self.listed_mask & has_price & is_trading & not_expiring
        self.valid_mask = self.tradable_mask
        self.cs_mask = self.tradable_mask

        # Build normalized features.
        warmup_rows = self._resolve_warmup_rows()
        self.feat_tensor, self.feature_valid_tensor = FeatureEngineer.compute_features(
            self.raw_data_cache,
            warmup_rows=warmup_rows,
            cross_sectional_mask=self.cs_mask,
            return_validity=True,
        )

        # Build target return.
        close = raw_tensors["CLOSE"]
        ret_1d = (torch.roll(close, -1, dims=0) / (close + 1e-9)) - 1.0
        ret_1d[~self.valid_mask] = 0.0
        ret_1d[~torch.isfinite(ret_1d)] = 0.0
        ret_1d[-1] = 0.0
        self.target_ret = ret_1d

        # NOTE:
        # SQLStrictLoader is used by sim/live strict replay path only.
        # Sim path does not consume train/val split, keep split_idx as a
        # compatibility placeholder to avoid emitting misleading warnings.
        self.split_idx = 0
        print(
            f"SQL data ready. feat={self.feat_tensor.shape}, "
            f"valid={self.valid_mask.sum().item()}/{self.valid_mask.numel()}, "
            f"present={self.present_mask.sum().item()}/{self.present_mask.numel()}, "
            f"split_idx={self.split_idx}, warmup_rows={warmup_rows}, "
            f"warmup_anchor_date={self.warmup_anchor_date or 'None'}"
        )

    def close(self):
        if self.sql_engine is not None:
            self.sql_engine.dispose()
