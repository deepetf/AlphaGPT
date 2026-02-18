#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a mock "today" cross-section in CB_DATA for live-path testing.

Design goals:
1) No changes to run_sim code path.
2) Reuse historical CB_DATA rows as anchor.
3) Produce internally consistent OHLCV and reasonably stable factor columns.
4) Safe by default: dry-run only unless --write is provided.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from data_pipeline.config import Config
from model_core.config import ModelConfig


def _valid_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _get_columns_meta(engine: Engine, table: str) -> pd.DataFrame:
    q = text(f"SHOW COLUMNS FROM {table}")
    with engine.connect() as conn:
        meta = pd.read_sql(q, conn)
    if meta.empty:
        raise RuntimeError(f"Table has no columns: {table}")
    return meta


def _mysql_base_type(type_text: str) -> str:
    t = str(type_text or "").strip().lower()
    if "(" in t:
        t = t.split("(", 1)[0]
    return t


def _resolve_anchor_date(engine: Engine, table: str, target_date: str, anchor_date: Optional[str]) -> str:
    if anchor_date:
        q = text(
            f"""
            SELECT COUNT(*) AS cnt
            FROM {table}
            WHERE trade_date = :d
            """
        )
        with engine.connect() as conn:
            cnt = int(pd.read_sql(q, conn, params={"d": anchor_date}).iloc[0]["cnt"])
        if cnt <= 0:
            raise RuntimeError(f"No rows found for anchor_date={anchor_date} in {table}")
        return anchor_date

    q = text(
        f"""
        SELECT MAX(trade_date) AS d
        FROM {table}
        WHERE trade_date < :target_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"target_date": target_date})
    if df.empty or pd.isna(df.iloc[0]["d"]):
        raise RuntimeError(
            f"Cannot resolve anchor date before target_date={target_date}. "
            "Please provide --anchor-date explicitly."
        )
    return str(df.iloc[0]["d"])[:10]


def _load_anchor_rows(engine: Engine, table: str, anchor_date: str) -> pd.DataFrame:
    q = text(
        f"""
        SELECT *
        FROM {table}
        WHERE trade_date = :d
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"d": anchor_date})
    if df.empty:
        raise RuntimeError(f"No rows found for anchor_date={anchor_date} in {table}")
    return df


def _detect_pct_unit(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return 1.0
    med_abs = float(x.abs().median())
    return 100.0 if med_abs > 1.0 else 1.0


def _clip_positive(x: pd.Series, floor: float = 0.0) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan).fillna(floor)
    return y.clip(lower=floor)


def _find_col_case_insensitive(columns: List[str], target: str) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    return lower.get(target.lower())


def _safe_numeric(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return y


def _format_timedelta_to_hms(value) -> Optional[str]:
    if pd.isna(value):
        return None

    td = pd.to_timedelta(value, errors="coerce")
    if pd.isna(td):
        return str(value)

    total_seconds = int(td.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{sign}{hh:02d}:{mm:02d}:{ss:02d}"


def _normalize_df_for_mysql_insert(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    insert_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    type_map = {str(r["Field"]): _mysql_base_type(r["Type"]) for _, r in meta.iterrows()}

    # Generic guard: any timedelta dtype should be converted to HH:MM:SS.
    for c in insert_cols:
        if c not in out.columns:
            continue
        if pd.api.types.is_timedelta64_dtype(out[c]):
            out[c] = out[c].apply(_format_timedelta_to_hms)

    # Type-aware normalization by MySQL column type.
    for c in insert_cols:
        if c not in out.columns:
            continue
        base_t = type_map.get(c, "")
        s = out[c]

        if base_t == "time":
            out[c] = s.apply(_format_timedelta_to_hms)
            continue

        if base_t == "date":
            dt = pd.to_datetime(s, errors="coerce")
            out[c] = dt.dt.strftime("%Y-%m-%d")
            out.loc[dt.isna(), c] = None
            continue

        if base_t in {"datetime", "timestamp"}:
            dt = pd.to_datetime(s, errors="coerce")
            out[c] = dt
            continue

        if base_t in {"float", "double", "decimal"}:
            out[c] = pd.to_numeric(s, errors="coerce")
            continue

        if base_t in {"tinyint", "smallint", "mediumint", "int", "integer", "bigint"}:
            # Keep nullable integer behavior for NaN.
            out[c] = pd.to_numeric(s, errors="coerce")
            continue

    return out


def _generate_copy(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = target_date
    if "updated_at" in out.columns:
        out["updated_at"] = datetime.now()
    return out


def _generate_perturb(
    df: pd.DataFrame,
    target_date: str,
    anchor_date: str,
    seed: int,
    ret_sigma: float,
    open_noise_sigma: float,
    intraday_sigma: float,
    vol_sigma: float,
    factor_sigma: float,
) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)

    # Price base
    close0 = _clip_positive(out.get("close", pd.Series([0] * len(out))), floor=1.0)
    open0 = _clip_positive(out.get("open", close0), floor=1.0)
    high0 = _clip_positive(out.get("high", close0), floor=1.0)
    low0 = _clip_positive(out.get("low", close0), floor=1.0)

    # Daily return shock around anchor close.
    r = rng.normal(loc=0.0, scale=max(ret_sigma, 1e-6), size=len(out))
    r = np.clip(r, -0.12, 0.12)

    close1 = close0 * (1.0 + r)
    open1 = close0 * (1.0 + 0.35 * r + rng.normal(0.0, open_noise_sigma, size=len(out)))

    # Intraday range around open/close.
    up = np.abs(rng.normal(loc=intraday_sigma, scale=intraday_sigma * 0.5, size=len(out)))
    dn = np.abs(rng.normal(loc=intraday_sigma, scale=intraday_sigma * 0.5, size=len(out)))

    high1 = np.maximum(open1, close1) * (1.0 + up)
    low1 = np.minimum(open1, close1) * (1.0 - dn)

    # Ensure positive and OHLC ordering.
    low1 = np.maximum(low1, 0.5)
    open1 = np.maximum(open1, 0.5)
    close1 = np.maximum(close1, 0.5)
    high1 = np.maximum(high1, np.maximum(open1, close1))
    low1 = np.minimum(low1, np.minimum(open1, close1))

    out["open"] = open1
    out["high"] = high1
    out["low"] = low1
    out["close"] = close1

    # Volume/amount
    vol_mult = np.exp(rng.normal(0.0, max(vol_sigma, 1e-6), size=len(out)))
    vol_mult = np.clip(vol_mult, 0.25, 4.0)

    if "vol" in out.columns:
        vol0 = _clip_positive(out["vol"], floor=0.0)
        out["vol"] = vol0 * vol_mult

    if "amount" in out.columns:
        amt0 = _clip_positive(out["amount"], floor=0.0)
        px_ratio = np.divide(close1, close0, out=np.ones_like(close1), where=(close0 > 0))
        out["amount"] = amt0 * vol_mult * px_ratio

    # Percent-return style fields (unit auto-detect: 1 or 100).
    if "pct_chg" in out.columns:
        unit = _detect_pct_unit(out["pct_chg"])
        pct1 = ((close1 / close0) - 1.0) * unit
        out["pct_chg"] = pct1
    else:
        pct1 = (close1 / close0) - 1.0

    if "pct_chg_5" in out.columns:
        unit5 = _detect_pct_unit(out["pct_chg_5"])
        base = pd.to_numeric(out["pct_chg_5"], errors="coerce").fillna(0.0).to_numpy()
        out["pct_chg_5"] = 0.7 * base + 0.3 * (pct1 * 5.0 * (unit5 / max(_detect_pct_unit(pd.Series(pct1)), 1e-9)))

    if "pct_chg_stk" in out.columns:
        unit_s = _detect_pct_unit(out["pct_chg_stk"])
        stk_base = pd.to_numeric(out["pct_chg_stk"], errors="coerce").fillna(0.0).to_numpy()
        out["pct_chg_stk"] = 0.8 * stk_base + 0.2 * (pct1 * (unit_s / max(_detect_pct_unit(pd.Series(pct1)), 1e-9)))

    if "pct_chg_5_stk" in out.columns:
        unit_5s = _detect_pct_unit(out["pct_chg_5_stk"])
        stk5_base = pd.to_numeric(out["pct_chg_5_stk"], errors="coerce").fillna(0.0).to_numpy()
        out["pct_chg_5_stk"] = 0.8 * stk5_base + 0.2 * (pct1 * 5.0 * (unit_5s / max(_detect_pct_unit(pd.Series(pct1)), 1e-9)))

    if "alpha_pct_chg_5" in out.columns and "pct_chg_5" in out.columns and "pct_chg_5_stk" in out.columns:
        out["alpha_pct_chg_5"] = (
            pd.to_numeric(out["pct_chg_5"], errors="coerce").fillna(0.0)
            - pd.to_numeric(out["pct_chg_5_stk"], errors="coerce").fillna(0.0)
        )

    if "turnover" in out.columns:
        out["turnover"] = _clip_positive(out["turnover"], floor=0.0) * vol_mult

    if "left_years" in out.columns:
        days = (_parse_date(target_date) - _parse_date(anchor_date)).days
        ly = pd.to_numeric(out["left_years"], errors="coerce").fillna(0.0)
        out["left_years"] = np.maximum(ly - days / 365.0, 0.0)

    # Perturb factor columns used by strategy/VM so that ranking meaningfully changes.
    _perturb_feature_columns(
        out=out,
        rng=rng,
        close_ret=((close1 / close0) - 1.0),
        base_scale=max(factor_sigma, 1e-6),
    )

    out["trade_date"] = target_date
    if "updated_at" in out.columns:
        out["updated_at"] = datetime.now()
    return out


def _perturb_feature_columns(
    out: pd.DataFrame,
    rng: np.random.Generator,
    close_ret: np.ndarray,
    base_scale: float,
) -> None:
    """
    Perturb ModelConfig.BASIC_FACTORS columns with bounded, type-aware noise.

    Goal:
    1) keep data realistic enough for live-path tests
    2) force cross-sectional factor ranking to move vs anchor day
    """
    cols = list(out.columns)

    feature_cols: List[str] = []
    for _, db_col, _ in ModelConfig.BASIC_FACTORS:
        c = _find_col_case_insensitive(cols, db_col)
        if c and c not in feature_cols:
            feature_cols.append(c)

    skip = {
        _find_col_case_insensitive(cols, "trade_date"),
        _find_col_case_insensitive(cols, "code"),
        _find_col_case_insensitive(cols, "name"),
        _find_col_case_insensitive(cols, "open"),
        _find_col_case_insensitive(cols, "high"),
        _find_col_case_insensitive(cols, "low"),
        _find_col_case_insensitive(cols, "close"),
        _find_col_case_insensitive(cols, "vol"),
        _find_col_case_insensitive(cols, "amount"),
        _find_col_case_insensitive(cols, "pct_chg"),
        _find_col_case_insensitive(cols, "pct_chg_5"),
        _find_col_case_insensitive(cols, "pct_chg_stk"),
        _find_col_case_insensitive(cols, "pct_chg_5_stk"),
        _find_col_case_insensitive(cols, "alpha_pct_chg_5"),
        _find_col_case_insensitive(cols, "left_years"),
    }
    skip = {c for c in skip if c}

    n = len(out)
    mkt = close_ret
    idio = rng.normal(0.0, base_scale, size=n)
    idio2 = rng.normal(0.0, base_scale * 0.7, size=n)

    for c in feature_cols:
        if c in skip:
            continue
        s = _safe_numeric(out[c])
        if s.notna().sum() == 0:
            continue

        x = s.fillna(s.median() if s.notna().any() else 0.0).to_numpy(dtype=float)
        x_min = float(np.nanpercentile(x, 1))
        x_max = float(np.nanpercentile(x, 99))
        x_std = float(np.nanstd(x))
        col_lower = c.lower()

        # zscore-like columns: additive noise.
        if "zscore" in col_lower or col_lower.endswith("_z"):
            eps = rng.normal(0.0, max(0.15, base_scale * 6.0), size=n) + 0.5 * mkt
            x1 = x + eps
            x1 = np.clip(x1, x_min - 3.0 * max(1e-6, x_std), x_max + 3.0 * max(1e-6, x_std))
            out[c] = x1
            continue

        # Ratio-like positive fields.
        if col_lower in {"cap_mv_rate"}:
            mult = np.exp(0.25 * idio + 0.15 * idio2 + 0.25 * mkt)
            x1 = np.clip(x * mult, 0.0, max(2.5, x_max * 1.35))
            out[c] = x1
            continue

        # Premium / valuation style can be signed, use additive perturbation.
        if col_lower in {"conv_prem", "dblow", "pure_value"}:
            scale = max(1e-6, x_std)
            eps = (0.35 * idio + 0.2 * idio2 + 0.4 * mkt) * scale
            x1 = x + eps
            x1 = np.clip(x1, x_min - 2.0 * scale, x_max + 2.0 * scale)
            out[c] = x1
            continue

        # Volatility / IV / turnover / remain size / stock_vol60d
        if col_lower in {"volatility_stk", "iv", "turnover", "remain_size", "stock_vol60d"}:
            strength = 0.22 if col_lower in {"volatility_stk", "iv", "stock_vol60d"} else 0.14
            mult = np.exp(strength * idio + 0.25 * np.abs(mkt))
            x1 = np.clip(x * mult, max(0.0, x_min * 0.6), x_max * 1.8 + 1e-9)
            out[c] = x1
            continue

        # Generic numeric fallback: mild multiplicative + additive.
        if np.isfinite(x).all():
            mult = np.exp(0.10 * idio)
            x1 = x * mult + 0.08 * x_std * idio2
            x1 = np.clip(x1, x_min - 1.5 * x_std, x_max + 1.5 * x_std)
            out[c] = x1


def _validate_mock_frame(df: pd.DataFrame, table: str) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    required_cols = ["trade_date", "code", "open", "high", "low", "close", "vol"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns for {table}: {missing}")
        return False, issues

    if df.empty:
        issues.append("Generated frame is empty")
        return False, issues

    if df["code"].astype(str).duplicated().any():
        issues.append("Duplicate code rows found in generated frame")

    for c in ["open", "high", "low", "close", "vol"]:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isna().any():
            issues.append(f"Column {c} has NaN values")
        if not np.isfinite(x.fillna(0)).all():
            issues.append(f"Column {c} has non-finite values")

    o = pd.to_numeric(df["open"], errors="coerce").fillna(0.0)
    h = pd.to_numeric(df["high"], errors="coerce").fillna(0.0)
    l = pd.to_numeric(df["low"], errors="coerce").fillna(0.0)
    c = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    bad_ohlc = ~((l <= o) & (l <= c) & (h >= o) & (h >= c) & (h >= l))
    if bad_ohlc.any():
        issues.append(f"OHLC constraints violated in {int(bad_ohlc.sum())} rows")

    non_pos = ((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum()
    if non_pos > 0:
        issues.append(f"Non-positive OHLC values in {int(non_pos)} rows")

    return len(issues) == 0, issues


def _print_summary(
    anchor_df: pd.DataFrame,
    mock_df: pd.DataFrame,
    target_date: str,
    anchor_date: str,
    summary_all_features: bool,
) -> None:
    def _stats(x: pd.Series) -> str:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.empty:
            return "NA"
        return (
            f"min={x.min():.4f}, p50={x.quantile(0.5):.4f}, "
            f"p95={x.quantile(0.95):.4f}, max={x.max():.4f}"
        )

    print("=" * 80)
    print("Mock live slice summary")
    print(f"anchor_date={anchor_date}, target_date={target_date}, rows={len(mock_df)}")
    base_cols = ["open", "high", "low", "close", "vol", "amount", "pct_chg", "pct_chg_stk"]
    feature_cols: List[str] = []
    if summary_all_features:
        for _, db_col, _ in ModelConfig.BASIC_FACTORS:
            real_col = _find_col_case_insensitive(list(mock_df.columns), db_col)
            if real_col and real_col not in base_cols and real_col not in feature_cols:
                feature_cols.append(real_col)
    for c in base_cols + feature_cols:
        if c in mock_df.columns:
            print(f"[anchor] {c}: {_stats(anchor_df[c])}")
            print(f"[mock  ] {c}: {_stats(mock_df[c])}")
    print("=" * 80)


def _write_to_cb_data(
    engine: Engine,
    table: str,
    backup_table: str,
    target_date: str,
    df: pd.DataFrame,
    meta: pd.DataFrame,
    insert_cols: List[str],
) -> None:
    insert_df = _normalize_df_for_mysql_insert(df[insert_cols], meta, insert_cols)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE IF NOT EXISTS {backup_table} LIKE {table}"))
        conn.execute(text(f"DELETE FROM {backup_table} WHERE trade_date = :d"), {"d": target_date})
        conn.execute(
            text(f"INSERT INTO {backup_table} SELECT * FROM {table} WHERE trade_date = :d"),
            {"d": target_date},
        )
        conn.execute(text(f"DELETE FROM {table} WHERE trade_date = :d"), {"d": target_date})
        insert_df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=500)

        count = conn.execute(
            text(f"SELECT COUNT(*) AS cnt FROM {table} WHERE trade_date = :d"),
            {"d": target_date},
        ).mappings().first()["cnt"]
        print(f"Write completed: {table} trade_date={target_date}, row_count={count}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mock today slice in CB_DATA for live testing")
    p.add_argument("--table", type=str, default="CB_DATA", help="source/target table name")
    p.add_argument("--backup-table", type=str, default="CB_DATA_MOCK_BACKUP", help="backup table name")
    p.add_argument("--target-date", type=str, default=_today_str(), help="target date YYYY-MM-DD")
    p.add_argument("--anchor-date", type=str, default=None, help="anchor trading date YYYY-MM-DD")
    p.add_argument("--mode", type=str, default="perturb", choices=["copy", "perturb"], help="generation mode")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--ret-sigma", type=float, default=0.018, help="daily return shock sigma")
    p.add_argument("--open-noise-sigma", type=float, default=0.004, help="open noise sigma")
    p.add_argument("--intraday-sigma", type=float, default=0.006, help="intraday range sigma")
    p.add_argument("--vol-sigma", type=float, default=0.35, help="volume multiplier lognormal sigma")
    p.add_argument("--factor-sigma", type=float, default=0.08, help="feature perturbation base sigma")
    p.add_argument("--summary-all-features", action="store_true", help="print summary for all BASIC_FACTORS columns")
    p.add_argument("--write", action="store_true", help="actually write into CB_DATA")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not _valid_identifier(args.table):
        raise ValueError(f"Invalid table name: {args.table}")
    if not _valid_identifier(args.backup_table):
        raise ValueError(f"Invalid backup table name: {args.backup_table}")

    _parse_date(args.target_date)
    if args.anchor_date:
        _parse_date(args.anchor_date)

    engine = create_engine(Config.CB_DB_DSN)
    try:
        meta = _get_columns_meta(engine, args.table)
        field_col = "Field"
        extra_col = "Extra"
        all_cols = meta[field_col].tolist()
        auto_inc_cols = meta.loc[
            meta[extra_col].astype(str).str.contains("auto_increment", case=False, na=False),
            field_col,
        ].tolist()
        insert_cols = [c for c in all_cols if c not in auto_inc_cols]

        if "trade_date" not in all_cols:
            raise RuntimeError(f"Column trade_date not found in {args.table}")
        if "code" not in all_cols:
            raise RuntimeError(f"Column code not found in {args.table}")

        anchor_date = _resolve_anchor_date(engine, args.table, args.target_date, args.anchor_date)
        anchor_df = _load_anchor_rows(engine, args.table, anchor_date)

        if args.mode == "copy":
            mock_df = _generate_copy(anchor_df, args.target_date)
        else:
            mock_df = _generate_perturb(
                anchor_df,
                target_date=args.target_date,
                anchor_date=anchor_date,
                seed=args.seed,
                ret_sigma=args.ret_sigma,
                open_noise_sigma=args.open_noise_sigma,
                intraday_sigma=args.intraday_sigma,
                vol_sigma=args.vol_sigma,
                factor_sigma=args.factor_sigma,
            )

        ok, issues = _validate_mock_frame(mock_df, args.table)
        _print_summary(
            anchor_df,
            mock_df,
            args.target_date,
            anchor_date,
            summary_all_features=args.summary_all_features,
        )
        if not ok:
            raise RuntimeError("Validation failed: " + " | ".join(issues))

        if not args.write:
            print("Dry-run only. Add --write to insert into CB_DATA.")
            return

        _write_to_cb_data(
            engine=engine,
            table=args.table,
            backup_table=args.backup_table,
            target_date=args.target_date,
            df=mock_df,
            meta=meta,
            insert_cols=insert_cols,
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
