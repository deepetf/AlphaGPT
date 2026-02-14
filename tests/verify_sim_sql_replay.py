#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify SQL replay outputs for sim_run.

Checks:
1) nav identity: nav == cash + holdings_value
2) holdings_count in nav matches real holdings table count
3) suspicious divergence days: holdings_value drops sharply while daily_ret is mild
4) optional candidate-vs-holdings comparison from candidates_history.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set

import pandas as pd
from sqlalchemy import create_engine, text

from data_pipeline.config import Config


@dataclass
class CheckResult:
    nav_days: int
    nav_identity_fail_days: int
    holdings_count_fail_days: int
    suspicious_divergence_days: int
    candidate_missing_days: int


def _load_nav(engine, strategy_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    sql = text(
        """
        SELECT trade_date, nav, cash, holdings_value, holdings_count, daily_ret, cum_ret, mdd
        FROM sim_nav_history
        WHERE strategy_id = :strategy_id
          AND trade_date >= :start_date
          AND trade_date <= :end_date
        ORDER BY trade_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"strategy_id": strategy_id, "start_date": start_date, "end_date": end_date},
        )
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    return df


def _load_holdings(engine, strategy_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    sql = text(
        """
        SELECT trade_date, code, shares, avg_cost, last_price, market_value
        FROM sim_daily_holdings
        WHERE strategy_id = :strategy_id
          AND trade_date >= :start_date
          AND trade_date <= :end_date
        ORDER BY trade_date, code
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"strategy_id": strategy_id, "start_date": start_date, "end_date": end_date},
        )
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    return df


def _group_holding_codes(holdings_df: pd.DataFrame) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    if holdings_df.empty:
        return out
    for d, sub in holdings_df.groupby("trade_date"):
        out[d] = set(sub["code"].astype(str).tolist())
    return out


def _load_candidate_topk(path: str, top_k: int) -> Dict[str, Set[str]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Set[str]] = {}
    for item in data:
        d = str(item.get("date", ""))[:10]
        cands = item.get("candidates", [])
        top_codes = [str(x.get("code")) for x in cands[:top_k] if x.get("code")]
        out[d] = set(top_codes)
    return out


def run_checks(
    strategy_id: str,
    start_date: str,
    end_date: str,
    nav_eps: float,
    drop_threshold: float,
    mild_ret_threshold: float,
    candidates_path: str,
    top_k: int,
) -> CheckResult:
    engine = create_engine(Config.CB_DB_DSN)
    nav = _load_nav(engine, strategy_id, start_date, end_date)
    holdings = _load_holdings(engine, strategy_id, start_date, end_date)

    if nav.empty:
        raise RuntimeError(
            f"No nav records in sim_nav_history for strategy={strategy_id}, {start_date}~{end_date}"
        )

    # 1) nav identity
    nav["nav_identity_diff"] = (nav["nav"] - (nav["cash"] + nav["holdings_value"])).abs()
    nav_identity_fail = nav[nav["nav_identity_diff"] > nav_eps].copy()

    # 2) holdings_count consistency
    hold_counts = (
        holdings.groupby("trade_date")["code"].count().rename("holdings_count_real").reset_index()
        if not holdings.empty
        else pd.DataFrame(columns=["trade_date", "holdings_count_real"])
    )
    merged = nav.merge(hold_counts, on="trade_date", how="left")
    merged["holdings_count_real"] = merged["holdings_count_real"].fillna(0).astype(int)
    merged["holdings_count_diff"] = merged["holdings_count"] - merged["holdings_count_real"]
    holdings_count_fail = merged[merged["holdings_count_diff"] != 0].copy()

    # 3) suspicious divergence days
    merged["holdings_value_chg_pct"] = merged["holdings_value"].pct_change().fillna(0.0)
    suspicious = merged[
        (merged["holdings_value_chg_pct"] <= -abs(drop_threshold))
        & (merged["daily_ret"].abs() <= abs(mild_ret_threshold))
    ].copy()

    # 4) candidate vs holdings
    candidate_topk = _load_candidate_topk(candidates_path, top_k=top_k)
    holding_codes = _group_holding_codes(holdings)
    candidate_missing_rows: List[Dict] = []
    if candidate_topk:
        for d, cand_codes in candidate_topk.items():
            if d < start_date or d > end_date:
                continue
            hold_codes = holding_codes.get(d, set())
            missing = sorted(cand_codes - hold_codes)
            if missing:
                candidate_missing_rows.append({"trade_date": d, "missing_codes": ",".join(missing)})

    artifacts_dir = os.path.join("tests", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    date_tag = f"{start_date}_{end_date}_{strategy_id}"
    merged.to_csv(os.path.join(artifacts_dir, f"sim_sql_nav_check_{date_tag}.csv"), index=False)
    nav_identity_fail.to_csv(
        os.path.join(artifacts_dir, f"sim_sql_nav_identity_fail_{date_tag}.csv"), index=False
    )
    holdings_count_fail.to_csv(
        os.path.join(artifacts_dir, f"sim_sql_holdings_count_fail_{date_tag}.csv"), index=False
    )
    suspicious.to_csv(
        os.path.join(artifacts_dir, f"sim_sql_suspicious_days_{date_tag}.csv"), index=False
    )
    pd.DataFrame(candidate_missing_rows).to_csv(
        os.path.join(artifacts_dir, f"sim_sql_candidate_missing_{date_tag}.csv"), index=False
    )

    result = CheckResult(
        nav_days=len(nav),
        nav_identity_fail_days=len(nav_identity_fail),
        holdings_count_fail_days=len(holdings_count_fail),
        suspicious_divergence_days=len(suspicious),
        candidate_missing_days=len(candidate_missing_rows),
    )

    report_path = os.path.join(artifacts_dir, f"sim_sql_replay_report_{date_tag}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Sim SQL Replay Report ({strategy_id})\n\n")
        f.write(f"- Range: `{start_date}` ~ `{end_date}`\n")
        f.write(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- nav_days: **{result.nav_days}**\n")
        f.write(f"- nav_identity_fail_days: **{result.nav_identity_fail_days}**\n")
        f.write(f"- holdings_count_fail_days: **{result.holdings_count_fail_days}**\n")
        f.write(f"- suspicious_divergence_days: **{result.suspicious_divergence_days}**\n")
        f.write(f"- candidate_missing_days: **{result.candidate_missing_days}**\n\n")
        f.write("## Artifacts\n\n")
        f.write(f"- `tests/artifacts/sim_sql_nav_check_{date_tag}.csv`\n")
        f.write(f"- `tests/artifacts/sim_sql_nav_identity_fail_{date_tag}.csv`\n")
        f.write(f"- `tests/artifacts/sim_sql_holdings_count_fail_{date_tag}.csv`\n")
        f.write(f"- `tests/artifacts/sim_sql_suspicious_days_{date_tag}.csv`\n")
        f.write(f"- `tests/artifacts/sim_sql_candidate_missing_{date_tag}.csv`\n")

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Verify sim_run SQL replay outputs")
    p.add_argument("--strategy-id", type=str, default="default")
    p.add_argument("--start-date", type=str, required=True)
    p.add_argument("--end-date", type=str, required=True)
    p.add_argument("--nav-eps", type=float, default=1e-4)
    p.add_argument("--drop-threshold", type=float, default=0.10, help="holdings_value drop pct threshold")
    p.add_argument("--mild-ret-threshold", type=float, default=0.02, help="abs(daily_ret) mild threshold")
    p.add_argument(
        "--candidates-path",
        type=str,
        default=os.path.join("execution", "portfolio", "default", "candidates_history.json"),
    )
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    result = run_checks(
        strategy_id=args.strategy_id,
        start_date=args.start_date,
        end_date=args.end_date,
        nav_eps=args.nav_eps,
        drop_threshold=args.drop_threshold,
        mild_ret_threshold=args.mild_ret_threshold,
        candidates_path=args.candidates_path,
        top_k=args.top_k,
    )

    print("=== Sim SQL Replay Check ===")
    print(f"range: {args.start_date} ~ {args.end_date}, strategy={args.strategy_id}")
    print(f"nav_days: {result.nav_days}")
    print(f"nav_identity_fail_days: {result.nav_identity_fail_days}")
    print(f"holdings_count_fail_days: {result.holdings_count_fail_days}")
    print(f"suspicious_divergence_days: {result.suspicious_divergence_days}")
    print(f"candidate_missing_days: {result.candidate_missing_days}")


if __name__ == "__main__":
    main()

