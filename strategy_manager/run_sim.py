#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_sim.py

Unified simulation entry with two modes:
1) live
2) strict_replay
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.realtime_provider import RealtimeDataProvider
from model_core.config_loader import get_loaded_config_path, load_config
from strategy_manager.multi_sim_runner import MultiSimRunner
from strategy_manager.sim_runner import SimulationRunner
from workflow.bundle_loader import load_strategy_bundle


def _ensure_console_utf8() -> None:
    """Best effort: force utf-8 output on Windows terminals."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def setup_logging(log_dir: Optional[str] = None):
    """Configure file + console logging."""
    log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"sim_{datetime.now().strftime('%Y%m%d')}.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return logging.getLogger(__name__)


def _dataset_by_mode(mode: str) -> str:
    return "live" if mode == "live" else "replay"


def _reset_runner_state(runner: SimulationRunner) -> None:
    """Reset in-memory and persisted state for one strategy runner."""
    logger = logging.getLogger(__name__)

    runner.portfolio.clear_all()
    runner.nav_tracker.reset()

    if hasattr(runner, "trader") and hasattr(runner.trader, "trade_history"):
        runner.trader.trade_history = []
        if hasattr(runner.trader, "save_history"):
            runner.trader.save_history()

    strategy_dir = os.path.join(runner.PORTFOLIO_BASE_DIR, runner.strategy_id)
    candidates_path = os.path.join(strategy_dir, "candidates_history.json")
    if os.path.exists(candidates_path):
        try:
            os.remove(candidates_path)
        except Exception as e:
            logger.warning(f"Failed to clear candidates history: {candidates_path}, error={e}")

    if getattr(runner, "sql_state_store", None) is not None:
        runner.sql_state_store.reset_strategy(runner.strategy_id)
        runner._hydrate_state_from_sql()

    logger.info(f"[state reset] strategy={runner.strategy_id}")


def _prepare_runner_state_for_single_day_replay(
    runner: SimulationRunner,
    trade_date: str,
) -> None:
    """Prepare one-day strict_replay state: clear only this date and hydrate from history."""
    logger = logging.getLogger(__name__)

    if getattr(runner, "sql_state_store", None) is None:
        logger.info(
            f"[state prepare] strategy={runner.strategy_id}, date={trade_date}, "
            "sql state backend not enabled, keep in-memory state as-is"
        )
        return

    runner.sql_state_store.reset_strategy_date(runner.strategy_id, trade_date)
    runner._hydrate_state_from_sql(as_of_date=trade_date)
    logger.info(
        f"[state prepare] strategy={runner.strategy_id}, date={trade_date}, "
        "reset current date rows and hydrate as-of date"
    )


def _build_runner(
    data_provider: RealtimeDataProvider,
    config_path: Optional[str],
    strategy_id: Optional[str],
    state_backend: Optional[str],
    mode: str,
    live_quote_source: str,
    replay_source_override: Optional[str],
    strict_start_date: Optional[str] = None,
    strict_end_date: Optional[str] = None,
    strict_anchor_date: Optional[str] = None,
) -> MultiSimRunner:
    strategy_ids = [strategy_id] if strategy_id else None
    return MultiSimRunner(
        data_provider=data_provider,
        config_path=config_path,
        strategy_ids=strategy_ids,
        state_backend=state_backend,
        dataset=_dataset_by_mode(mode),
        live_quote_source=live_quote_source,
        replay_source_override=replay_source_override,
        strict_start_date=strict_start_date,
        strict_end_date=strict_end_date,
        strict_anchor_date=strict_anchor_date,
    )


def run_once(
    mode: str,
    date: Optional[str] = None,
    config_path: Optional[str] = None,
    strategy_id: Optional[str] = None,
    state_backend: Optional[str] = None,
    live_quote_source: str = "dummy",
    replay_source_override: Optional[str] = None,
) -> Dict:
    """Run one trading date."""
    logger = logging.getLogger(__name__)
    trade_date = date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"========== Simulation Run {trade_date} mode={mode} ==========")

    data_provider = RealtimeDataProvider()
    try:
        strict_start_date = None
        strict_end_date = None
        strict_anchor_date = trade_date
        if mode == "strict_replay":
            strict_end_date = trade_date
            strict_start_date = _get_warmup_start_date(data_provider, trade_date, warmup_days=65)
            logger.info(
                f"strict replay load window resolved: [{strict_start_date}, {strict_end_date}]"
            )

        runner = _build_runner(
            data_provider=data_provider,
            config_path=config_path,
            strategy_id=strategy_id,
            state_backend=state_backend,
            mode=mode,
            live_quote_source=live_quote_source,
            replay_source_override=replay_source_override,
            strict_start_date=strict_start_date,
            strict_end_date=strict_end_date,
            strict_anchor_date=strict_anchor_date,
        )

        # strict_replay single-day run: clear only the target date and continue from historical SQL state.
        if mode == "strict_replay":
            for r in runner.runners.values():
                _prepare_runner_state_for_single_day_replay(r, trade_date)

        results = runner.run_all_strategies(trade_date, mode=mode)
        runner.print_summary()
        return {"status": "success", "date": trade_date, "results": results}
    except Exception as e:
        logger.exception(f"Run failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        data_provider.close()


def _get_trading_dates(
    data_provider: RealtimeDataProvider,
    start_date: str,
    end_date: str,
) -> List[str]:
    """Fetch trading dates from SQL in [start_date, end_date]."""
    query = text(
        """
        SELECT DISTINCT trade_date
        FROM CB_DATA
        WHERE trade_date >= :start_date AND trade_date <= :end_date
        ORDER BY trade_date
        """
    )
    with data_provider.sql_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"start_date": start_date, "end_date": end_date})
    if df.empty:
        return []
    return [str(d)[:10] for d in df["trade_date"].tolist()]


def _get_warmup_start_date(
    data_provider: RealtimeDataProvider,
    anchor_date: str,
    warmup_days: int = 65,
) -> str:
    """
    Return the earliest date among latest `warmup_days` trading days up to anchor_date.
    """
    query = text(
        """
        SELECT trade_date
        FROM (
            SELECT DISTINCT trade_date
            FROM CB_DATA
            WHERE trade_date <= :anchor_date
            ORDER BY trade_date DESC
            LIMIT :limit_days
        ) t
        ORDER BY trade_date ASC
        """
    )
    with data_provider.sql_engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "anchor_date": anchor_date,
                "limit_days": int(warmup_days),
            },
        )
    if df.empty:
        return anchor_date
    return str(df["trade_date"].iloc[0])[:10]


def run_range(
    mode: str,
    start_date: str,
    end_date: str,
    config_path: Optional[str] = None,
    strategy_id: Optional[str] = None,
    state_backend: Optional[str] = None,
    live_quote_source: str = "dummy",
    replay_source_override: Optional[str] = None,
) -> Dict:
    """Run range replay. Only supports strict_replay mode."""
    logger = logging.getLogger(__name__)

    if mode != "strict_replay":
        raise ValueError("Range replay only supports mode=strict_replay")

    logger.info(f"========== Range Replay {start_date} ~ {end_date} mode={mode} ==========")
    if start_date > end_date:
        raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")

    data_provider = RealtimeDataProvider()
    try:
        trade_dates = _get_trading_dates(data_provider, start_date, end_date)
        if not trade_dates:
            logger.warning(f"No trading dates in range: {start_date} ~ {end_date}")
            return {"status": "no_data", "start_date": start_date, "end_date": end_date}

        logger.info(f"Found {len(trade_dates)} trading dates, replay will run sequentially")
        strict_start_date = _get_warmup_start_date(data_provider, trade_dates[0], warmup_days=65)
        strict_end_date = trade_dates[-1]
        strict_anchor_date = trade_dates[0]
        logger.info(
            f"strict replay load window resolved: [{strict_start_date}, {strict_end_date}]"
        )
        runner = _build_runner(
            data_provider=data_provider,
            config_path=config_path,
            strategy_id=strategy_id,
            state_backend=state_backend,
            mode=mode,
            live_quote_source=live_quote_source,
            replay_source_override=replay_source_override,
            strict_start_date=strict_start_date,
            strict_end_date=strict_end_date,
            strict_anchor_date=strict_anchor_date,
        )

        for r in runner.runners.values():
            _reset_runner_state(r)

        all_results: Dict[str, Dict] = {}
        for i, d in enumerate(trade_dates, 1):
            logger.info(f"[{i}/{len(trade_dates)}] Replay {d}")
            all_results[d] = runner.run_all_strategies(d, mode=mode)

        runner.print_summary()
        return {
            "status": "success",
            "start_date": start_date,
            "end_date": end_date,
            "days": len(trade_dates),
            "results": all_results,
        }
    except Exception as e:
        logger.exception(f"Range replay failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        data_provider.close()


def wait_until(target_hour: int, target_minute: int) -> None:
    """Sleep until target wall-clock time."""
    logger = logging.getLogger(__name__)
    while True:
        now = datetime.now()
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info(
            f"Waiting until {target.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"remaining {wait_seconds/60:.1f} minutes"
        )
        if wait_seconds > 60:
            time.sleep(60)
        else:
            time.sleep(wait_seconds)
            return


def run_scheduled(
    mode: str,
    target_hour: int = 14,
    target_minute: int = 50,
    config_path: Optional[str] = None,
    strategy_id: Optional[str] = None,
    state_backend: Optional[str] = None,
    live_quote_source: str = "dummy",
    replay_source_override: Optional[str] = None,
) -> None:
    """Optional internal scheduler, mostly for compatibility."""
    logger = logging.getLogger(__name__)
    logger.info(
        f"Scheduled mode started, daily at {target_hour}:{target_minute:02d}, mode={mode}"
    )
    while True:
        wait_until(target_hour, target_minute)
        run_once(
            mode=mode,
            date=None,
            config_path=config_path,
            strategy_id=strategy_id,
            state_backend=state_backend,
            live_quote_source=live_quote_source,
            replay_source_override=replay_source_override,
        )
        time.sleep(60)


def parse_args():
    parser = argparse.ArgumentParser(description="CB simulation runner (live/strict_replay)")
    parser.add_argument("--mode", type=str, default="strict_replay", choices=["live", "strict_replay"])
    parser.add_argument("--schedule", action="store_true", help="run in scheduled mode")
    parser.add_argument(
        "--strategies-config",
        type=str,
        default=None,
        help="path to strategy config json (default: strategy_manager/strategies_config.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="model_core yaml config path (default: model_core/default_config.yaml)",
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="strategy bundle path generated by workflow.bundle_builder; when set, sim runs the bundled single strategy",
    )
    parser.add_argument("--strategy-id", type=str, default=None, help="run only one strategy_id from config")
    parser.add_argument("--date", type=str, default=None, help="run date: YYYY-MM-DD")
    parser.add_argument("--start-date", type=str, default=None, help="replay start date")
    parser.add_argument("--end-date", type=str, default=None, help="replay end date")
    parser.add_argument(
        "--state-backend",
        type=str,
        default=None,
        choices=["sql", "json"],
        help="override state backend for all selected strategies",
    )
    parser.add_argument(
        "--live-quote-source",
        type=str,
        default="dummy",
        choices=["dummy", "qmt"],
        help="live quote source",
    )

    # Backward-compatible alias
    parser.add_argument("--replay-strict", action="store_true", help="deprecated alias, no longer needed")
    parser.add_argument(
        "--replay-source",
        type=str,
        default=None,
        choices=["sql_eod", "parquet"],
        help="override strategy replay_source for this run",
    )

    parser.add_argument("--hour", type=int, default=14, help="schedule hour")
    parser.add_argument("--minute", type=int, default=50, help="schedule minute")
    return parser.parse_args()


def _resolve_bundle_runtime(args, logger):
    bundle_info = None
    effective_model_config = args.config
    effective_strategies_config = args.strategies_config
    effective_strategy_id = args.strategy_id

    if args.bundle:
        bundle_info = load_strategy_bundle(args.bundle)
        bundle_strategy_id = bundle_info.get("strategy_id")
        if effective_strategy_id and bundle_strategy_id and effective_strategy_id != bundle_strategy_id:
            raise ValueError(
                f"--strategy-id={effective_strategy_id} 与 bundle.strategy_id={bundle_strategy_id} 不一致"
            )

        if effective_model_config:
            logger.info(
                "Bundle model config is ignored because --config was explicitly provided: bundle=%s, cli=%s",
                bundle_info["model_config_path"],
                effective_model_config,
            )
        else:
            effective_model_config = bundle_info["model_config_path"]

        generated_strategy_config = bundle_info.get("generated_strategy_config_path")
        if effective_strategies_config:
            logger.info("--bundle is set; --strategies-config will be ignored")
        elif not generated_strategy_config:
            raise FileNotFoundError(
                f"bundle 目录中缺少 generated_strategy_config.json: {bundle_info['bundle_dir']}"
            )
        effective_strategies_config = generated_strategy_config
        effective_strategy_id = bundle_strategy_id or effective_strategy_id

    return bundle_info, effective_model_config, effective_strategies_config, effective_strategy_id


def main():
    _ensure_console_utf8()
    args = parse_args()
    logger = setup_logging()
    bundle_info, effective_model_config, effective_strategies_config, effective_strategy_id = _resolve_bundle_runtime(
        args,
        logger,
    )
    load_config(effective_model_config)
    logger.info(
        "Loaded model config for sim: %s",
        get_loaded_config_path() or "model_core/default_config.yaml",
    )

    # No CLI args: use the requested default profile for daily live run.
    if len(sys.argv) == 1:
        args.mode = "live"
        args.state_backend = "sql"
        args.live_quote_source = "dummy"
        logger.info(
            "No CLI args detected, apply defaults: "
            "--mode live --state-backend sql --live-quote-source dummy"
        )

    mode = args.mode
    if args.replay_strict:
        mode = "strict_replay"
    if args.replay_source:
        logger.info(f"Use replay_source override: {args.replay_source}")

    if mode == "strict_replay" and args.replay_source and args.replay_source != "sql_eod":
        raise ValueError("strict_replay 目前仅允许 replay_source=sql_eod")

    logger.info("=" * 60)
    logger.info("CB simulation started")
    logger.info(
        f"mode={mode}, "
        f"config={effective_strategies_config or 'strategy_manager/strategies_config.json'}, "
        f"strategy_id={effective_strategy_id or 'ALL'}, "
        f"state_backend_override={args.state_backend or 'None'}, "
        f"live_quote_source={args.live_quote_source}"
    )
    if bundle_info:
        logger.info("Bundle mode enabled: bundle=%s", bundle_info["bundle_path"])
    logger.info("=" * 60)

    if args.schedule:
        run_scheduled(
            mode=mode,
            target_hour=args.hour,
            target_minute=args.minute,
            config_path=effective_strategies_config,
            strategy_id=effective_strategy_id,
            state_backend=args.state_backend,
            live_quote_source=args.live_quote_source,
            replay_source_override=args.replay_source,
        )
        return

    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise ValueError("Range replay requires both --start-date and --end-date")
        run_range(
            mode=mode,
            start_date=args.start_date,
            end_date=args.end_date,
            config_path=effective_strategies_config,
            strategy_id=effective_strategy_id,
            state_backend=args.state_backend,
            live_quote_source=args.live_quote_source,
            replay_source_override=args.replay_source,
        )
        return

    run_once(
        mode=mode,
        date=args.date,
        config_path=effective_strategies_config,
        strategy_id=effective_strategy_id,
        state_backend=args.state_backend,
        live_quote_source=args.live_quote_source,
        replay_source_override=args.replay_source,
    )


if __name__ == "__main__":
    main()
