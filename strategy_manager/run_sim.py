#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_sim.py

统一以策略配置文件驱动模拟盘：
- 配置可包含 1 个或多个策略；
- 不再提供 `--multi` 参数；
- 策略隔离依赖配置中的 `strategy_id`。
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
from strategy_manager.multi_sim_runner import MultiSimRunner
from strategy_manager.sim_runner import SimulationRunner


def _ensure_console_utf8():
    """Best effort: force utf-8 output on Windows terminals."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def setup_logging(log_dir: Optional[str] = None):
    """配置日志。"""
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


def _reset_runner_state(runner: SimulationRunner):
    """重置策略状态（内存 + 持久化）。"""
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


def _build_runner(
    data_provider: RealtimeDataProvider,
    config_path: Optional[str],
    strategy_id: Optional[str],
    state_backend: Optional[str],
) -> MultiSimRunner:
    strategy_ids = [strategy_id] if strategy_id else None
    return MultiSimRunner(
        data_provider=data_provider,
        config_path=config_path,
        strategy_ids=strategy_ids,
        state_backend=state_backend,
    )


def run_once(
    date: Optional[str] = None,
    config_path: Optional[str] = None,
    strategy_id: Optional[str] = None,
    state_backend: Optional[str] = None,
) -> Dict:
    """运行单日（配置中的单个或多个策略）。"""
    logger = logging.getLogger(__name__)
    date = date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"========== Simulation Run {date} ==========")

    data_provider = RealtimeDataProvider()
    try:
        runner = _build_runner(data_provider, config_path, strategy_id, state_backend)

        if date != datetime.now().strftime("%Y-%m-%d"):
            for r in runner.runners.values():
                _reset_runner_state(r)

        results = runner.run_all_strategies(date)
        runner.print_summary()
        return {"status": "success", "date": date, "results": results}
    except Exception as e:
        logger.exception(f"Run failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        data_provider.close()


def _get_trading_dates(data_provider: RealtimeDataProvider, start_date: str, end_date: str) -> List[str]:
    """从 SQL 获取区间交易日。"""
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


def run_range(
    start_date: str,
    end_date: str,
    config_path: Optional[str] = None,
    strategy_id: Optional[str] = None,
    state_backend: Optional[str] = None,
) -> Dict:
    """区间回放（连续持仓，不每日重置）。"""
    logger = logging.getLogger(__name__)
    logger.info(f"========== Range Replay {start_date} ~ {end_date} ==========")

    if start_date > end_date:
        raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")

    data_provider = RealtimeDataProvider()
    try:
        trade_dates = _get_trading_dates(data_provider, start_date, end_date)
        if not trade_dates:
            logger.warning(f"No trading dates in range: {start_date} ~ {end_date}")
            return {"status": "no_data", "start_date": start_date, "end_date": end_date}

        logger.info(f"Found {len(trade_dates)} trading dates, replay will run sequentially")
        runner = _build_runner(data_provider, config_path, strategy_id, state_backend)

        for r in runner.runners.values():
            _reset_runner_state(r)

        all_results: Dict[str, Dict] = {}
        for i, d in enumerate(trade_dates, 1):
            logger.info(f"[{i}/{len(trade_dates)}] Replay {d}")
            all_results[d] = runner.run_all_strategies(d)

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


def wait_until(target_hour: int, target_minute: int):
    """等待到指定时间。"""
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
    target_hour: int = 14,
    target_minute: int = 50,
    config_path: Optional[str] = None,
    strategy_id: Optional[str] = None,
    state_backend: Optional[str] = None,
):
    """定时执行模拟盘。"""
    logger = logging.getLogger(__name__)
    logger.info(f"Scheduled mode started, daily at {target_hour}:{target_minute:02d}")
    while True:
        wait_until(target_hour, target_minute)
        run_once(
            date=None,
            config_path=config_path,
            strategy_id=strategy_id,
            state_backend=state_backend,
        )
        time.sleep(60)


def parse_args():
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="CB simulation runner (config-driven)")
    parser.add_argument("--schedule", action="store_true", help="run in scheduled mode")
    parser.add_argument(
        "--strategies-config",
        type=str,
        default=None,
        help="path to strategy config json (default: strategy_manager/strategies_config.json)",
    )
    parser.add_argument(
        "--strategy-id",
        type=str,
        default=None,
        help="run only one strategy_id from config",
    )
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
    parser.add_argument("--hour", type=int, default=14, help="schedule hour")
    parser.add_argument("--minute", type=int, default=50, help="schedule minute")
    return parser.parse_args()


def main():
    _ensure_console_utf8()
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 50)
    logger.info("CB simulation started (config-driven)")
    logger.info(
        f"config={args.strategies_config or 'strategy_manager/strategies_config.json'}, "
        f"strategy_id={args.strategy_id or 'ALL'}, "
        f"state_backend_override={args.state_backend or 'None'}"
    )
    logger.info("=" * 50)

    if args.schedule:
        run_scheduled(
            target_hour=args.hour,
            target_minute=args.minute,
            config_path=args.strategies_config,
            strategy_id=args.strategy_id,
            state_backend=args.state_backend,
        )
        return

    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise ValueError("Range replay requires both --start-date and --end-date")
        run_range(
            start_date=args.start_date,
            end_date=args.end_date,
            config_path=args.strategies_config,
            strategy_id=args.strategy_id,
            state_backend=args.state_backend,
        )
        return

    run_once(
        date=args.date,
        config_path=args.strategies_config,
        strategy_id=args.strategy_id,
        state_backend=args.state_backend,
    )


if __name__ == "__main__":
    main()
