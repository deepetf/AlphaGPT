#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CB 妯℃嫙鐩樿繍琛屽叆鍙?(run_sim.py)

浣跨敤鏂瑰紡:
    # 鍗曠瓥鐣ユā寮?(浼犵粺)
    python run_sim.py
    
    # 澶氱瓥鐣ユā寮?(鏂?
    python run_sim.py --multi
    python run_sim.py --multi --strategies-config path/to/config.json
    
    # 瀹氭椂杩愯 (14:50 鑷姩瑙﹀彂)
    python run_sim.py --schedule
    python run_sim.py --schedule --multi
    
    # 鎸囧畾鏃ユ湡鍥炴斁
    python run_sim.py --date 2026-02-07

    # 鍖洪棿鍥炴斁锛堣繛缁寔浠擄紝涓嶆瘡鏃ラ噸缃級
    python run_sim.py --start-date 2025-12-01 --end-date 2025-12-31 --replay-strict --replay-source sql_eod
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import pandas as pd
from sqlalchemy import text

# 娣诲姞椤圭洰鏍圭洰褰曞埌 Python 璺緞
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.realtime_provider import RealtimeDataProvider
from strategy_manager.cb_portfolio import CBPortfolioManager
from strategy_manager.nav_tracker import NavTracker
from strategy_manager.sim_runner import SimulationRunner
from model_core.config import RobustConfig

# ===================== Utilities =====================

def _ensure_console_utf8():
    """Best effort: force utf-8 output on Windows terminals."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ===================== Logging =====================

def setup_logging(log_dir: str = None):
    """閰嶇疆鏃ュ織"""
    log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"sim_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 閰嶇疆鏍煎紡
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 鏂囦欢澶勭悊鍣?
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 鎺у埗鍙板鐞嗗櫒
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 鏍规棩蹇楀櫒
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def _reset_runner_state(runner: SimulationRunner):
    """Reset runner in-memory state and persistent state for replay."""
    logger = logging.getLogger(__name__)

    runner.portfolio.clear_all()
    runner.nav_tracker.reset()

    # Clear simulated trade history (append-only by default).
    if hasattr(runner, "trader") and hasattr(runner.trader, "trade_history"):
        runner.trader.trade_history = []
        if hasattr(runner.trader, "save_history"):
            runner.trader.save_history()

    # Clear candidate history for the same strategy id.
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


# ===================== 鏍稿績杩愯閫昏緫 =====================

def run_once(
    date: str = None,
    top_k: int = 10,
    take_profit: float = RobustConfig.TAKE_PROFIT,
    replay_strict: bool = False,
    replay_source: str = "sql_eod",
    state_backend: str = "sql",
):
    """Run one simulation day for single-strategy mode."""
    logger = logging.getLogger(__name__)
    date = date or datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"========== Simulation Run {date} ==========")
    
    # 鍒濆鍖栫粍浠?
    data_provider = RealtimeDataProvider()
    portfolio = CBPortfolioManager()
    nav_tracker = NavTracker(initial_capital=1_000_000.0)
    
    # 鍒濆鍖栬繍琛屽櫒
    runner = SimulationRunner(
        data_provider=data_provider,
        portfolio=portfolio,
        nav_tracker=nav_tracker,
        top_k=top_k,
        take_profit_ratio=take_profit,
        replay_strict=replay_strict,
        replay_source=replay_source,
        state_backend=state_backend,
    )

    # 鍘嗗彶鍥炴斁榛樿娓呯┖鐘舵€侊紝閬垮厤鍐欏叆閲嶅鎴愪氦璁板綍
    if date != datetime.now().strftime('%Y-%m-%d'):
        _reset_runner_state(runner)
    
    # 鎵ц
    try:
        result = runner.run_daily(date=date)
        logger.info(f"Run result: {result}")
        
        # 鎵撳嵃鎽樿
        logger.info(portfolio.summary())
        logger.info(nav_tracker.summary())
        
        return result
    except Exception as e:
        logger.exception(f"Run failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        data_provider.close()


def run_multi(date: str = None, config_path: str = None, state_backend: str = "sql"):
    """
    鎵ц澶氱瓥鐣ユā鎷?
    
    Args:
        date: 杩愯鏃ユ湡锛岄粯璁や负浠婂ぉ
        config_path: 绛栫暐閰嶇疆鏂囦欢璺緞
    """
    logger = logging.getLogger(__name__)
    date = date or datetime.now().strftime('%Y-%m-%d')
    
    # 寤惰繜瀵煎叆閬垮厤寰幆寮曠敤
    from strategy_manager.multi_sim_runner import MultiSimRunner
    
    logger.info(f"========== Multi-Strategy Run {date} ==========")
    
    # 鍒濆鍖栨暟鎹彁渚涜€?(鎵€鏈夌瓥鐣ュ叡浜?
    data_provider = RealtimeDataProvider()
    
    try:
        # 鍒濆鍖栧绛栫暐杩愯鍣?
        multi_runner = MultiSimRunner(
            data_provider=data_provider,
            config_path=config_path,
            state_backend=state_backend,
        )

        # Historical replay defaults to reset state before run.
        if date != datetime.now().strftime('%Y-%m-%d'):
            for runner in multi_runner.runners.values():
                _reset_runner_state(runner)
        
        # 杩愯鎵€鏈夌瓥鐣?
        results = multi_runner.run_all_strategies(date)
        
        # 鎵撳嵃姹囨€?
        multi_runner.print_summary()
        
        return results
    except Exception as e:
        logger.exception(f"Multi-strategy run failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        data_provider.close()


def _get_trading_dates(data_provider: RealtimeDataProvider, start_date: str, end_date: str) -> List[str]:
    """Get trading dates from SQL for [start_date, end_date]."""
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
    multi: bool = False,
    config_path: str = None,
    top_k: int = 10,
    take_profit: float = RobustConfig.TAKE_PROFIT,
    replay_strict: bool = False,
    replay_source: str = "sql_eod",
    state_backend: str = "sql",
):
    """
    鍖洪棿鍥炴斁锛堜繚鎸佽繛缁寔浠撶姸鎬侊級

    Args:
        start_date: 璧峰鏃ユ湡锛圷YYY-MM-DD锛?        end_date: 缁撴潫鏃ユ湡锛圷YYY-MM-DD锛?        multi: 鏄惁澶氱瓥鐣ユā寮?        config_path: 澶氱瓥鐣ラ厤缃矾寰?        top_k: 鍗曠瓥鐣ユ寔浠撴暟閲?        take_profit: 鍗曠瓥鐣ユ鐩堟瘮渚?        replay_strict: 鍗曠瓥鐣ヤ弗鏍煎洖鏀?        replay_source: 鍗曠瓥鐣ヤ弗鏍煎洖鏀炬暟鎹簮
    """
    logger = logging.getLogger(__name__)
    mode_str = "multi" if multi else "single"
    logger.info(f"========== Range Replay ({mode_str}) {start_date} ~ {end_date} ==========")

    if start_date > end_date:
        raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")

    data_provider = RealtimeDataProvider()
    try:
        trade_dates = _get_trading_dates(data_provider, start_date, end_date)
        if not trade_dates:
            logger.warning(f"No trading dates in range: {start_date} ~ {end_date}")
            return {"status": "no_data", "start_date": start_date, "end_date": end_date}

        logger.info(f"Found {len(trade_dates)} trading dates, replay will run sequentially")

        if multi:
            # 寤惰繜瀵煎叆閬垮厤寰幆寮曠敤
            from strategy_manager.multi_sim_runner import MultiSimRunner

            multi_runner = MultiSimRunner(
                data_provider=data_provider,
                config_path=config_path,
                state_backend=state_backend,
            )

            # 鍖洪棿鍥炴斁榛樿娓呯┖鐘舵€侊紝闃叉绱鍐欏叆
            for runner in multi_runner.runners.values():
                _reset_runner_state(runner)

            all_results = {}
            for i, d in enumerate(trade_dates, 1):
                logger.info(f"[{i}/{len(trade_dates)}] Replay {d}")
                all_results[d] = multi_runner.run_all_strategies(d)

            multi_runner.print_summary()
            return {
                "status": "success",
                "start_date": start_date,
                "end_date": end_date,
                "days": len(trade_dates),
                "results": all_results,
            }

        # single strategy: reuse one runner to keep continuous portfolio state
        portfolio = CBPortfolioManager()
        nav_tracker = NavTracker(initial_capital=1_000_000.0)
        runner = SimulationRunner(
            data_provider=data_provider,
            portfolio=portfolio,
            nav_tracker=nav_tracker,
            top_k=top_k,
            take_profit_ratio=take_profit,
            replay_strict=replay_strict,
            replay_source=replay_source,
            state_backend=state_backend,
        )

        # 鍖洪棿鍥炴斁榛樿娓呯┖鐘舵€侊紝闃叉绱鍐欏叆
        _reset_runner_state(runner)

        all_results = {}
        for i, d in enumerate(trade_dates, 1):
            logger.info(f"[{i}/{len(trade_dates)}] Replay {d}")
            all_results[d] = runner.run_daily(d)

        logger.info(portfolio.summary())
        logger.info(nav_tracker.summary())
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
    """
    绛夊緟鍒版寚瀹氭椂闂?
    
    Args:
        target_hour: 鐩爣灏忔椂
        target_minute: 鐩爣鍒嗛挓
    """
    logger = logging.getLogger(__name__)
    
    while True:
        now = datetime.now()
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        # 濡傛灉宸茬粡杩囦簡浠婂ぉ鐨勭洰鏍囨椂闂达紝绛夊埌鏄庡ぉ
        if now >= target:
            target += timedelta(days=1)
        
        wait_seconds = (target - now).total_seconds()
        logger.info(
            f"Waiting until {target.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"remaining {wait_seconds/60:.1f} minutes"
        )
        
        # 姣忓垎閽熸鏌ヤ竴娆?
        if wait_seconds > 60:
            time.sleep(60)
        else:
            time.sleep(wait_seconds)
            return


def run_scheduled(target_hour: int = 14, target_minute: int = 50, multi: bool = False, **kwargs):
    """
    瀹氭椂杩愯妯″紡
    
    姣忓ぉ 14:50 鑷姩鎵ц妯℃嫙
    """
    logger = logging.getLogger(__name__)
    mode_str = "multi" if multi else "single"
    logger.info(f"Scheduled mode started ({mode_str}), daily at {target_hour}:{target_minute:02d}")
    
    while True:
        # 绛夊緟鍒扮洰鏍囨椂闂?
        wait_until(target_hour, target_minute)
        
        # 杩愯
        if multi:
            run_multi(
                config_path=kwargs.get('config_path'),
                state_backend=kwargs.get('state_backend', 'sql'),
            )
        else:
            run_once(**{k: v for k, v in kwargs.items() if k in ['top_k', 'take_profit', 'replay_strict', 'replay_source', 'state_backend']})
        
        # 绛夊緟 1 鍒嗛挓閬垮厤閲嶅瑙﹀彂
        time.sleep(60)


# ===================== 鍛戒护琛屽叆鍙?=====================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CB simulation runner")

    parser.add_argument("--multi", action="store_true", help="run multi-strategy mode")
    parser.add_argument("--schedule", action="store_true", help="run in scheduled mode")
    parser.add_argument("--strategies-config", type=str, default=None, help="path to strategy config json")

    parser.add_argument("--date", type=str, default=None, help="run date: YYYY-MM-DD")
    parser.add_argument("--start-date", type=str, default=None, help="replay start date")
    parser.add_argument("--end-date", type=str, default=None, help="replay end date")
    parser.add_argument("--top-k", type=int, default=10, help="target holding count")
    parser.add_argument(
        "--take-profit",
        type=float,
        default=RobustConfig.TAKE_PROFIT,
        help=f"take profit ratio (default {RobustConfig.TAKE_PROFIT})",
    )
    parser.add_argument(
        "--replay-strict",
        action="store_true",
        help="strict replay alignment mode (historical replay only)",
    )
    parser.add_argument(
        "--replay-source",
        type=str,
        default="sql_eod",
        choices=["sql_eod", "parquet"],
        help="strict replay source",
    )
    parser.add_argument(
        "--state-backend",
        type=str,
        default="sql",
        choices=["sql", "json"],
        help="state backend for holdings/nav/trades",
    )
    parser.add_argument("--hour", type=int, default=14, help="schedule hour")
    parser.add_argument("--minute", type=int, default=50, help="schedule minute")

    return parser.parse_args()


def main():
    """Main entry point."""
    _ensure_console_utf8()
    args = parse_args()
    logger = setup_logging()
    
    mode_str = "multi" if args.multi else "single"
    logger.info("=" * 50)
    logger.info(f"CB simulation started ({mode_str} mode)")
    if not args.multi:
        logger.info(
            f"Params: top_k={args.top_k}, take_profit={args.take_profit}, "
            f"replay_strict={args.replay_strict}, replay_source={args.replay_source}, "
            f"state_backend={args.state_backend}"
        )
    logger.info("=" * 50)
    
    if args.schedule:
        run_scheduled(
            target_hour=args.hour,
            target_minute=args.minute,
            multi=args.multi,
            config_path=args.strategies_config,
            top_k=args.top_k,
            take_profit=args.take_profit,
            replay_strict=args.replay_strict,
            replay_source=args.replay_source,
            state_backend=args.state_backend,
        )
    elif args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise ValueError("Range replay requires both --start-date and --end-date")
        run_range(
            start_date=args.start_date,
            end_date=args.end_date,
            multi=args.multi,
            config_path=args.strategies_config,
            top_k=args.top_k,
            take_profit=args.take_profit,
            replay_strict=args.replay_strict,
            replay_source=args.replay_source,
            state_backend=args.state_backend,
        )
    elif args.multi:
        run_multi(
            date=args.date,
            config_path=args.strategies_config,
            state_backend=args.state_backend,
        )
    else:
        run_once(
            date=args.date,
            top_k=args.top_k,
            take_profit=args.take_profit,
            replay_strict=args.replay_strict,
            replay_source=args.replay_source,
            state_backend=args.state_backend,
        )


if __name__ == "__main__":
    main()



