#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CB 模拟盘运行入口 (run_sim.py)

使用方式:
    # 单策略模式 (传统)
    python run_sim.py
    
    # 多策略模式 (新)
    python run_sim.py --multi
    python run_sim.py --multi --strategies-config path/to/config.json
    
    # 定时运行 (14:50 自动触发)
    python run_sim.py --schedule
    python run_sim.py --schedule --multi
    
    # 指定日期回放
    python run_sim.py --date 2026-02-07

    # 区间回放（连续持仓，不每日重置）
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

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.realtime_provider import RealtimeDataProvider
from strategy_manager.cb_portfolio import CBPortfolioManager
from strategy_manager.nav_tracker import NavTracker
from strategy_manager.sim_runner import SimulationRunner
from model_core.config import RobustConfig

# ===================== 日志配置 =====================

def setup_logging(log_dir: str = None):
    """配置日志"""
    log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"sim_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 配置格式
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def _reset_runner_state(runner: SimulationRunner):
    """清空 runner 的状态文件，用于回放前重置。"""
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
            logger.warning(f"清理候选历史失败: {candidates_path}, error={e}")

    logger.info(f"[state reset] strategy={runner.strategy_id}")


# ===================== 核心运行逻辑 =====================

def run_once(
    date: str = None,
    top_k: int = 10,
    take_profit: float = RobustConfig.TAKE_PROFIT,
    replay_strict: bool = False,
    replay_source: str = "sql_eod",
):
    """
    执行单次模拟 (单策略模式)
    
    Args:
        date: 运行日期，默认为今天
        top_k: 持仓数量
        take_profit: 止盈比例
        replay_strict: 是否启用严格回放对齐模式
        replay_source: 严格回放数据源
    """
    logger = logging.getLogger(__name__)
    date = date or datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"========== 模拟盘运行: {date} ==========")
    
    # 初始化组件
    data_provider = RealtimeDataProvider()
    portfolio = CBPortfolioManager()
    nav_tracker = NavTracker(initial_capital=1_000_000.0)
    
    # 初始化运行器
    runner = SimulationRunner(
        data_provider=data_provider,
        portfolio=portfolio,
        nav_tracker=nav_tracker,
        top_k=top_k,
        take_profit_ratio=take_profit,
        replay_strict=replay_strict,
        replay_source=replay_source,
    )

    # 历史回放默认清空状态，避免写入重复成交记录
    if date != datetime.now().strftime('%Y-%m-%d'):
        _reset_runner_state(runner)
    
    # 执行
    try:
        result = runner.run_daily(date=date)
        logger.info(f"运行结果: {result}")
        
        # 打印摘要
        logger.info(portfolio.summary())
        logger.info(nav_tracker.summary())
        
        return result
    except Exception as e:
        logger.exception(f"运行失败: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        data_provider.close()


def run_multi(date: str = None, config_path: str = None):
    """
    执行多策略模拟
    
    Args:
        date: 运行日期，默认为今天
        config_path: 策略配置文件路径
    """
    logger = logging.getLogger(__name__)
    date = date or datetime.now().strftime('%Y-%m-%d')
    
    # 延迟导入避免循环引用
    from strategy_manager.multi_sim_runner import MultiSimRunner
    
    logger.info(f"========== 多策略模拟运行: {date} ==========")
    
    # 初始化数据提供者 (所有策略共享)
    data_provider = RealtimeDataProvider()
    
    try:
        # 初始化多策略运行器
        multi_runner = MultiSimRunner(
            data_provider=data_provider,
            config_path=config_path
        )

        # 历史回放默认清空状态，避免在旧回放记录上追加
        if date != datetime.now().strftime('%Y-%m-%d'):
            for runner in multi_runner.runners.values():
                _reset_runner_state(runner)
        
        # 运行所有策略
        results = multi_runner.run_all_strategies(date)
        
        # 打印汇总
        multi_runner.print_summary()
        
        return results
    except Exception as e:
        logger.exception(f"多策略运行失败: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        data_provider.close()


def _get_trading_dates(data_provider: RealtimeDataProvider, start_date: str, end_date: str) -> List[str]:
    """从 SQL 获取区间交易日列表（升序）"""
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
):
    """
    区间回放（保持连续持仓状态）

    Args:
        start_date: 起始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        multi: 是否多策略模式
        config_path: 多策略配置路径
        top_k: 单策略持仓数量
        take_profit: 单策略止盈比例
        replay_strict: 单策略严格回放
        replay_source: 单策略严格回放数据源
    """
    logger = logging.getLogger(__name__)
    mode_str = "多策略" if multi else "单策略"
    logger.info(f"========== 区间回放 ({mode_str}) {start_date} ~ {end_date} ==========")

    if start_date > end_date:
        raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")

    data_provider = RealtimeDataProvider()
    try:
        trade_dates = _get_trading_dates(data_provider, start_date, end_date)
        if not trade_dates:
            logger.warning(f"区间内无交易日: {start_date} ~ {end_date}")
            return {"status": "no_data", "start_date": start_date, "end_date": end_date}

        logger.info(f"共 {len(trade_dates)} 个交易日，将顺序回放")

        if multi:
            # 延迟导入避免循环引用
            from strategy_manager.multi_sim_runner import MultiSimRunner

            multi_runner = MultiSimRunner(
                data_provider=data_provider,
                config_path=config_path
            )

            # 区间回放默认清空状态，防止累计写入
            for runner in multi_runner.runners.values():
                _reset_runner_state(runner)

            all_results = {}
            for i, d in enumerate(trade_dates, 1):
                logger.info(f"[{i}/{len(trade_dates)}] 回放 {d}")
                all_results[d] = multi_runner.run_all_strategies(d)

            multi_runner.print_summary()
            return {
                "status": "success",
                "start_date": start_date,
                "end_date": end_date,
                "days": len(trade_dates),
                "results": all_results,
            }

        # 单策略：一个 runner 连续跑，保持组合状态
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
        )

        # 区间回放默认清空状态，防止累计写入
        _reset_runner_state(runner)

        all_results = {}
        for i, d in enumerate(trade_dates, 1):
            logger.info(f"[{i}/{len(trade_dates)}] 回放 {d}")
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
        logger.exception(f"区间回放失败: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        data_provider.close()


def wait_until(target_hour: int, target_minute: int):
    """
    等待到指定时间
    
    Args:
        target_hour: 目标小时
        target_minute: 目标分钟
    """
    logger = logging.getLogger(__name__)
    
    while True:
        now = datetime.now()
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        # 如果已经过了今天的目标时间，等到明天
        if now >= target:
            target += timedelta(days=1)
        
        wait_seconds = (target - now).total_seconds()
        logger.info(f"等待至 {target.strftime('%Y-%m-%d %H:%M:%S')}，还有 {wait_seconds/60:.1f} 分钟")
        
        # 每分钟检查一次
        if wait_seconds > 60:
            time.sleep(60)
        else:
            time.sleep(wait_seconds)
            return


def run_scheduled(target_hour: int = 14, target_minute: int = 50, multi: bool = False, **kwargs):
    """
    定时运行模式
    
    每天 14:50 自动执行模拟
    """
    logger = logging.getLogger(__name__)
    mode_str = "多策略" if multi else "单策略"
    logger.info(f"启动定时模式 ({mode_str})，每日 {target_hour}:{target_minute:02d} 运行")
    
    while True:
        # 等待到目标时间
        wait_until(target_hour, target_minute)
        
        # 运行
        if multi:
            run_multi(config_path=kwargs.get('config_path'))
        else:
            run_once(**{k: v for k, v in kwargs.items() if k in ['top_k', 'take_profit', 'replay_strict', 'replay_source']})
        
        # 等待 1 分钟避免重复触发
        time.sleep(60)


# ===================== 命令行入口 =====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CB 模拟盘运行器')
    
    # 运行模式
    parser.add_argument(
        '--multi', 
        action='store_true',
        help='多策略模式 (从 strategies_config.json 加载)'
    )
    parser.add_argument(
        '--schedule', 
        action='store_true',
        help='定时运行模式 (每日 14:50)'
    )
    
    # 多策略配置
    parser.add_argument(
        '--strategies-config',
        type=str,
        default=None,
        help='多策略配置文件路径 (默认: strategy_manager/strategies_config.json)'
    )
    
    # 单策略参数 (传统模式)
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='指定运行日期 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='区间回放开始日期 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='区间回放结束日期 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='持仓数量 (默认 10, 仅单策略模式)'
    )
    parser.add_argument(
        '--take-profit',
        type=float,
        default=RobustConfig.TAKE_PROFIT,
        help=f'止盈比例 (默认 {RobustConfig.TAKE_PROFIT}, 仅单策略模式)'
    )
    parser.add_argument(
        '--replay-strict',
        action='store_true',
        help='启用严格回放对齐模式（仅历史回放生效）'
    )
    parser.add_argument(
        '--replay-source',
        type=str,
        default='sql_eod',
        choices=['sql_eod', 'parquet'],
        help='严格回放数据源 (默认 sql_eod)'
    )
    
    # 定时参数
    parser.add_argument(
        '--hour',
        type=int,
        default=14,
        help='定时运行小时 (默认 14)'
    )
    parser.add_argument(
        '--minute',
        type=int,
        default=50,
        help='定时运行分钟 (默认 50)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging()
    
    mode_str = "多策略" if args.multi else "单策略"
    logger.info("=" * 50)
    logger.info(f"CB 模拟盘启动 ({mode_str}模式)")
    if not args.multi:
        logger.info(
            f"参数: top_k={args.top_k}, take_profit={args.take_profit}, "
            f"replay_strict={args.replay_strict}, replay_source={args.replay_source}"
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
        )
    elif args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise ValueError("区间回放需要同时提供 --start-date 和 --end-date")
        run_range(
            start_date=args.start_date,
            end_date=args.end_date,
            multi=args.multi,
            config_path=args.strategies_config,
            top_k=args.top_k,
            take_profit=args.take_profit,
            replay_strict=args.replay_strict,
            replay_source=args.replay_source,
        )
    elif args.multi:
        run_multi(
            date=args.date,
            config_path=args.strategies_config,
        )
    else:
        run_once(
            date=args.date,
            top_k=args.top_k,
            take_profit=args.take_profit,
            replay_strict=args.replay_strict,
            replay_source=args.replay_source,
        )


if __name__ == "__main__":
    main()
