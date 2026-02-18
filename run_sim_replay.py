import logging
import argparse
from datetime import datetime
from typing import List
import pandas as pd
import json
import os

from data_pipeline.realtime_provider import RealtimeDataProvider
from strategy_manager.sim_runner import SimulationRunner
from strategy_manager.strategy_config import StrategyConfig
from model_core.config import RobustConfig

from sqlalchemy import text

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sim_replay.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def get_trading_days(provider: RealtimeDataProvider, start_date: str, end_date: str) -> List[str]:
    """获取指定区间的交易日列表"""
    query = """
    SELECT DISTINCT trade_date
    FROM CB_DATA 
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date
    """
    with provider.sql_engine.connect() as conn:
        df = pd.read_sql(
            text(query),  # Wrap query with text()
            conn, 
            params={"start_date": start_date, "end_date": end_date}
        )
    return df['trade_date'].astype(str).tolist()

def run_replay(start_date: str, end_date: str, strategy_config: StrategyConfig = None):
    """
    执行历史回放
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        strategy_config: 策略配置 (可选，若为 None 则使用默认配置)
    """
    logger.info(f"开始历史回放: {start_date} -> {end_date}")
    
    # 1. 初始化数据提供者
    provider = RealtimeDataProvider()
    
    # 2. 获取交易日列表
    trading_days = get_trading_days(provider, start_date, end_date)
    if not trading_days:
        logger.error(f"指定区间 {start_date} 至 {end_date} 无交易日")
        return
    
    logger.info(f"共发现 {len(trading_days)} 个交易日")
    
    # 3. 初始化 Runner
    # 注意: 如果提供 strategy_config，Runner 会自动创建隔离的 portfolio/nav目录
    # 如果是默认模式，可能需要清空之前的记录以避免混淆
    if strategy_config:
        runner = SimulationRunner(data_provider=provider, strategy_config=strategy_config)
        logger.info(f"使用策略配置: {strategy_config.name}")
    else:
        # 默认回放策略配置
        from strategy_manager.strategy_config import StrategyParams
        params = StrategyParams()  # 使用默认参数 (100W, Top10, TP 8%, Fee 5bp)
        default_config = StrategyConfig(
            id="replay_default",
            name="Default Replay Strategy",
            params=params,
            formula_path="model_core/best_cb_formula.json"  # 默认公式
        )
        runner = SimulationRunner(data_provider=provider, strategy_config=default_config)
        logger.info("使用默认回放配置 (ID: replay_default, Top10, TP 8%)")

    # 4. 按日循环执行
    for date in trading_days:
        try:
            # 模拟盘只在 14:50 运行，这里模拟该时刻
            logger.info(f"--------------------------------------------------")
            logger.info(f"正在回放: {date}")
            
            result = runner.run_daily(date)
            
            if result['status'] == 'success':
                logger.info(f"[{date}] NAV: {result['nav']:.4f}, Ret: {result['daily_ret']:.2%}, "
                          f"Holdings: {result['holdings_count']}, "
                          f"TP: {result['tp_orders']}, Rebal: {result['rebalance_orders']}")
            else:
                logger.warning(f"[{date}] 运行失败或无数据")
                
        except Exception as e:
            logger.error(f"[{date}] 回放发生错误: {e}", exc_info=True)
            # 回放模式下遇到错误通常不应中断，继续下一天
            continue

    # 5. 回放结束
    logger.info(f"--------------------------------------------------")
    logger.info(f"历史回放完成")
    logger.info(f"最终净值: {runner.nav_tracker.get_latest_nav():.4f}")
    
    # 输出结果文件路径
    strategy_dir = os.path.join(runner.PORTFOLIO_BASE_DIR, runner.strategy_id)
    logger.info(f"结果保存在: {strategy_dir}")

def load_strategy_config(config_path: str) -> StrategyConfig:
    """加载 YAML 策略配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
    # 这里可以使用 strategy_manager.strategy_config 中的加载逻辑
    # 假设 load_from_yaml 是一个静态方法或辅助函数
    from strategy_manager.strategy_config import load_strategy_config as _load
    return _load(config_path)

def main():
    parser = argparse.ArgumentParser(description="AlphaGPT 历史模拟回放工具")
    parser.add_argument("--start", type=str, required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--strategy", type=str, help="策略配置文件路径 (YAML), 若不指定则使用默认配置")
    
    args = parser.parse_args()
    
    # 校验日期格式
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        logger.error("日期格式错误，请使用 YYYY-MM-DD")
        return

    # 加载策略配置 (如有)
    config = None
    if args.strategy:
        try:
            config = load_strategy_config(args.strategy)
            logger.info(f"已加载策略配置: {args.strategy}")
        except Exception as e:
            logger.error(f"加载策略配置失败: {e}")
            return
        
    run_replay(args.start, args.end, config)

if __name__ == "__main__":
    main()
