"""
MultiSimRunner - 多策略并行运行器

负责加载多策略配置并调度各策略的 SimulationRunner 实例。
"""
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime

from data_pipeline.realtime_provider import RealtimeDataProvider
from strategy_manager.strategy_config import load_strategies_config, StrategyConfig
from strategy_manager.sim_runner import SimulationRunner

logger = logging.getLogger(__name__)


class MultiSimRunner:
    """
    多策略并行运行器
    
    职责:
    1. 加载 strategies_config.json
    2. 为每个启用的策略创建独立的 SimulationRunner
    3. 统一调度所有策略的每日运行
    4. 汇总运行结果
    """
    
    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "strategies_config.json"
    )
    
    def __init__(
        self,
        data_provider: RealtimeDataProvider,
        config_path: Optional[str] = None
    ):
        """
        初始化多策略运行器
        
        Args:
            data_provider: 实时数据提供者 (所有策略共享)
            config_path: 策略配置文件路径
        """
        self.data_provider = data_provider
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        
        # 加载配置
        self.config = load_strategies_config(self.config_path)
        
        # 为每个启用的策略创建 SimulationRunner
        self.runners: Dict[str, SimulationRunner] = {}
        for strategy_cfg in self.config.get_enabled_strategies():
            self.runners[strategy_cfg.id] = SimulationRunner(
                data_provider=data_provider,
                strategy_config=strategy_cfg
            )
        
        logger.info(f"MultiSimRunner 初始化完成: {len(self.runners)} 个策略已加载")
    
    def run_all_strategies(self, date: str) -> Dict[str, Dict]:
        """
        运行所有启用的策略
        
        Args:
            date: 运行日期 (YYYY-MM-DD)
            
        Returns:
            策略运行结果字典 {strategy_id: result}
        """
        results = {}
        
        for strategy_id, runner in self.runners.items():
            logger.info(f"{'='*20} 策略: {strategy_id} {'='*20}")
            try:
                result = runner.run_daily(date)
                results[strategy_id] = result
                
                # 简要日志
                if result.get('status') == 'success':
                    logger.info(
                        f"[{strategy_id}] NAV={result.get('nav', 0):.2f}, "
                        f"收益率={result.get('daily_ret', 0)*100:.2f}%"
                    )
                else:
                    logger.warning(f"[{strategy_id}] 运行状态: {result.get('status')}")
                    
            except Exception as e:
                logger.error(f"[{strategy_id}] 运行失败: {e}")
                results[strategy_id] = {"status": "error", "error": str(e)}
        
        return results
    
    def get_strategy_summary(self) -> List[Dict]:
        """
        获取所有策略的摘要信息
        
        Returns:
            策略摘要列表
        """
        summaries = []
        for strategy_id, runner in self.runners.items():
            summary = {
                "id": strategy_id,
                "name": runner.strategy_name,
                "top_k": runner.top_k,
                "take_profit_ratio": runner.take_profit_ratio,
                "nav": runner.nav_tracker.get_latest_nav(),
                "cash": runner.nav_tracker.cash,
                "holdings_count": runner.portfolio.get_holdings_count()
            }
            summaries.append(summary)
        return summaries
    
    def print_summary(self):
        """打印所有策略的摘要"""
        summaries = self.get_strategy_summary()
        logger.info("=" * 60)
        logger.info("多策略摘要")
        logger.info("=" * 60)
        for s in summaries:
            logger.info(
                f"[{s['id']}] {s['name']}: "
                f"NAV={s['nav']:.2f}, 现金={s['cash']:.2f}, 持仓={s['holdings_count']}"
            )
        logger.info("=" * 60)
