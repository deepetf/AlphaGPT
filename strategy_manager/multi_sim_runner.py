"""
MultiSimRunner - 多策略运行器
"""

import copy
import logging
import os
from typing import Dict, List, Optional

from data_pipeline.realtime_provider import RealtimeDataProvider
from strategy_manager.sim_runner import SimulationRunner
from strategy_manager.strategy_config import load_strategies_config

logger = logging.getLogger(__name__)


class MultiSimRunner:
    """统一管理多个策略运行。"""

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "strategies_config.json",
    )

    def __init__(
        self,
        data_provider: RealtimeDataProvider,
        config_path: Optional[str] = None,
        strategy_ids: Optional[List[str]] = None,
        state_backend: Optional[str] = None,
        dataset: str = "replay",
        live_quote_source: str = "dummy",
        replay_source_override: Optional[str] = None,
        strict_start_date: Optional[str] = None,
        strict_end_date: Optional[str] = None,
    ):
        self.data_provider = data_provider
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.dataset = dataset
        self.live_quote_source = live_quote_source
        self.replay_source_override = replay_source_override
        self.strict_start_date = strict_start_date
        self.strict_end_date = strict_end_date
        self.config = load_strategies_config(self.config_path)

        enabled = self.config.get_enabled_strategies()
        if strategy_ids:
            expected = {sid.strip() for sid in strategy_ids if sid and sid.strip()}
            enabled = [s for s in enabled if s.id in expected]
            missing = expected - {s.id for s in enabled}
            if missing:
                missing_txt = ", ".join(sorted(missing))
                raise ValueError(f"配置中找不到可用策略: {missing_txt}")

        if not enabled:
            raise ValueError("没有可运行策略（enabled=true 且通过 strategy_ids 过滤）")

        self.runners: Dict[str, SimulationRunner] = {}
        for strategy_cfg in enabled:
            cfg = copy.deepcopy(strategy_cfg)
            if state_backend is not None:
                cfg.params.state_backend = state_backend
            if replay_source_override is not None:
                cfg.params.replay_source = replay_source_override

            self.runners[cfg.id] = SimulationRunner(
                data_provider=self.data_provider,
                strategy_config=cfg,
                dataset=self.dataset,
                live_quote_source=self.live_quote_source,
                strict_start_date=self.strict_start_date,
                strict_end_date=self.strict_end_date,
            )

        logger.info(
            f"MultiSimRunner ready: runners={len(self.runners)}, "
            f"dataset={self.dataset}, live_quote_source={self.live_quote_source}, "
            f"replay_source_override={self.replay_source_override or 'None'}, "
            f"strict_range=[{self.strict_start_date or 'default'}, {self.strict_end_date or 'latest'}]"
        )

    def run_all_strategies(self, date: str, mode: str = "auto") -> Dict[str, Dict]:
        """按顺序运行全部策略。"""
        results: Dict[str, Dict] = {}
        for strategy_id, runner in self.runners.items():
            logger.info(f"{'=' * 20} Strategy: {strategy_id} {'=' * 20}")
            try:
                results[strategy_id] = runner.run_daily(date, mode=mode)
            except Exception as e:
                logger.exception(f"[{strategy_id}] run failed: {e}")
                results[strategy_id] = {"status": "error", "error": str(e)}
        return results

    def get_strategy_summary(self) -> List[Dict]:
        """返回策略摘要。"""
        summaries = []
        for strategy_id, runner in self.runners.items():
            nav = runner.nav_tracker.get_latest_nav()
            summaries.append(
                {
                    "id": strategy_id,
                    "name": runner.strategy_name,
                    "top_k": runner.top_k,
                    "take_profit_ratio": runner.take_profit_ratio,
                    "nav": float(nav) if nav is not None else 0.0,
                    "cash": float(runner.nav_tracker.cash),
                    "holdings_count": int(runner.portfolio.get_holdings_count()),
                }
            )
        return summaries

    def print_summary(self):
        """打印策略摘要。"""
        logger.info("=" * 60)
        logger.info("Strategy Summary")
        logger.info("=" * 60)
        for s in self.get_strategy_summary():
            logger.info(
                f"[{s['id']}] {s['name']}: NAV={s['nav']:.2f}, "
                f"cash={s['cash']:.2f}, holdings={s['holdings_count']}"
            )
        logger.info("=" * 60)
