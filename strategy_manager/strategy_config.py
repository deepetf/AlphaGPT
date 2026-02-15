"""
策略配置加载器。

统一从 JSON 配置加载一个或多个策略，供 run_sim / MultiSimRunner 使用。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from model_core.config import RobustConfig

logger = logging.getLogger(__name__)

_PARAM_FIELDS = {
    "initial_capital",
    "top_k",
    "take_profit_ratio",
    "fee_rate",
    "replay_strict",
    "replay_source",
    "state_backend",
}


@dataclass
class StrategyParams:
    """策略参数。"""

    initial_capital: float = 1_000_000.0
    top_k: int = 10
    take_profit_ratio: float = RobustConfig.TAKE_PROFIT
    fee_rate: float = 0.0005
    replay_strict: bool = False
    replay_source: str = "sql_eod"
    state_backend: str = "sql"


@dataclass
class StrategyConfig:
    """单个策略配置。"""

    id: str
    name: str
    enabled: bool = True
    formula: Optional[List[str]] = None
    formula_path: Optional[str] = None
    params: StrategyParams = field(default_factory=StrategyParams)

    def get_formula(self, base_dir: str = "") -> List[str]:
        """返回策略公式 token 列表。"""
        if self.formula:
            return self.formula

        if not self.formula_path:
            raise ValueError(f"策略 {self.id} 未定义 formula 或 formula_path")

        full_path = os.path.join(base_dir, self.formula_path) if base_dir else self.formula_path
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"公式文件不存在: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "best" in data and isinstance(data["best"], dict) and "formula" in data["best"]:
            formula = data["best"]["formula"]
        elif "formula" in data:
            formula = data["formula"]
        else:
            raise ValueError(f"无法从公式文件解析 formula: {full_path}")

        if not isinstance(formula, list) or not formula:
            raise ValueError(f"公式为空或格式错误: {full_path}")
        return formula


@dataclass
class GlobalConfig:
    """全局配置。"""

    data_source: str = "mini_qmt"
    log_level: str = "INFO"


@dataclass
class StrategiesConfig:
    """策略配置集合。"""

    strategies: List[StrategyConfig]
    global_config: GlobalConfig

    def get_enabled_strategies(self) -> List[StrategyConfig]:
        return [s for s in self.strategies if s.enabled]


def _validate_strategy_shape(raw_strategy: Dict, idx: int):
    sid = raw_strategy.get("id", f"index_{idx}")
    duplicated_param_keys = _PARAM_FIELDS.intersection(raw_strategy.keys())
    if duplicated_param_keys:
        keys = ", ".join(sorted(duplicated_param_keys))
        raise ValueError(
            f"策略 {sid} 参数必须仅放在 params 中，禁止顶层重复定义: {keys}"
        )

    if not raw_strategy.get("formula") and not raw_strategy.get("formula_path"):
        raise ValueError(f"策略 {sid} 缺少公式定义：必须提供 formula 或 formula_path")

    if "formula" in raw_strategy and raw_strategy["formula"] is not None:
        formula = raw_strategy["formula"]
        if not isinstance(formula, list) or len(formula) == 0:
            raise ValueError(f"策略 {sid} 的 formula 必须是非空数组")


def _build_strategy_params(defaults: Dict, params_raw: Dict, strategy_id: str) -> StrategyParams:
    merged = dict(defaults)
    merged.update(params_raw or {})

    replay_source = merged.get("replay_source", "sql_eod")
    if replay_source not in ("sql_eod", "parquet"):
        logger.warning(
            f"Invalid replay_source '{replay_source}' for strategy '{strategy_id}', fallback to 'sql_eod'"
        )
        replay_source = "sql_eod"

    state_backend = merged.get("state_backend", "sql")
    if state_backend not in ("sql", "json"):
        logger.warning(
            f"Invalid state_backend '{state_backend}' for strategy '{strategy_id}', fallback to 'sql'"
        )
        state_backend = "sql"

    return StrategyParams(
        initial_capital=float(merged.get("initial_capital", 1_000_000.0)),
        top_k=int(merged.get("top_k", 10)),
        take_profit_ratio=float(merged.get("take_profit_ratio", RobustConfig.TAKE_PROFIT)),
        fee_rate=float(merged.get("fee_rate", 0.0005)),
        replay_strict=bool(merged.get("replay_strict", False)),
        replay_source=replay_source,
        state_backend=state_backend,
    )


def load_strategies_config(config_path: str) -> StrategiesConfig:
    """加载策略配置文件。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"策略配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    strategies_raw = raw.get("strategies", [])
    if not strategies_raw:
        raise ValueError("策略配置不能为空：strategies 至少包含 1 个策略")

    defaults = raw.get("defaults", {}) or {}
    strategy_ids = set()
    strategies: List[StrategyConfig] = []

    for idx, s_raw in enumerate(strategies_raw):
        _validate_strategy_shape(s_raw, idx)
        sid = str(s_raw["id"]).strip()
        if not sid:
            raise ValueError(f"策略 index={idx} 的 id 不能为空")
        if sid in strategy_ids:
            raise ValueError(f"策略 id 重复: {sid}")
        strategy_ids.add(sid)

        params = _build_strategy_params(defaults, s_raw.get("params", {}), sid)
        strategy = StrategyConfig(
            id=sid,
            name=s_raw.get("name", sid),
            enabled=bool(s_raw.get("enabled", True)),
            formula=s_raw.get("formula"),
            formula_path=s_raw.get("formula_path"),
            params=params,
        )
        strategies.append(strategy)

    global_raw = raw.get("global", {}) or {}
    global_config = GlobalConfig(
        data_source=global_raw.get("data_source", "mini_qmt"),
        log_level=global_raw.get("log_level", "INFO"),
    )

    logger.info(f"Loaded {len(strategies)} strategy configs from {config_path}")
    return StrategiesConfig(strategies=strategies, global_config=global_config)
