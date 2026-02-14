"""
策略配置加载器

负责解析 strategies_config.json，为多策略并行运行提供配置支持。
"""
import json
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from model_core.config import RobustConfig

logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    """策略参数"""
    initial_capital: float = 1000000.0
    top_k: int = 10
    take_profit_ratio: float = RobustConfig.TAKE_PROFIT
    fee_rate: float = 0.0005
    replay_strict: bool = False
    replay_source: str = "sql_eod"
    state_backend: str = "sql"


@dataclass
class StrategyConfig:
    """单个策略配置"""
    id: str
    name: str
    enabled: bool = True
    formula: Optional[List[str]] = None       # 内联公式
    formula_path: Optional[str] = None        # 公式文件路径
    params: StrategyParams = field(default_factory=StrategyParams)
    
    def get_formula(self, base_dir: str = "") -> List[str]:
        """
        获取公式列表
        
        Args:
            base_dir: 基础目录，用于解析相对路径
            
        Returns:
            公式 token 列表
        """
        # 优先使用内联公式
        if self.formula:
            return self.formula
        
        # 否则从文件加载
        if self.formula_path:
            full_path = os.path.join(base_dir, self.formula_path) if base_dir else self.formula_path
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"公式文件不存在: {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 兼容 best_cb_formula.json 格式
            if 'best' in data and 'formula' in data['best']:
                return data['best']['formula']
            elif 'formula' in data:
                return data['formula']
            else:
                raise ValueError(f"无法从 {full_path} 中解析公式")
        
        raise ValueError(f"策略 {self.id} 未定义公式 (formula 或 formula_path)")


@dataclass
class GlobalConfig:
    """全局配置"""
    data_source: str = "mini_qmt"
    log_level: str = "INFO"


@dataclass
class StrategiesConfig:
    """多策略总配置"""
    strategies: List[StrategyConfig]
    global_config: GlobalConfig
    
    def get_enabled_strategies(self) -> List[StrategyConfig]:
        """获取所有启用的策略"""
        return [s for s in self.strategies if s.enabled]


def load_strategies_config(config_path: str) -> StrategiesConfig:
    """
    加载策略配置文件
    
    Args:
        config_path: JSON 配置文件路径
        
    Returns:
        StrategiesConfig 实例
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"策略配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    strategies = []
    for s_raw in raw.get('strategies', []):
        # 解析参数
        params_raw = s_raw.get('params', {})
        replay_source = params_raw.get('replay_source', 'sql_eod')
        if replay_source not in ('sql_eod', 'parquet'):
            logger.warning(
                f"Invalid replay_source '{replay_source}' for strategy '{s_raw.get('id', 'unknown')}', "
                f"fallback to 'sql_eod'"
            )
            replay_source = 'sql_eod'
        state_backend = params_raw.get('state_backend', 'sql')
        if state_backend not in ('sql', 'json'):
            logger.warning(
                f"Invalid state_backend '{state_backend}' for strategy '{s_raw.get('id', 'unknown')}', "
                f"fallback to 'sql'"
            )
            state_backend = 'sql'

        params = StrategyParams(
            initial_capital=params_raw.get('initial_capital', 1000000.0),
            top_k=params_raw.get('top_k', 10),
            take_profit_ratio=params_raw.get('take_profit_ratio', RobustConfig.TAKE_PROFIT),
            fee_rate=params_raw.get('fee_rate', 0.0005),
            replay_strict=params_raw.get('replay_strict', False),
            replay_source=replay_source,
            state_backend=state_backend,
        )
        
        strategy = StrategyConfig(
            id=s_raw['id'],
            name=s_raw.get('name', s_raw['id']),
            enabled=s_raw.get('enabled', True),
            formula=s_raw.get('formula'),
            formula_path=s_raw.get('formula_path'),
            params=params
        )
        strategies.append(strategy)
    
    # 解析全局配置
    global_raw = raw.get('global', {})
    global_config = GlobalConfig(
        data_source=global_raw.get('data_source', 'mini_qmt'),
        log_level=global_raw.get('log_level', 'INFO')
    )
    
    logger.info(f"加载 {len(strategies)} 个策略配置")
    return StrategiesConfig(strategies=strategies, global_config=global_config)
