"""
测试多策略配置加载器和多策略运行器
"""
import pytest
import os
import json
import tempfile
import shutil

from strategy_manager.strategy_config import (
    load_strategies_config,
    StrategyConfig,
    StrategyParams,
    StrategiesConfig
)
from model_core.config import RobustConfig


class TestStrategyConfig:
    """策略配置加载测试"""
    
    def test_load_config_with_formula_path(self, tmp_path):
        """测试加载包含 formula_path 的配置"""
        # 创建临时公式文件
        formula_file = tmp_path / "test_formula.json"
        formula_file.write_text(json.dumps({
            "best": {"formula": ["CLOSE", "PREM", "/"]}
        }))
        
        # 创建临时配置文件
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "strategies": [{
                "id": "test_strategy",
                "name": "Test Strategy",
                "formula_path": str(formula_file),
                "enabled": True,
                "params": {
                    "initial_capital": 500000,
                    "top_k": 5,
                    "take_profit_ratio": 0.1,
                    "fee_rate": 0.001
                }
            }],
            "global": {
                "data_source": "sql"
            }
        }))
        
        config = load_strategies_config(str(config_file))
        
        assert len(config.strategies) == 1
        assert config.strategies[0].id == "test_strategy"
        assert config.strategies[0].params.initial_capital == 500000
        assert config.strategies[0].params.top_k == 5
        
        # 测试公式加载
        formula = config.strategies[0].get_formula()
        assert formula == ["CLOSE", "PREM", "/"]
    
    def test_load_config_with_inline_formula(self, tmp_path):
        """测试加载包含内联公式的配置"""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "strategies": [{
                "id": "inline_strategy",
                "formula": ["HIGH", "LOW", "-"],
                "enabled": True,
                "params": {}
            }],
            "global": {}
        }))
        
        config = load_strategies_config(str(config_file))
        
        assert len(config.strategies) == 1
        formula = config.strategies[0].get_formula()
        assert formula == ["HIGH", "LOW", "-"]
    
    def test_get_enabled_strategies(self, tmp_path):
        """测试筛选启用的策略"""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "strategies": [
                {"id": "enabled_1", "formula": ["A"], "enabled": True},
                {"id": "disabled_1", "formula": ["B"], "enabled": False},
                {"id": "enabled_2", "formula": ["C"], "enabled": True}
            ],
            "global": {}
        }))
        
        config = load_strategies_config(str(config_file))
        enabled = config.get_enabled_strategies()
        
        assert len(enabled) == 2
        assert enabled[0].id == "enabled_1"
        assert enabled[1].id == "enabled_2"
    
    def test_default_params(self, tmp_path):
        """测试默认参数"""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "strategies": [{
                "id": "minimal",
                "formula": ["X"]
            }],
            "global": {}
        }))
        
        config = load_strategies_config(str(config_file))
        params = config.strategies[0].params
        
        assert params.initial_capital == 1000000.0
        assert params.top_k == 10
        assert params.take_profit_ratio == RobustConfig.TAKE_PROFIT
        assert params.fee_rate == 0.0005


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
