"""
SimulationRunner 单元测试（V5 配置驱动接口）。
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.realtime_provider import RealtimeDataProvider
from strategy_manager.sim_runner import SimulationRunner
from strategy_manager.strategy_config import StrategyConfig, StrategyParams


class TestSimulationRunner:
    """测试 SimulationRunner 核心逻辑。"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录。"""
        import shutil

        directory = tempfile.mkdtemp()
        yield directory
        shutil.rmtree(directory)

    @pytest.fixture
    def mock_data_provider(self):
        """创建 Mock 数据提供者。"""
        provider = MagicMock(spec=RealtimeDataProvider)
        provider.sql_engine = MagicMock()

        provider.get_cb_features.return_value = pd.DataFrame(
            {
                "code": ["123001.SZ", "127050.SZ", "128001.SZ"],
                "name": ["转债A", "转债B", "转债C"],
                "close": [100.0, 110.0, 120.0],
                "open": [99.0, 109.0, 119.0],
                "high": [102.0, 115.0, 125.0],
                "vol": [1000.0, 2000.0, 3000.0],
                "conv_prem": [0.1, 0.15, 0.2],
            }
        )

        provider.get_realtime_quotes.return_value = pd.DataFrame(
            {
                "code": ["123001.SZ", "127050.SZ", "128001.SZ"],
                "close": [101.0, 111.0, 121.0],
            }
        )

        provider.get_realtime_quotes_dummy.return_value = pd.DataFrame()
        provider.get_prev_close.return_value = {}
        provider.build_feat_tensor.return_value = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        provider.get_asset_list.return_value = ["123001.SZ", "127050.SZ", "128001.SZ"]
        provider.get_names_dict.return_value = {
            "123001.SZ": "转债A",
            "127050.SZ": "转债B",
            "128001.SZ": "转债C",
        }
        return provider

    @pytest.fixture
    def runner_factory(self, temp_dir, mock_data_provider, monkeypatch):
        """创建符合 V5 接口的 SimulationRunner。"""
        monkeypatch.setattr(SimulationRunner, "PORTFOLIO_BASE_DIR", temp_dir)

        def _create_runner(
            strategy_id: str,
            top_k: int = 10,
            take_profit_ratio: float = 0.0,
        ) -> SimulationRunner:
            cfg = StrategyConfig(
                id=strategy_id,
                name=f"test_{strategy_id}",
                formula=["CLOSE"],
                params=StrategyParams(
                    initial_capital=1_000_000.0,
                    top_k=top_k,
                    take_profit_ratio=take_profit_ratio,
                    fee_rate=0.0005,
                    replay_strict=False,
                    replay_source="sql_eod",
                    state_backend="json",
                ),
            )
            return SimulationRunner(
                data_provider=mock_data_provider,
                strategy_config=cfg,
            )

        return _create_runner

    def test_check_take_profit_gap_up(self, mock_data_provider, runner_factory):
        """测试跳空止盈。"""
        runner = runner_factory("tp_gap", top_k=10, take_profit_ratio=0.08)
        runner.portfolio.add_position("127050.SZ", "转债B", 100, 100.0, "2026-02-07")

        mock_data_provider.get_cb_features.return_value = pd.DataFrame(
            {
                "code": ["127050.SZ"],
                "name": ["转债B"],
                "close": [115.0],
                "open": [112.0],
                "high": [118.0],
                "vol": [2000.0],
            }
        )
        mock_data_provider.get_prev_close.return_value = {"127050.SZ": 100.0}

        cb_features = mock_data_provider.get_cb_features()
        realtime_quotes = pd.DataFrame(columns=["code", "open", "high", "close"])
        prices = {"127050.SZ": 115.0}

        orders = runner._check_take_profit(cb_features, realtime_quotes, prices, "2026-02-08")

        assert len(orders) == 1
        assert orders[0].code == "127050.SZ"
        assert orders[0].target_price == 112.0

    def test_check_take_profit_intraday(self, mock_data_provider, runner_factory):
        """测试盘中止盈。"""
        runner = runner_factory("tp_intraday", top_k=10, take_profit_ratio=0.08)
        runner.portfolio.add_position("127050.SZ", "转债B", 100, 100.0, "2026-02-07")

        mock_data_provider.get_cb_features.return_value = pd.DataFrame(
            {
                "code": ["127050.SZ"],
                "name": ["转债B"],
                "close": [107.0],
                "open": [101.0],
                "high": [110.0],
                "vol": [2000.0],
            }
        )
        mock_data_provider.get_prev_close.return_value = {"127050.SZ": 100.0}

        cb_features = mock_data_provider.get_cb_features()
        realtime_quotes = pd.DataFrame(columns=["code", "open", "high", "close"])
        prices = {"127050.SZ": 107.0}

        orders = runner._check_take_profit(cb_features, realtime_quotes, prices, "2026-02-08")

        assert len(orders) == 1
        assert orders[0].target_price == 108.0

    def test_select_top_k(self, runner_factory):
        """测试 Top-K 选股。"""
        runner = runner_factory("topk", top_k=2, take_profit_ratio=0.0)

        factor_values = torch.tensor([0.5, 0.9, 0.3])
        asset_list = ["123001.SZ", "127050.SZ", "128001.SZ"]
        prices = {"123001.SZ": 100.0, "127050.SZ": 110.0, "128001.SZ": 120.0}

        selected = runner._select_top_k(factor_values, asset_list, prices)

        assert len(selected) == 2
        assert "127050.SZ" in selected
        assert "123001.SZ" in selected

    def test_generate_rebalance_orders(self, runner_factory):
        """测试调仓订单生成。"""
        runner = runner_factory("rebalance", top_k=2, take_profit_ratio=0.0)
        runner.portfolio.add_position("123001.SZ", "转债A", 100, 100.0, "2026-02-07")
        runner.nav_tracker.cash = 500_000.0

        target_codes = ["127050.SZ", "128001.SZ"]
        prices = {"123001.SZ": 100.0, "127050.SZ": 110.0, "128001.SZ": 120.0}
        names_dict = {"127050.SZ": "转债B", "128001.SZ": "转债C"}

        orders = runner._generate_rebalance_orders(target_codes, prices, names_dict, "2026-02-08")

        sell_orders = [order for order in orders if order.side.value == "SELL"]
        buy_orders = [order for order in orders if order.side.value == "BUY"]

        assert len(sell_orders) == 1
        assert sell_orders[0].code == "123001.SZ"
        assert len(buy_orders) == 2

    def test_run_daily_live_passes_strict_valid_mask(self, mock_data_provider, runner_factory, monkeypatch):
        """测试 live 选股使用 strict 对齐的 valid_mask。"""
        runner = runner_factory("live_mask", top_k=2, take_profit_ratio=0.0)
        trade_date = "2026-02-08"

        mock_data_provider.get_cb_features.return_value = pd.DataFrame(
            {
                "code": ["123001.SZ", "127050.SZ", "128001.SZ"],
                "name": ["转债A", "转债B", "转债C"],
                "trade_date": [trade_date, trade_date, trade_date],
                "close": [100.0, 110.0, 120.0],
                "open": [99.0, 109.0, 119.0],
                "high": [102.0, 112.0, 122.0],
                "vol": [1000.0, 2000.0, 3000.0],
                "left_years": [3.0, 3.0, 3.0],
            }
        )
        mock_data_provider.get_realtime_quotes_dummy.return_value = pd.DataFrame(columns=["code", "close"])
        mock_data_provider.get_trading_days_before.return_value = ["2026-02-07", trade_date]

        fake_valid_mask = torch.tensor([True, False, True], dtype=torch.bool)

        class FakeLoader:
            def __init__(self, *args, **kwargs):
                self.dates_list = [trade_date]
                self.assets_list = ["123001.SZ", "127050.SZ", "128001.SZ"]
                self.names_dict = {
                    "123001.SZ": "转债A",
                    "127050.SZ": "转债B",
                    "128001.SZ": "转债C",
                }
                self.feat_tensor = torch.zeros((1, 3, 1), dtype=torch.float32)
                self.valid_mask = torch.tensor([[True, False, True]], dtype=torch.bool)

            def load_data(self):
                return None

        import data_pipeline.sql_strict_loader as strict_loader_module

        monkeypatch.setattr(strict_loader_module, "SQLStrictLoader", FakeLoader)
        runner.vm.execute = MagicMock(return_value=torch.tensor([[0.2, 9.9, 0.1]], dtype=torch.float32))
        runner._select_top_k = MagicMock(return_value=["123001.SZ", "128001.SZ"])
        runner._generate_rebalance_orders = MagicMock(return_value=[])

        result = runner.run_daily(trade_date, mode="live")

        assert result["status"] == "success"
        kwargs = runner._select_top_k.call_args.kwargs
        assert "valid_mask" in kwargs
        assert torch.equal(kwargs["valid_mask"], fake_valid_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
