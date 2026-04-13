import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_core.engine as engine_module
import model_core.vm as vm_module
from model_core.backtest import CBBacktest


def _build_single_asset_inputs():
    factors = torch.ones((10, 1), dtype=torch.float32)
    target_ret = torch.zeros((10, 1), dtype=torch.float32)
    target_ret[1, 0] = 0.50
    target_ret[2, 0] = 0.30
    valid_mask = torch.ones((10, 1), dtype=torch.bool)

    open_prices = torch.full((10, 1), 100.0, dtype=torch.float32)
    high_prices = torch.full((10, 1), 100.0, dtype=torch.float32)
    prev_close = torch.full((10, 1), 100.0, dtype=torch.float32)

    open_prices[1, 0] = 112.0
    high_prices[1, 0] = 112.0
    open_prices[2, 0] = 101.0
    high_prices[2, 0] = 110.0

    return factors, target_ret, valid_mask, open_prices, high_prices, prev_close


def test_evaluate_with_details_applies_take_profit_price_path():
    factors, target_ret, valid_mask, open_prices, high_prices, prev_close = _build_single_asset_inputs()

    bt = CBBacktest(top_k=1, fee_rate=0.0, take_profit=0.08)
    bt.min_valid_count = 1

    details = bt.evaluate_with_details(
        factors=factors,
        target_ret=target_ret,
        valid_mask=valid_mask,
        open_prices=open_prices,
        high_prices=high_prices,
        prev_close=prev_close,
    )

    daily_returns = details["daily_returns"]
    assert daily_returns[0] == 0.0
    assert abs(daily_returns[1] - 0.12) < 1e-6
    assert abs(daily_returns[2] - 0.08) < 1e-6


def test_evaluate_with_details_falls_back_when_tp_inputs_are_incomplete():
    factors, target_ret, valid_mask, open_prices, high_prices, _ = _build_single_asset_inputs()

    bt = CBBacktest(top_k=1, fee_rate=0.0, take_profit=0.08)
    bt.min_valid_count = 1

    details = bt.evaluate_with_details(
        factors=factors,
        target_ret=target_ret,
        valid_mask=valid_mask,
        open_prices=open_prices,
        high_prices=high_prices,
    )

    daily_returns = details["daily_returns"]
    assert abs(daily_returns[1] - 0.50) < 1e-6
    assert abs(daily_returns[2] - 0.30) < 1e-6


def test_save_king_trades_builds_time_aligned_tp_inputs(monkeypatch, tmp_path):
    captured = {}

    class FakeStackVM:
        def execute(self, formula, feat_tensor, cs_mask=None):
            return torch.ones((3, 1), dtype=torch.float32)

    class FakeBacktest:
        def __init__(self, top_k):
            captured["top_k"] = top_k

        def evaluate_with_details(
            self,
            factors,
            target_ret,
            valid_mask,
            open_prices=None,
            high_prices=None,
            prev_close=None,
        ):
            captured["open_prices"] = open_prices.clone() if open_prices is not None else None
            captured["high_prices"] = high_prices.clone() if high_prices is not None else None
            captured["prev_close"] = prev_close.clone() if prev_close is not None else None
            return {
                "reward": 1.0,
                "cum_ret": 0.1,
                "sharpe": 1.2,
                "daily_holdings": [[0], [0], [0]],
                "daily_returns": [0.0, 0.0, 0.0],
            }

    monkeypatch.setattr(vm_module, "StackVM", FakeStackVM)
    monkeypatch.setattr(engine_module, "CBBacktest", FakeBacktest)
    monkeypatch.setattr(
        engine_module.RobustConfig,
        "_loader",
        lambda: {"robust_config": {"take_profit": 0.08, "top_k": 1}},
        raising=False,
    )

    engine = engine_module.AlphaEngine.__new__(engine_module.AlphaEngine)
    engine.run_context = {"train_dir": str(tmp_path)}
    engine.loader = SimpleNamespace(
        feat_tensor=torch.ones((3, 1, 1), dtype=torch.float32),
        cs_mask=torch.ones((3, 1), dtype=torch.bool),
        target_ret=torch.zeros((3, 1), dtype=torch.float32),
        valid_mask=torch.ones((3, 1), dtype=torch.bool),
        raw_data_cache={
            "OPEN": torch.tensor([[100.0], [112.0], [101.0]], dtype=torch.float32),
            "HIGH": torch.tensor([[100.0], [115.0], [110.0]], dtype=torch.float32),
            "CLOSE": torch.tensor([[100.0], [100.0], [100.0]], dtype=torch.float32),
        },
        dates_list=["2026-04-01", "2026-04-02", "2026-04-03"],
        assets_list=["123001.SZ"],
        names_dict={"123001.SZ": "转债A"},
    )
    engine.decode_formula = lambda formula: " ".join(formula)
    engine._write_json = lambda path, result: captured.setdefault("writes", []).append((path, result))

    engine._save_king_trades(king_num=1, formula=["CLOSE"], score=1.0, sharpe=1.2, ret=0.1)

    assert torch.equal(
        captured["open_prices"],
        torch.tensor([[112.0], [101.0], [1e9]], dtype=torch.float32),
    )
    assert torch.equal(
        captured["high_prices"],
        torch.tensor([[115.0], [110.0], [1e9]], dtype=torch.float32),
    )
    assert torch.equal(
        captured["prev_close"],
        torch.tensor([[100.0], [100.0], [100.0]], dtype=torch.float32),
    )


def test_save_king_trades_falls_back_when_open_high_are_missing(monkeypatch, tmp_path):
    captured = {}

    class FakeStackVM:
        def execute(self, formula, feat_tensor, cs_mask=None):
            return torch.ones((2, 1), dtype=torch.float32)

    class FakeBacktest:
        def __init__(self, top_k):
            pass

        def evaluate_with_details(
            self,
            factors,
            target_ret,
            valid_mask,
            open_prices=None,
            high_prices=None,
            prev_close=None,
        ):
            captured["open_prices"] = open_prices
            captured["high_prices"] = high_prices
            captured["prev_close"] = prev_close
            return {
                "reward": 1.0,
                "cum_ret": 0.1,
                "sharpe": 1.2,
                "daily_holdings": [[0], [0]],
                "daily_returns": [0.0, 0.0],
            }

    monkeypatch.setattr(vm_module, "StackVM", FakeStackVM)
    monkeypatch.setattr(engine_module, "CBBacktest", FakeBacktest)
    monkeypatch.setattr(
        engine_module.RobustConfig,
        "_loader",
        lambda: {"robust_config": {"take_profit": 0.08, "top_k": 1}},
        raising=False,
    )

    engine = engine_module.AlphaEngine.__new__(engine_module.AlphaEngine)
    engine.run_context = {"train_dir": str(tmp_path)}
    engine.loader = SimpleNamespace(
        feat_tensor=torch.ones((2, 1, 1), dtype=torch.float32),
        cs_mask=torch.ones((2, 1), dtype=torch.bool),
        target_ret=torch.zeros((2, 1), dtype=torch.float32),
        valid_mask=torch.ones((2, 1), dtype=torch.bool),
        raw_data_cache={
            "CLOSE": torch.tensor([[100.0], [100.0]], dtype=torch.float32),
        },
        dates_list=["2026-04-01", "2026-04-02"],
        assets_list=["123001.SZ"],
        names_dict={"123001.SZ": "转债A"},
    )
    engine.decode_formula = lambda formula: " ".join(formula)
    engine._write_json = lambda path, result: None

    engine._save_king_trades(king_num=1, formula=["CLOSE"], score=1.0, sharpe=1.2, ret=0.1)

    assert captured["open_prices"] is None
    assert captured["high_prices"] is None
    assert captured["prev_close"] is None
