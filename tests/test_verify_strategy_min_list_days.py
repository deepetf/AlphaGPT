import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.verify_strategy as verify_module


def test_run_backtest_excludes_short_list_days_asset_from_daily_holdings(monkeypatch):
    """端到端验证：verify_strategy 的向量回测结果不会持有短上市天数标的。"""

    class FakeVM:
        def execute(self, formula, feat_tensor, cs_mask=None):
            return torch.tensor(
                [
                    [0.1, 9.9, 0.2],
                    [0.1, 9.8, 0.3],
                    [0.2, 9.7, 0.4],
                    [0.2, 9.6, 0.5],
                    [0.3, 9.5, 0.6],
                    [0.3, 9.4, 0.7],
                    [0.4, 9.3, 0.8],
                    [0.4, 9.2, 0.9],
                    [0.5, 9.1, 1.0],
                    [0.5, 9.0, 1.1],
                ],
                dtype=torch.float32,
            )

    monkeypatch.setattr(verify_module, "StackVM", FakeVM)
    monkeypatch.setattr(
        "model_core.config.RobustConfig._loader",
        lambda: {
            "robust_config": {
                "signal_min_valid_count": 1,
                "min_valid_count": 1,
                "signal_clean_enabled": True,
                "signal_winsor_q": 0.01,
                "signal_clip": 5.0,
                "signal_rank_output": True,
            }
        },
        raising=False,
    )

    verifier = verify_module.StrategyVerifier.__new__(verify_module.StrategyVerifier)
    verifier.formula = ["CLOSE"]
    verifier.top_k = 1
    verifier.fee_rate = 0.0
    verifier.take_profit_ratio = 0.0
    verifier.dates = [f"2026-02-{day:02d}" for day in range(1, 11)]
    verifier.loader = SimpleNamespace(
        feat_tensor=torch.zeros((10, 3, 1), dtype=torch.float32),
        target_ret=torch.zeros((10, 3), dtype=torch.float32),
        valid_mask=torch.tensor(
            [[True, False, True]] * 10,
            dtype=torch.bool,
        ),
        tradable_mask=torch.tensor(
            [[True, False, True]] * 10,
            dtype=torch.bool,
        ),
        cs_mask=torch.tensor(
            [[True, False, True]] * 10,
            dtype=torch.bool,
        ),
        raw_data_cache={
            "CLOSE": torch.full((10, 3), 100.0, dtype=torch.float32),
            "OPEN": torch.full((10, 3), 100.0, dtype=torch.float32),
            "HIGH": torch.full((10, 3), 100.0, dtype=torch.float32),
        },
        dates_list=[f"2026-02-{day:02d}" for day in range(1, 11)],
        assets_list=["123001.SZ", "127050.SZ", "128001.SZ"],
        names_dict={
            "123001.SZ": "转债A",
            "127050.SZ": "转债B",
            "128001.SZ": "转债C",
        },
    )

    result = verifier.run_backtest()

    assert result["daily_holdings"][0] == [2]
    assert all(1 not in day for day in result["daily_holdings"])
