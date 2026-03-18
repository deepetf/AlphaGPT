import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.backtest import CBBacktest
from model_core.config_loader import load_config
from model_core.engine import _init_worker, _worker_eval


def test_evaluate_robust_reports_full_day_metrics_and_valid_ratio():
    load_config()

    factors = torch.tensor(
        [
            [2.0, 1.0],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [1.0, 2.0],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    target_ret = torch.tensor(
        [
            [0.10, 0.00],
            [0.05, 0.00],
            [-0.02, 0.00],
            [0.03, 0.00],
            [-0.01, 0.00],
            [0.00, 0.20],
            [0.00, 0.03],
            [0.00, -0.01],
            [0.00, 0.04],
            [0.00, 0.01],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.ones_like(factors, dtype=torch.bool)

    backtest = CBBacktest(top_k=1, fee_rate=0.0, take_profit=0.0)
    backtest.min_valid_count = 1
    metrics = backtest.evaluate_robust(
        factors=factors,
        target_ret=target_ret,
        valid_mask=valid_mask,
        split_idx=5,
    )

    assert metrics["valid_signal_days"] == 2
    assert abs(metrics["valid_day_ratio"] - 0.2) < 1e-6
    assert metrics["annualized_ret"] != metrics["annualized_ret_valid_days"]
    assert metrics["sharpe_all"] != metrics["sharpe_all_valid_days"]
    assert metrics["sharpe_train"] > 0
    assert metrics["sharpe_val"] > 0


def test_worker_eval_rejects_sparse_formula_by_valid_day_ratio():
    config = load_config()
    rc = config["robust_config"]
    rc["min_valid_day_ratio"] = 0.5
    rc["min_sharpe_val"] = 0.0
    rc["min_active_ratio"] = 0.0
    rc["min_valid_days"] = 1
    rc["signal_min_valid_count"] = 1
    rc["top_k"] = 1
    rc["fee_rate"] = 0.0
    rc["take_profit"] = 0.0

    feat_tensor = torch.full((12, 1, 1), float("nan"), dtype=torch.float32)
    feat_tensor[2, 0, 0] = 1.0
    feat_tensor[5, 0, 0] = 2.0
    feat_tensor[8, 0, 0] = 3.0
    feat_tensor[11, 0, 0] = 4.0
    target_ret = torch.tensor(
        [[0.00], [0.00], [0.01], [0.02], [0.01], [0.00], [0.00], [0.00], [0.02], [0.01], [0.00], [0.03]],
        dtype=torch.float32,
    )
    valid_mask = torch.ones((12, 1), dtype=torch.bool)
    cs_mask = valid_mask.clone()
    split_idx = 6

    _init_worker(feat_tensor, target_ret, valid_mask, cs_mask, split_idx)
    reward, best_info, status, detail = _worker_eval(["CLOSE"])

    assert best_info is None
    assert status == "METRIC_VALID_RATIO"
    assert "ratio=" in detail


def test_full_day_metrics_carry_positions_when_signal_missing():
    load_config()

    factors = torch.tensor(
        [
            [2.0, 1.0],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    target_ret = torch.tensor(
        [
            [0.10, 0.00],
            [0.02, 0.00],
            [0.03, 0.00],
            [-0.01, 0.00],
            [0.04, 0.00],
            [0.01, 0.00],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.ones_like(factors, dtype=torch.bool)

    backtest = CBBacktest(top_k=1, fee_rate=0.0, take_profit=0.0)
    backtest.min_valid_count = 1
    metrics = backtest.evaluate_robust(
        factors=factors,
        target_ret=target_ret,
        valid_mask=valid_mask,
        split_idx=3,
    )

    assert metrics["valid_signal_days"] == 1
    assert abs(metrics["valid_day_ratio"] - (1.0 / 6.0)) < 1e-6
    assert metrics["sharpe_all"] > 0
    assert metrics["annualized_ret"] > 0
