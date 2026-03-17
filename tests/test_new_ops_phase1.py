import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.formula_validator import validate_formula
from model_core.ops_registry import OpsRegistry


def _rolling_std_manual(x: torch.Tensor, window: int) -> torch.Tensor:
    if x.dim() != 2:
        return torch.full_like(x, float("nan"))
    pad = torch.full((window - 1, x.shape[1]), float("nan"), dtype=x.dtype, device=x.device)
    padded = torch.cat([pad, x], dim=0)
    unfolded = padded.unfold(0, window, 1)
    valid = torch.isfinite(unfolded)
    valid_count = valid.sum(dim=-1)
    safe = torch.where(valid, unfolded, torch.zeros_like(unfolded))
    mean = safe.sum(dim=-1) / valid_count.clamp_min(1).to(x.dtype)
    centered = torch.where(valid, unfolded - mean.unsqueeze(-1), torch.zeros_like(unfolded))
    denom = (valid_count - 1).clamp_min(1).to(x.dtype)
    var = centered.pow(2).sum(dim=-1) / denom
    out = torch.full_like(x, float("nan"))
    enough = valid_count >= max(2, window)
    out[enough] = torch.sqrt(var.clamp_min(0.0))[enough]
    return out


def _rolling_max_manual(x: torch.Tensor, window: int) -> torch.Tensor:
    if x.dim() != 2:
        return torch.full_like(x, float("nan"))
    pad = torch.full((window - 1, x.shape[1]), float("nan"), dtype=x.dtype, device=x.device)
    padded = torch.cat([pad, x], dim=0)
    unfolded = padded.unfold(0, window, 1)
    valid = torch.isfinite(unfolded)
    valid_count = valid.sum(dim=-1)
    safe = torch.where(valid, unfolded, torch.full_like(unfolded, float("-inf")))
    out = torch.full_like(x, float("nan"))
    enough = valid_count >= window
    out[enough] = safe.max(dim=-1).values[enough]
    return out


def _rolling_min_manual(x: torch.Tensor, window: int) -> torch.Tensor:
    if x.dim() != 2:
        return torch.full_like(x, float("nan"))
    pad = torch.full((window - 1, x.shape[1]), float("nan"), dtype=x.dtype, device=x.device)
    padded = torch.cat([pad, x], dim=0)
    unfolded = padded.unfold(0, window, 1)
    valid = torch.isfinite(unfolded)
    valid_count = valid.sum(dim=-1)
    safe = torch.where(valid, unfolded, torch.full_like(unfolded, float("inf")))
    out = torch.full_like(x, float("nan"))
    enough = valid_count >= window
    out[enough] = safe.min(dim=-1).values[enough]
    return out


def test_new_ops_registered():
    expected = {
        "TS_MOM10",
        "TS_MOM20",
        "TS_STD20",
        "TS_STD60",
        "TS_MAX20",
        "TS_MIN20",
    }
    ops = set(OpsRegistry.list_ops())
    assert expected.issubset(ops)


def test_ts_mom_boundary_and_values():
    x = torch.arange(1, 26, dtype=torch.float32).unsqueeze(1)

    mom10 = OpsRegistry.get_op("TS_MOM10")["func"](x)
    assert torch.isnan(mom10[:10]).all()
    assert float(mom10[10, 0]) == 10.0
    assert float(mom10[-1, 0]) == 10.0

    mom20 = OpsRegistry.get_op("TS_MOM20")["func"](x)
    assert torch.isnan(mom20[:20]).all()
    assert float(mom20[20, 0]) == 20.0
    assert float(mom20[-1, 0]) == 20.0


def test_ts_std20_std60_match_template_ddof():
    torch.manual_seed(7)
    x = torch.randn(80, 3, dtype=torch.float32)

    std20 = OpsRegistry.get_op("TS_STD20")["func"](x)
    manual20 = _rolling_std_manual(x, 20)
    assert torch.allclose(std20, manual20, atol=1e-6, equal_nan=True)

    std60 = OpsRegistry.get_op("TS_STD60")["func"](x)
    manual60 = _rolling_std_manual(x, 60)
    assert torch.allclose(std60, manual60, atol=1e-6, equal_nan=True)

    short_x = torch.randn(30, 3, dtype=torch.float32)
    short_std60 = OpsRegistry.get_op("TS_STD60")["func"](short_x)
    assert torch.isnan(short_std60).all()


def test_ts_max20_min20_match_template_and_boundary():
    x = torch.tensor(
        [
            [1.0], [3.0], [2.0], [5.0], [4.0],
            [6.0], [8.0], [7.0], [9.0], [10.0],
            [12.0], [11.0], [13.0], [15.0], [14.0],
            [16.0], [18.0], [17.0], [19.0], [20.0],
            [5.0], [6.0], [7.0], [8.0], [9.0],
        ],
        dtype=torch.float32,
    )

    max20 = OpsRegistry.get_op("TS_MAX20")["func"](x)
    min20 = OpsRegistry.get_op("TS_MIN20")["func"](x)

    assert torch.isnan(max20[:19]).all()
    assert torch.isnan(min20[:19]).all()

    manual_max20 = _rolling_max_manual(x, 20)
    manual_min20 = _rolling_min_manual(x, 20)
    assert torch.allclose(max20, manual_max20, atol=1e-6, equal_nan=True)
    assert torch.allclose(min20, manual_min20, atol=1e-6, equal_nan=True)


def test_new_ts_ops_nan_to_num():
    x = torch.arange(1, 31, dtype=torch.float32).unsqueeze(1)
    x[10, 0] = float("nan")
    x[20, 0] = float("inf")

    for op_name in ["TS_MOM10", "TS_MOM20", "TS_STD20", "TS_MAX20", "TS_MIN20"]:
        out = OpsRegistry.get_op(op_name)["func"](x)
        assert out.dim() == 2, f"{op_name} output shape invalid"
        assert torch.isfinite(out[torch.isfinite(out)]).all(), f"{op_name} finite slice invalid"


def test_ts_std60_nan_to_num_with_long_window():
    x = torch.arange(1, 81, dtype=torch.float32).unsqueeze(1)
    x[65, 0] = float("nan")
    x[70, 0] = float("inf")

    out = OpsRegistry.get_op("TS_STD60")["func"](x)
    assert torch.isfinite(out[torch.isfinite(out)]).all()
    assert torch.isnan(out[:59]).all()


def test_validator_new_penalties():
    ok_valid, ok_penalty, _ = validate_formula(["CLOSE", "ABS"])
    assert ok_valid

    mom_div_valid, mom_div_penalty, mom_div_reason = validate_formula(
        ["CLOSE", "CLOSE", "TS_MOM10", "DIV"]
    )
    assert mom_div_valid
    assert mom_div_penalty < ok_penalty
    assert "TS_MOM10->DIV" in mom_div_reason

    std20_div_valid, std20_div_penalty, std20_div_reason = validate_formula(
        ["CLOSE", "CLOSE", "TS_STD20", "DIV"]
    )
    assert std20_div_valid
    assert std20_div_penalty < ok_penalty
    assert "TS_STD20->DIV" in std20_div_reason

    noise_valid, noise_penalty, noise_reason = validate_formula(
        ["CLOSE", "TS_STD5", "TS_MAX20"]
    )
    assert noise_valid
    assert noise_penalty < ok_penalty
    assert "TS_STD5->TS_MAX20" in noise_reason

    div_dense_valid, div_dense_penalty, div_dense_reason = validate_formula(
        ["CLOSE", "CLOSE", "DIV", "CLOSE", "DIV", "CLOSE", "DIV"]
    )
    assert div_dense_valid
    assert div_dense_penalty < ok_penalty
    assert "Total DIV=3" in div_dense_reason
