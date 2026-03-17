import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.ops_registry import OpsRegistry


def test_ts_delay_requires_valid_history():
    x = torch.tensor(
        [
            [1.0],
            [2.0],
            [float("nan")],
            [4.0],
        ],
        dtype=torch.float32,
    )

    out = OpsRegistry.get_op("TS_DELAY")["func"](x)
    assert torch.isnan(out[0, 0])
    assert torch.allclose(out[1, 0], torch.tensor(1.0))
    assert torch.allclose(out[2, 0], torch.tensor(2.0))
    assert torch.isnan(out[3, 0])


def test_ts_mean5_requires_full_valid_window():
    x = torch.tensor(
        [
            [1.0],
            [2.0],
            [float("nan")],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
        ],
        dtype=torch.float32,
    )

    out = OpsRegistry.get_op("TS_MEAN5")["func"](x)
    assert torch.isnan(out[:7]).all()


def test_ts_bias5_is_causal_when_appending_future_dates():
    base = torch.tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
        ],
        dtype=torch.float32,
    )
    extended = torch.tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [100.0],
            [200.0],
        ],
        dtype=torch.float32,
    )

    base_out = OpsRegistry.get_op("TS_BIAS5")["func"](base)
    ext_out = OpsRegistry.get_op("TS_BIAS5")["func"](extended)

    assert torch.allclose(base_out, ext_out[: base_out.shape[0]], atol=1e-6, equal_nan=True)


def test_ts_ret_propagates_invalid_current_or_history():
    x = torch.tensor(
        [
            [1.0],
            [2.0],
            [float("nan")],
            [4.0],
            [8.0],
        ],
        dtype=torch.float32,
    )

    out = OpsRegistry.get_op("TS_RET")["func"](x)
    assert torch.isnan(out[0, 0])
    assert torch.allclose(out[1, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.isnan(out[2, 0])
    assert torch.isnan(out[3, 0])
    assert torch.allclose(out[4, 0], torch.tensor(1.0), atol=1e-6)
