import torch

from model_core.config import ModelConfig
from model_core.vm import StackVM


def _build_feat_tensor(feature_name: str, rows):
    t = len(rows)
    a = len(rows[0])
    f = len(ModelConfig.INPUT_FEATURES)
    x = torch.zeros((t, a, f), dtype=torch.float32)
    feat_idx = ModelConfig.INPUT_FEATURES.index(feature_name)
    x[:, :, feat_idx] = torch.tensor(rows, dtype=torch.float32)
    return x


def test_ts_ret_then_cs_rank_excludes_invalid_operand():
    vm = StackVM()
    feat = _build_feat_tensor(
        "CLOSE",
        [
            [1.0, 2.0, 4.0],
            [2.0, 6.0, 8.0],
            [float("nan"), 12.0, 24.0],
        ],
    )
    mask = torch.ones((3, 3), dtype=torch.bool)

    out = vm.execute(["CLOSE", "TS_RET", "CS_RANK"], feat, cs_mask=mask)
    assert out is not None

    expected_day2 = torch.tensor([float("nan"), 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(out[2], expected_day2, atol=1e-6, equal_nan=True)


def test_ts_std20_then_cs_demean_excludes_invalid_operand():
    vm = StackVM()
    asset_a = [float(i) for i in range(1, 22)]
    asset_b = [float(i * 2) for i in range(1, 22)]
    asset_c = [float(i * 3) for i in range(1, 22)]
    asset_a[10] = float("nan")

    feat = _build_feat_tensor(
        "PREM",
        [[asset_a[i], asset_b[i], asset_c[i]] for i in range(21)],
    )
    mask = torch.ones((21, 3), dtype=torch.bool)

    out = vm.execute(["PREM", "TS_STD20", "CS_DEMEAN"], feat, cs_mask=mask)
    assert out is not None

    assert torch.isnan(out[20, 0])
    assert torch.isfinite(out[20, 1])
    assert torch.isfinite(out[20, 2])
    assert torch.allclose(out[20, 1] + out[20, 2], torch.tensor(0.0), atol=1e-6)


def test_ts_bias5_then_cs_robust_z_is_history_invariant_after_append():
    vm = StackVM()
    base_rows = [
        [1.0, 2.0, 5.0],
        [2.0, 3.0, 6.0],
        [3.0, 5.0, 8.0],
        [4.0, 7.0, 9.0],
        [5.0, 11.0, 10.0],
        [6.0, 13.0, 12.0],
        [7.0, 17.0, 15.0],
    ]
    ext_rows = base_rows + [
        [100.0, 200.0, 300.0],
        [101.0, 201.0, 301.0],
    ]

    base_feat = _build_feat_tensor("CLOSE", base_rows)
    ext_feat = _build_feat_tensor("CLOSE", ext_rows)

    base_mask = torch.ones((len(base_rows), 3), dtype=torch.bool)
    ext_mask = torch.ones((len(ext_rows), 3), dtype=torch.bool)

    base_out = vm.execute(["CLOSE", "TS_BIAS5", "CS_ROBUST_Z"], base_feat, cs_mask=base_mask)
    ext_out = vm.execute(["CLOSE", "TS_BIAS5", "CS_ROBUST_Z"], ext_feat, cs_mask=ext_mask)

    assert base_out is not None and ext_out is not None
    assert torch.allclose(base_out, ext_out[: base_out.shape[0]], atol=1e-6, equal_nan=True)
