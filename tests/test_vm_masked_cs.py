import torch

from model_core.config import ModelConfig
from model_core.vm import StackVM


def _build_feat_tensor(close_rows):
    """
    close_rows: List[List[float]] -> [T, A]
    """
    t = len(close_rows)
    a = len(close_rows[0])
    f = len(ModelConfig.INPUT_FEATURES)
    x = torch.zeros((t, a, f), dtype=torch.float32)
    close_idx = ModelConfig.INPUT_FEATURES.index("CLOSE")
    x[:, :, close_idx] = torch.tensor(close_rows, dtype=torch.float32)
    return x


def test_masked_cs_rank_excludes_outside_universe():
    vm = StackVM()
    feat = _build_feat_tensor([[1.0, 2.0, 3.0, 100.0]])
    mask = torch.tensor([[True, True, True, False]])

    out = vm.execute(["CLOSE", "CS_RANK"], feat, cs_mask=mask)
    assert out is not None

    # 仅在 mask 内部排名: [1,2,3] -> [0,0.5,1.0]
    expected = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    assert torch.allclose(out[:, :3], expected, atol=1e-6)
    assert torch.isnan(out[:, 3]).all()


def test_masked_cs_accepts_1d_mask_and_applies_per_day():
    vm = StackVM()
    feat = _build_feat_tensor(
        [
            [1.0, 2.0, 3.0, 100.0],
            [3.0, 2.0, 1.0, 100.0],
        ]
    )
    # 1D mask 会自动扩展到每个时点
    mask = torch.tensor([True, True, True, False])

    out = vm.execute(["CLOSE", "CS_DEMEAN"], feat, cs_mask=mask)
    assert out is not None

    # day0: [1,2,3] demean -> [-1,0,1], day1: [3,2,1] demean -> [1,0,-1]
    expected = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(out[:, :3], expected, atol=1e-6)
    assert torch.isnan(out[:, 3]).all()


def test_cs_op_without_mask_raises():
    vm = StackVM()
    feat = _build_feat_tensor([[1.0, 2.0, 3.0]])

    try:
        vm.execute(["CLOSE", "CS_RANK"], feat)
    except ValueError as exc:
        assert "cs_mask" in str(exc)
    else:
        raise AssertionError("cross-sectional op should require cs_mask")
