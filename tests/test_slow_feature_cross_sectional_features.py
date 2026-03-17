import torch

from model_core.factors import FeatureEngineer


def test_slow_cross_sectional_features_preserve_cross_sectional_outputs(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        [
            "PURE_VALUE_CS_RANK",
            "PURE_VALUE_CS_ROBUST_Z",
            "PREM_CS_RANK",
            "PREM_CS_ROBUST_Z",
            "REMAIN_SIZE_CS_RANK",
            "CAP_MV_RATE_CS_RANK",
            "DBLOW_CS_RANK",
            "DBLOW_CS_ROBUST_Z",
        ],
        raising=False,
    )

    raw_data = {
        "PURE_VALUE": torch.tensor(
            [
                [100.0, 110.0, 120.0],
                [102.0, 101.0, 103.0],
            ],
            dtype=torch.float32,
        ),
        "PREM": torch.tensor(
            [
                [20.0, 10.0, 30.0],
                [15.0, 25.0, 5.0],
            ],
            dtype=torch.float32,
        ),
        "REMAIN_SIZE": torch.tensor(
            [
                [5.0, 10.0, 15.0],
                [7.0, 6.0, 8.0],
            ],
            dtype=torch.float32,
        ),
        "CAP_MV_RATE": torch.tensor(
            [
                [0.2, 0.1, 0.3],
                [0.4, 0.6, 0.5],
            ],
            dtype=torch.float32,
        ),
        "DBLOW": torch.tensor(
            [
                [90.0, 80.0, 100.0],
                [70.0, 90.0, 80.0],
            ],
            dtype=torch.float32,
        ),
    }

    cs_mask = torch.ones((2, 3), dtype=torch.bool)
    feat_tensor = FeatureEngineer.compute_features(
        raw_data,
        warmup_rows=0,
        cross_sectional_mask=cs_mask,
    )

    expected_pure_rank = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_prem_rank = torch.tensor(
        [
            [0.5, 0.0, 1.0],
            [0.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_remain_rank = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_cap_rank = torch.tensor(
        [
            [0.5, 0.0, 1.0],
            [0.0, 1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    expected_dblow_rank = torch.tensor(
        [
            [0.5, 0.0, 1.0],
            [0.0, 1.0, 0.5],
        ],
        dtype=torch.float32,
    )

    assert feat_tensor.shape == (2, 3, 8)
    assert torch.allclose(feat_tensor[:, :, 0], expected_pure_rank)
    assert torch.allclose(feat_tensor[:, :, 2], expected_prem_rank)
    assert torch.allclose(feat_tensor[:, :, 4], expected_remain_rank)
    assert torch.allclose(feat_tensor[:, :, 5], expected_cap_rank)
    assert torch.allclose(feat_tensor[:, :, 6], expected_dblow_rank)
    assert torch.isfinite(feat_tensor[:, :, 7]).all()
    assert torch.isfinite(feat_tensor).all()
    assert torch.all((feat_tensor[:, :, 0] >= 0.0) & (feat_tensor[:, :, 0] <= 1.0))
    assert torch.all((feat_tensor[:, :, 2] >= 0.0) & (feat_tensor[:, :, 2] <= 1.0))
    assert torch.all((feat_tensor[:, :, 4] >= 0.0) & (feat_tensor[:, :, 4] <= 1.0))
    assert torch.all((feat_tensor[:, :, 5] >= 0.0) & (feat_tensor[:, :, 5] <= 1.0))
    assert torch.all((feat_tensor[:, :, 6] >= 0.0) & (feat_tensor[:, :, 6] <= 1.0))


def test_slow_cross_sectional_features_skip_time_normalization_zeroing(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["PURE_VALUE_CS_RANK", "PURE_VALUE_CS_ROBUST_Z"],
        raising=False,
    )

    raw_data = {
        "PURE_VALUE": torch.tensor(
            [
                [100.0, 110.0, 120.0],
                [98.0, 101.0, 104.0],
            ],
            dtype=torch.float32,
        ),
    }

    cs_mask = torch.ones((2, 3), dtype=torch.bool)
    feat_tensor = FeatureEngineer.compute_features(
        raw_data,
        warmup_rows=0,
        cross_sectional_mask=cs_mask,
    )

    assert not torch.allclose(feat_tensor[0], torch.zeros_like(feat_tensor[0]))
    assert torch.isfinite(feat_tensor).all()
