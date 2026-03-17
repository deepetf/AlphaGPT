import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.factors import FeatureEngineer


def test_robust_normalize_marks_insufficient_window_invalid():
    x = torch.tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=torch.float32,
    )

    norm, valid = FeatureEngineer._robust_normalize(
        x,
        window=3,
        min_valid_obs=3,
        return_validity=True,
    )

    assert torch.equal(valid.squeeze(1), torch.tensor([False, False, True, True]))
    assert torch.isnan(norm[0, 0])
    assert torch.isnan(norm[1, 0])
    assert torch.isfinite(norm[2:, 0]).all()


def test_robust_normalize_uses_only_valid_history():
    x = torch.tensor(
        [
            [float("nan")],
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=torch.float32,
    )

    norm, valid = FeatureEngineer._robust_normalize(
        x,
        window=3,
        min_valid_obs=3,
        return_validity=True,
    )

    assert torch.equal(valid.squeeze(1), torch.tensor([False, False, False, True]))
    assert torch.isnan(norm[:3, 0]).all()
    assert torch.allclose(norm[3, 0], torch.tensor(1.0), atol=1e-6)


def test_robust_normalize_is_causal_when_appending_future_dates():
    base = torch.tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
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
            [100.0],
            [200.0],
        ],
        dtype=torch.float32,
    )

    base_norm, base_valid = FeatureEngineer._robust_normalize(
        base,
        window=3,
        min_valid_obs=3,
        return_validity=True,
    )
    extended_norm, extended_valid = FeatureEngineer._robust_normalize(
        extended,
        window=3,
        min_valid_obs=3,
        return_validity=True,
    )

    assert torch.equal(base_valid, extended_valid[: base_valid.shape[0]])
    assert torch.allclose(base_norm, extended_norm[: base_norm.shape[0]], atol=1e-6, equal_nan=True)


def test_build_feature_tensor_can_return_feature_validity(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["CLOSE"],
        raising=False,
    )

    raw_data = {
        "CLOSE": torch.tensor(
            [
                [1.0, float("nan")],
                [2.0, 10.0],
                [3.0, 11.0],
            ],
            dtype=torch.float32,
        )
    }

    feat_tensor, valid_tensor = FeatureEngineer.compute_features(
        raw_data,
        warmup_rows=0,
        return_validity=True,
    )

    assert feat_tensor.shape == valid_tensor.shape
    assert valid_tensor.dtype == torch.bool
    assert not valid_tensor.any().item()
    assert torch.isnan(feat_tensor).all()
