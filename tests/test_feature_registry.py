import torch

from model_core.features_registry import (
    get_required_raw_feature_names,
    get_feature_spec,
    validate_feature_names,
)
from model_core.factors import FeatureEngineer


def test_validate_feature_names_accepts_registered_features():
    validate_feature_names(
        ["CLOSE", "LOG_MONEYNESS", "PURE_VALUE_CS_RANK", "DBLOW_CS_RANK", "DBLOW_CS_ROBUST_Z"]
    )


def test_validate_feature_names_rejects_unknown_feature():
    try:
        validate_feature_names(["CLOSE", "NOT_EXIST"])
    except ValueError as exc:
        assert "NOT_EXIST" in str(exc)
    else:
        raise AssertionError("validate_feature_names should reject unknown features")


def test_required_raw_features_expand_derived_dependencies():
    required = get_required_raw_feature_names(["LOG_MONEYNESS"])
    assert required == ("CLOSE_STK", "CONV_PRICE")


def test_required_raw_features_expand_slow_cross_sectional_dependencies():
    required = get_required_raw_feature_names(
        [
            "PURE_VALUE_CS_RANK",
            "PREM_CS_ROBUST_Z",
            "REMAIN_SIZE_CS_RANK",
            "DBLOW_CS_RANK",
            "DBLOW_CS_ROBUST_Z",
        ]
    )
    assert required == ("PURE_VALUE", "PREM", "REMAIN_SIZE", "DBLOW")


def test_feature_engineer_can_build_registered_derived_feature(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["LOG_MONEYNESS"],
        raising=False,
    )

    raw_data = {
        "CLOSE_STK": torch.tensor([[10.0, 20.0], [11.0, 22.0]], dtype=torch.float32),
        "CONV_PRICE": torch.tensor([[8.0, 16.0], [8.0, 16.0]], dtype=torch.float32),
    }

    feat_tensor = FeatureEngineer.compute_features(raw_data, warmup_rows=60)
    assert feat_tensor.shape == (2, 2, 1)
    assert torch.isfinite(feat_tensor).all()


def test_raw_feature_spec_keeps_source_metadata():
    spec = get_feature_spec("CLOSE")
    assert spec is not None
    assert spec.kind == "raw"
    assert spec.raw_column == "close"


def test_slow_cross_sectional_feature_specs_skip_time_normalization():
    for name in (
        "PURE_VALUE_CS_RANK",
        "PURE_VALUE_CS_ROBUST_Z",
        "PREM_CS_RANK",
        "PREM_CS_ROBUST_Z",
        "REMAIN_SIZE_CS_RANK",
        "CAP_MV_RATE_CS_RANK",
        "DBLOW_CS_RANK",
        "DBLOW_CS_ROBUST_Z",
    ):
        spec = get_feature_spec(name)
        assert spec is not None
        assert spec.kind == "derived"
        assert spec.apply_time_normalization is False


def test_pre_standardized_features_skip_time_normalization():
    for name in ("PREM_Z", "LOG_MONEYNESS", "ALPHA_PCT_CHG_5"):
        spec = get_feature_spec(name)
        assert spec is not None
        assert spec.apply_time_normalization is False


def test_feature_normalization_override_can_force_time_z(monkeypatch):
    monkeypatch.setattr(
        "model_core.factors.get_feature_normalization_overrides",
        lambda: {"PREM_Z": True},
    )

    raw = {
        "PREM_Z": torch.tensor(
            [[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]],
            dtype=torch.float32,
        )
    }

    expected = FeatureEngineer._robust_normalize(raw["PREM_Z"], warmup_rows=60)
    built = FeatureEngineer.build_feature_tensor(
        raw_data=raw,
        feature_names=["PREM_Z"],
        normalize=True,
        warmup_rows=60,
    )[:, :, 0]
    assert torch.allclose(built, expected, equal_nan=True)


def test_cross_sectional_feature_requires_cs_mask(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["DBLOW_CS_RANK"],
        raising=False,
    )

    raw_data = {
        "DBLOW": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
    }

    try:
        FeatureEngineer.compute_features(raw_data, warmup_rows=0)
    except ValueError as exc:
        assert "cs_mask" in str(exc)
    else:
        raise AssertionError("cross-sectional feature should require cs_mask")
