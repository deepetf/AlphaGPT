import torch

from model_core.features_registry import (
    get_required_raw_feature_names,
    get_feature_spec,
    validate_feature_names,
)
from model_core.factors import FeatureEngineer


def test_validate_feature_names_accepts_registered_features():
    validate_feature_names(["CLOSE", "LOG_MONEYNESS", "PURE_VALUE_CS_RANK"])


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
        ["PURE_VALUE_CS_RANK", "PREM_CS_ROBUST_Z", "REMAIN_SIZE_CS_RANK"]
    )
    assert required == ("PURE_VALUE", "PREM", "REMAIN_SIZE")


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
    ):
        spec = get_feature_spec(name)
        assert spec is not None
        assert spec.kind == "derived"
        assert spec.apply_time_normalization is False
