from pathlib import Path

import pytest
import torch

from model_core.features_registry import FEATURE_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_CHAIN_FILES = (
    "model_core/data_loader.py",
    "model_core/factors.py",
    "model_core/features_registry.py",
    "model_core/vm.py",
    "model_core/ops_registry.py",
    "model_core/engine.py",
    "model_core/select_top_factors.py",
)
FORBIDDEN_ZERO_FILL_PATTERNS = (
    "nan_to_num(",
    ".fillna(0)",
    ".fillna(0.0)",
)


def test_all_registered_cross_sectional_features_require_cs_mask():
    raw_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    for feature_name, spec in FEATURE_REGISTRY.items():
        if spec.kind != "derived" or "_CS_" not in feature_name:
            continue
        assert spec.compute_fn is not None
        with pytest.raises(ValueError, match="cs_mask"):
            spec.compute_fn(lambda _: raw_tensor, lambda: None)


def test_main_chain_files_do_not_zero_fill_invalid_values():
    for rel_path in MAIN_CHAIN_FILES:
        content = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
        for pattern in FORBIDDEN_ZERO_FILL_PATTERNS:
            assert pattern not in content, f"{rel_path} contains forbidden pattern: {pattern}"
