from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

from .config import ModelConfig


ComputeFeatureFn = Callable[[Callable[[str], torch.Tensor]], torch.Tensor]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str  # "raw" | "derived"
    raw_column: Optional[str] = None
    fill_method: Optional[str] = None
    deps: Tuple[str, ...] = ()
    compute_fn: Optional[ComputeFeatureFn] = None
    apply_time_normalization: bool = True


def _compute_log_moneyness(get_feature_tensor: Callable[[str], torch.Tensor]) -> torch.Tensor:
    stock_close = get_feature_tensor("CLOSE_STK")
    conv_price = get_feature_tensor("CONV_PRICE")

    valid = (stock_close > 0) & (conv_price > 0)
    ratio = torch.where(
        valid,
        stock_close / (conv_price + 1e-9),
        torch.ones_like(stock_close),
    )
    out = torch.where(
        valid,
        torch.log(torch.clamp(ratio, min=1e-12)),
        torch.zeros_like(stock_close),
    )
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _cross_sectional_rank(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 2:
        return tensor
    ranks = tensor.argsort(dim=1).argsort(dim=1).float()
    denom = tensor.shape[1] - 1 + 1e-9
    out = ranks / denom
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _cross_sectional_robust_z(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 2:
        return tensor
    median = tensor.median(dim=1, keepdim=True).values
    mad = (tensor - median).abs().median(dim=1, keepdim=True).values + 1e-9
    out = (tensor - median) / (mad * 1.4826)
    out = torch.clamp(out, -5.0, 5.0)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _make_cs_rank_feature(dep_name: str) -> ComputeFeatureFn:
    def _compute(get_feature_tensor: Callable[[str], torch.Tensor]) -> torch.Tensor:
        return _cross_sectional_rank(get_feature_tensor(dep_name))

    return _compute


def _make_cs_robust_z_feature(dep_name: str) -> ComputeFeatureFn:
    def _compute(get_feature_tensor: Callable[[str], torch.Tensor]) -> torch.Tensor:
        return _cross_sectional_robust_z(get_feature_tensor(dep_name))

    return _compute


def _build_registry() -> Dict[str, FeatureSpec]:
    registry: Dict[str, FeatureSpec] = {}
    skip_time_norm_features = {
        "PREM_Z",
        "ALPHA_PCT_CHG_5",
    }

    for internal_name, raw_column, fill_method in ModelConfig.BASIC_FACTORS:
        registry[internal_name] = FeatureSpec(
            name=internal_name,
            kind="raw",
            raw_column=raw_column,
            fill_method=fill_method,
            apply_time_normalization=internal_name not in skip_time_norm_features,
        )

    derived_specs = (
        FeatureSpec(
            name="LOG_MONEYNESS",
            kind="derived",
            deps=("CLOSE_STK", "CONV_PRICE"),
            compute_fn=_compute_log_moneyness,
            apply_time_normalization=False,
        ),
        FeatureSpec(
            name="PURE_VALUE_CS_RANK",
            kind="derived",
            deps=("PURE_VALUE",),
            compute_fn=_make_cs_rank_feature("PURE_VALUE"),
            apply_time_normalization=False,
        ),
        FeatureSpec(
            name="PURE_VALUE_CS_ROBUST_Z",
            kind="derived",
            deps=("PURE_VALUE",),
            compute_fn=_make_cs_robust_z_feature("PURE_VALUE"),
            apply_time_normalization=False,
        ),
        FeatureSpec(
            name="PREM_CS_RANK",
            kind="derived",
            deps=("PREM",),
            compute_fn=_make_cs_rank_feature("PREM"),
            apply_time_normalization=False,
        ),
        FeatureSpec(
            name="PREM_CS_ROBUST_Z",
            kind="derived",
            deps=("PREM",),
            compute_fn=_make_cs_robust_z_feature("PREM"),
            apply_time_normalization=False,
        ),
        FeatureSpec(
            name="REMAIN_SIZE_CS_RANK",
            kind="derived",
            deps=("REMAIN_SIZE",),
            compute_fn=_make_cs_rank_feature("REMAIN_SIZE"),
            apply_time_normalization=False,
        ),
        FeatureSpec(
            name="CAP_MV_RATE_CS_RANK",
            kind="derived",
            deps=("CAP_MV_RATE",),
            compute_fn=_make_cs_rank_feature("CAP_MV_RATE"),
            apply_time_normalization=False,
        ),
        FeatureSpec(
            name="DBLOW_CS_RANK",
            kind="derived",
            deps=("DBLOW",),
            compute_fn=_make_cs_rank_feature("DBLOW"),
            apply_time_normalization=False,
        ),
        FeatureSpec(
            name="DBLOW_CS_ROBUST_Z",
            kind="derived",
            deps=("DBLOW",),
            compute_fn=_make_cs_robust_z_feature("DBLOW"),
            apply_time_normalization=False,
        ),
    )
    for spec in derived_specs:
        registry[spec.name] = spec

    return registry


FEATURE_REGISTRY: Dict[str, FeatureSpec] = _build_registry()


def get_feature_spec(feature_name: str) -> Optional[FeatureSpec]:
    return FEATURE_REGISTRY.get(feature_name)


def list_registered_features() -> List[str]:
    return list(FEATURE_REGISTRY.keys())


def list_derived_feature_names() -> List[str]:
    return [name for name, spec in FEATURE_REGISTRY.items() if spec.kind == "derived"]


def validate_feature_names(feature_names: Iterable[str]) -> None:
    unknown = [name for name in feature_names if name not in FEATURE_REGISTRY]
    if not unknown:
        return

    known = ", ".join(sorted(FEATURE_REGISTRY.keys()))
    unknown_str = ", ".join(unknown)
    raise ValueError(
        f"Unknown input_features: {unknown_str}. "
        f"Registered features: {known}"
    )


def get_required_raw_feature_names(feature_names: Iterable[str]) -> Tuple[str, ...]:
    ordered: List[str] = []
    seen = set()

    def visit(feature_name: str) -> None:
        spec = FEATURE_REGISTRY.get(feature_name)
        if spec is None:
            raise KeyError(f"Feature '{feature_name}' is not registered")

        if spec.kind == "raw":
            if feature_name not in seen:
                seen.add(feature_name)
                ordered.append(feature_name)
            return

        for dep in spec.deps:
            visit(dep)

    for feature_name in feature_names:
        visit(feature_name)

    return tuple(ordered)
