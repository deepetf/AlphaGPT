"""
特征工程模块 (Feature Engineering)

使用统一注册中心解析 raw / derived feature，确保训练、verify、sim
主链路共享同一套特征定义与计算逻辑。
"""

from typing import Optional

import torch

from .config import ModelConfig
from .config_loader import get_feature_normalization_overrides
from .features_registry import (
    get_feature_spec,
    get_required_raw_feature_names,
    list_derived_feature_names,
)


class FeatureEngineer:
    """
    动态特征工程器

    根据 ModelConfig.INPUT_FEATURES 自动构建输入张量。
    """

    @staticmethod
    def _resolve_feature_tensor(
        feat_name: str,
        raw_data: dict,
        feature_cache: dict,
        configured_features: list,
        cross_sectional_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if feat_name in feature_cache:
            return feature_cache[feat_name]

        spec = get_feature_spec(feat_name)
        if spec is None:
            raise KeyError(
                f"Feature '{feat_name}' is not registered. "
                f"Configured features: {configured_features}"
            )

        if spec.kind == "raw":
            if feat_name not in raw_data:
                required_raw = get_required_raw_feature_names(configured_features)
                raise KeyError(
                    f"Raw feature '{feat_name}' not found in raw_data. "
                    f"Required raw: {list(required_raw)}, "
                    f"available raw: {list(raw_data.keys())}"
                )
            tensor = raw_data[feat_name]
        elif spec.compute_fn is not None:
            tensor = spec.compute_fn(
                lambda dep_name: FeatureEngineer._resolve_feature_tensor(
                    dep_name,
                    raw_data,
                    feature_cache,
                    configured_features,
                    cross_sectional_mask=cross_sectional_mask,
                )
                ,
                lambda: cross_sectional_mask,
            )
        else:
            raise KeyError(
                f"Derived feature '{feat_name}' has no compute_fn. "
                f"Available derived: {list_derived_feature_names()}"
            )

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Feature '{feat_name}' must be torch.Tensor, got {type(tensor)}")
        if tensor.dim() != 2:
            raise ValueError(
                f"Feature '{feat_name}' must be 2D [Time, Assets], got shape={tuple(tensor.shape)}"
            )

        feature_cache[feat_name] = tensor
        return tensor

    @staticmethod
    def build_feature_tensor(
        raw_data: dict,
        feature_names: list | None = None,
        normalize: bool = True,
        warmup_rows: int = 0,
        cross_sectional_mask: Optional[torch.Tensor] = None,
        return_validity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        从原始数据中构建特征张量。

        Args:
            raw_data: dict[str, Tensor[Time, Assets]]
            feature_names: 需要构建的特征列表，默认使用 ModelConfig.INPUT_FEATURES
            normalize: 是否做滚动稳健标准化
            warmup_rows: 预热区间的真实行数

        Returns:
            feat_tensor: [Time, Assets, Features]
        """
        feature_names = list(feature_names or ModelConfig.INPUT_FEATURES)
        features = []
        validities = []
        feature_cache = {}
        normalization_overrides = get_feature_normalization_overrides() if normalize else {}
        for feat_name in feature_names:
            spec = get_feature_spec(feat_name)
            if spec is None:
                raise KeyError(f"Feature '{feat_name}' is not registered")
            feat = FeatureEngineer._resolve_feature_tensor(
                feat_name,
                raw_data,
                feature_cache,
                feature_names,
                cross_sectional_mask=cross_sectional_mask,
            )
            apply_time_normalization = normalization_overrides.get(
                feat_name, spec.apply_time_normalization
            )
            if normalize and apply_time_normalization:
                feat, feature_valid = FeatureEngineer._robust_normalize(
                    feat,
                    warmup_rows=warmup_rows,
                    return_validity=True,
                )
            else:
                feature_valid = torch.isfinite(feat)
            features.append(feat)
            validities.append(feature_valid)

        feat_tensor = torch.stack(features, dim=2)
        validity_tensor = torch.stack(validities, dim=2)
        if return_validity:
            return feat_tensor, validity_tensor
        return feat_tensor

    @staticmethod
    def compute_features(
        raw_data: dict,
        warmup_rows: int = 0,
        cross_sectional_mask: Optional[torch.Tensor] = None,
        return_validity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return FeatureEngineer.build_feature_tensor(
            raw_data=raw_data,
            feature_names=list(ModelConfig.INPUT_FEATURES),
            normalize=True,
            warmup_rows=warmup_rows,
            cross_sectional_mask=cross_sectional_mask,
            return_validity=return_validity,
        )

    @classmethod
    def get_optional_raw_feature_names(cls):
        """
        返回所有派生特征依赖的 raw feature 集合。
        SQLStrictLoader 会用它来决定哪些原始列可按“可选列”处理。
        """
        return set(get_required_raw_feature_names(list_derived_feature_names()))

    @staticmethod
    def _robust_normalize(
        t: torch.Tensor,
        window: int = 60,
        warmup_rows: int = 0,
        min_valid_obs: Optional[int] = None,
        return_validity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        严格因果的滚动标准化。

        规则:
        - 仅使用 <= t 的历史样本
        - 仅使用窗口内有限值参与统计
        - 当前样本非有限值时，输出无效
        - 有效样本数不足时，输出无效
        """
        del warmup_rows
        if t.dim() != 2:
            feature_valid = torch.isfinite(t)
            if return_validity:
                return t, feature_valid
            return t

        T, _ = t.shape
        min_valid_obs = int(min_valid_obs or window)
        if T == 0:
            feature_valid = torch.zeros_like(t, dtype=torch.bool)
            if return_validity:
                return t, feature_valid
            return t

        padding = torch.full((window - 1, t.shape[1]), float("nan"), device=t.device, dtype=t.dtype)
        padded = torch.cat([padding, t], dim=0)
        unfolded = padded.unfold(0, window, 1)

        hist_valid = torch.isfinite(unfolded)
        valid_count = hist_valid.sum(dim=-1)
        safe_windows = torch.where(hist_valid, unfolded, torch.zeros_like(unfolded))
        roll_sum = safe_windows.sum(dim=-1)
        roll_mean = roll_sum / valid_count.clamp_min(1).to(t.dtype)

        centered = torch.where(hist_valid, unfolded - roll_mean.unsqueeze(-1), torch.zeros_like(unfolded))
        denom = (valid_count - 1).clamp_min(1).to(t.dtype)
        roll_var = centered.pow(2).sum(dim=-1) / denom
        roll_std = torch.sqrt(roll_var).clamp_min(1e-9)

        current_valid = torch.isfinite(t)
        feature_valid = current_valid & (valid_count >= min_valid_obs)
        norm = torch.full_like(t, float("nan"))
        norm[feature_valid] = ((t - roll_mean) / roll_std)[feature_valid]
        norm = torch.clamp(norm, -5.0, 5.0)

        if return_validity:
            return norm, feature_valid
        return norm


class DerivedFeatures:
    """
    预留的派生特征工具函数。

    这些函数本身不会自动进入 INPUT_FEATURES；
    若要正式启用，应在 features_registry 中注册。
    """

    @staticmethod
    def compute_return(close: torch.Tensor, lag: int = 1) -> torch.Tensor:
        prev = torch.roll(close, lag, dims=0)
        ret = (close - prev) / (prev + 1e-9)
        ret[:lag] = 0
        return ret

    @staticmethod
    def compute_momentum(close: torch.Tensor, window: int = 5) -> torch.Tensor:
        prev = torch.roll(close, window, dims=0)
        mom = (close / (prev + 1e-9)) - 1.0
        mom[:window] = 0
        return mom

    @staticmethod
    def compute_volatility(close: torch.Tensor, window: int = 20) -> torch.Tensor:
        if close.shape[0] < window:
            return torch.zeros_like(close)

        ret = DerivedFeatures.compute_return(close, lag=1)

        padded = torch.cat([ret[: window - 1], ret], dim=0)
        unfolded = padded.unfold(0, window, 1)
        vol = unfolded.std(dim=-1)
        return vol
