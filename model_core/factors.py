"""
特征工程模块 (Feature Engineering)

使用统一注册中心解析 raw / derived feature，确保训练、verify、sim
主链路共享同一套特征定义与计算逻辑。
"""

import torch

from .config import ModelConfig
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
                )
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
    ) -> torch.Tensor:
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
        feature_cache = {}
        for feat_name in feature_names:
            spec = get_feature_spec(feat_name)
            if spec is None:
                raise KeyError(f"Feature '{feat_name}' is not registered")
            feat = FeatureEngineer._resolve_feature_tensor(
                feat_name,
                raw_data,
                feature_cache,
                feature_names,
            )
            if normalize and spec.apply_time_normalization:
                feat = FeatureEngineer._robust_normalize(feat, warmup_rows=warmup_rows)
            features.append(feat)

        feat_tensor = torch.stack(features, dim=2)
        return feat_tensor

    @staticmethod
    def compute_features(raw_data: dict, warmup_rows: int = 0) -> torch.Tensor:
        return FeatureEngineer.build_feature_tensor(
            raw_data=raw_data,
            feature_names=list(ModelConfig.INPUT_FEATURES),
            normalize=True,
            warmup_rows=warmup_rows,
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
    ) -> torch.Tensor:
        """
        滚动稳健标准化。
        """
        if t.dim() != 2:
            return t

        T, _ = t.shape
        if T < window:
            return torch.zeros_like(t)

        padding = t[0].unsqueeze(0).repeat(window - 1, 1)
        padded = torch.cat([padding, t], dim=0)

        unfolded = padded.unfold(0, window, 1)
        roll_mean = unfolded.mean(dim=-1)
        roll_std = unfolded.std(dim=-1) + 1e-9

        norm = (t - roll_mean) / roll_std

        zero_rows = max(0, window - warmup_rows)
        if zero_rows > 0:
            norm[:zero_rows] = 0.0

        return torch.clamp(norm, -5.0, 5.0)


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
