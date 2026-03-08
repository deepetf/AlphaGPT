"""
配置加载模块。

支持从默认 YAML 与自定义 YAML 合并加载配置，并在启动阶段完成基础校验。
"""

import os
from typing import Any, Dict, Optional

import yaml

from .features_registry import validate_feature_names


_config_cache: Optional[Dict[str, Any]] = None
_loaded_config_path: Optional[str] = None

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    global _config_cache, _loaded_config_path

    if not os.path.exists(_DEFAULT_CONFIG_PATH):
        raise FileNotFoundError(f"默认配置文件不存在: {_DEFAULT_CONFIG_PATH}")

    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config_path:
        target_path = config_path
        if not os.path.exists(target_path):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(module_dir, config_path)
            if os.path.exists(potential_path):
                target_path = potential_path
            else:
                raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            custom_config = yaml.safe_load(f)

        if custom_config:
            config = _deep_merge(config, custom_config)

    _validate_config(config)

    _config_cache = config
    _loaded_config_path = config_path
    return config


def get_loaded_config_path() -> Optional[str]:
    return _loaded_config_path


def get_config() -> Dict[str, Any]:
    global _config_cache

    if _config_cache is None:
        load_config()

    return _config_cache


def get_input_features() -> list:
    return get_config().get("input_features", [])


def get_robust_config() -> Dict[str, Any]:
    return get_config().get("robust_config", {})


def get_config_val(key: str, default: Any = None) -> Any:
    return get_config().get(key, default)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _validate_config(config: Dict[str, Any]) -> None:
    input_features = config.get("input_features")
    if not input_features:
        raise ValueError("配置缺少 'input_features' 或列表为空")
    validate_feature_names(input_features)

    if "robust_config" not in config:
        raise ValueError("配置缺少 'robust_config'")

    batch_size = config.get("batch_size", 512)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    rc = config["robust_config"]
    required_fields = [
        "train_test_split_date",
        "rolling_window",
        "stability_k",
        "min_sharpe_val",
        "min_active_ratio",
        "min_valid_days",
        "top_k",
        "fee_rate",
        "train_weight",
        "val_weight",
        "stability_w",
        "ret_w",
        "mdd_w",
        "len_w",
        "scale",
        "entropy_beta_start",
        "entropy_beta_end",
        "diversity_pool_size",
    ]

    for field in required_fields:
        if field not in rc:
            raise ValueError(f"robust_config 缺少字段: {field}")

    if rc["top_k"] < 1:
        raise ValueError("top_k 必须 >= 1")

    if rc["fee_rate"] < 0 or rc["fee_rate"] > 0.1:
        raise ValueError("fee_rate 应在 0 ~ 0.1 范围内")

    if rc["train_weight"] + rc["val_weight"] != 1.0:
        raise ValueError("train_weight + val_weight 应等于 1.0")


def reset_config() -> None:
    global _config_cache, _loaded_config_path
    _config_cache = None
    _loaded_config_path = None
