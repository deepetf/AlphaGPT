import json
import os
from typing import Any, Dict, List


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_project_path(path: str) -> str:
    if not path:
        raise ValueError("路径不能为空")
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root(), path))


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_formula(formula_obj: Dict[str, Any], formula_path: str) -> List[str]:
    formula = formula_obj.get("formula")
    if isinstance(formula, list) and formula and isinstance(formula[0], str):
        return formula
    best = formula_obj.get("best")
    if isinstance(best, dict):
        best_formula = best.get("formula")
        if isinstance(best_formula, list) and best_formula and isinstance(best_formula[0], str):
            return best_formula
    raise ValueError(f"无法从 bundle 引用的公式文件解析 formula: {formula_path}")


def load_strategy_bundle(bundle_path: str) -> Dict[str, Any]:
    bundle_abs = resolve_project_path(bundle_path)
    if not os.path.exists(bundle_abs):
        raise FileNotFoundError(f"bundle 不存在: {bundle_path}")

    bundle = _read_json(bundle_abs)
    formula_path_raw = bundle.get("formula_path")
    model_config_path_raw = bundle.get("model_config_path")
    if not isinstance(formula_path_raw, str) or not formula_path_raw.strip():
        raise ValueError(f"bundle 缺少 formula_path: {bundle_path}")
    if not isinstance(model_config_path_raw, str) or not model_config_path_raw.strip():
        raise ValueError(f"bundle 缺少 model_config_path: {bundle_path}")

    formula_abs = resolve_project_path(formula_path_raw)
    model_config_abs = resolve_project_path(model_config_path_raw)
    if not os.path.exists(formula_abs):
        raise FileNotFoundError(f"bundle 引用的公式文件不存在: {formula_abs}")
    if not os.path.exists(model_config_abs):
        raise FileNotFoundError(f"bundle 引用的模型配置不存在: {model_config_abs}")

    formula_obj = _read_json(formula_abs)
    formula = _extract_formula(formula_obj, formula_abs)
    params = bundle.get("params", {}) or {}
    generated_strategy_config_abs = os.path.join(os.path.dirname(bundle_abs), "generated_strategy_config.json")
    if not os.path.exists(generated_strategy_config_abs):
        generated_strategy_config_abs = None

    return {
        "bundle_path": bundle_abs,
        "bundle": bundle,
        "bundle_dir": os.path.dirname(bundle_abs),
        "formula_path": formula_abs,
        "formula": formula,
        "model_config_path": model_config_abs,
        "strategy_id": bundle.get("strategy_id"),
        "strategy_name": bundle.get("strategy_name"),
        "params": params,
        "formula_summary": bundle.get("formula_summary", {}) or {},
        "source": bundle.get("source", {}) or {},
        "generated_strategy_config_path": generated_strategy_config_abs,
    }
