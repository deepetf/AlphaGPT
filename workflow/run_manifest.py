import copy
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

import yaml


_DEFAULT_ARTIFACTS_ROOT = os.path.join("artifacts", "runs")
_SAFE_RUN_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    project_root = _project_root()
    direct = os.path.abspath(os.path.join(project_root, path))
    if os.path.exists(direct):
        return direct
    model_core_relative = os.path.abspath(os.path.join(project_root, "model_core", path))
    if os.path.exists(model_core_relative):
        return model_core_relative
    return direct


def _default_model_config_path() -> str:
    return os.path.join(_project_root(), "model_core", "default_config.yaml")


def _safe_run_id(text: str) -> str:
    candidate = _SAFE_RUN_ID_PATTERN.sub("_", (text or "").strip()).strip("._-")
    if not candidate:
        raise ValueError("run_id 不能为空")
    return candidate


def generate_run_id(config_path: Optional[str] = None, explicit_run_id: Optional[str] = None) -> str:
    if explicit_run_id:
        return _safe_run_id(explicit_run_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = os.path.splitext(os.path.basename(config_path or "default_config"))[0]
    return f"{timestamp}_{_safe_run_id(config_name)}"


def _to_relpath(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.relpath(path, _project_root())


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_yaml(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def prepare_training_run(
    *,
    config: Dict[str, Any],
    config_path: Optional[str],
    data_start_date: Optional[str],
    run_id: Optional[str] = None,
    artifacts_root: Optional[str] = None,
) -> Dict[str, Any]:
    project_root = _project_root()
    artifacts_root_abs = _resolve_path(artifacts_root) if artifacts_root else os.path.join(project_root, _DEFAULT_ARTIFACTS_ROOT)
    run_id_val = generate_run_id(config_path=config_path, explicit_run_id=run_id)
    run_dir = os.path.join(artifacts_root_abs, run_id_val)
    train_dir = os.path.join(run_dir, "train")
    king_trades_dir = os.path.join(train_dir, "king_trades")
    os.makedirs(king_trades_dir, exist_ok=True)

    resolved_config_path = _resolve_path(config_path) or _default_model_config_path()
    resolved_config_snapshot_path = os.path.join(run_dir, "resolved_model_config.yaml")
    _write_yaml(resolved_config_snapshot_path, copy.deepcopy(config))

    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {
        "run_id": run_id_val,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": "initialized",
        "project_root": project_root,
        "training": {
            "entry": "python -m model_core.engine",
            "config_path": _to_relpath(resolved_config_path),
            "requested_config_path": config_path,
            "resolved_config_snapshot_path": _to_relpath(resolved_config_snapshot_path),
            "data_start_date": data_start_date or "2022-08-01",
        },
        "artifacts": {
            "run_dir": _to_relpath(run_dir),
            "train_dir": _to_relpath(train_dir),
            "king_trades_dir": _to_relpath(king_trades_dir),
        },
    }
    _write_json(manifest_path, manifest)

    return {
        "run_id": run_id_val,
        "project_root": project_root,
        "run_dir": run_dir,
        "train_dir": train_dir,
        "king_trades_dir": king_trades_dir,
        "manifest_path": manifest_path,
        "resolved_config_snapshot_path": resolved_config_snapshot_path,
    }


def update_training_manifest(
    run_context: Dict[str, Any],
    *,
    stage: str,
    artifacts: Optional[Dict[str, Optional[str]]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    manifest_path = run_context.get("manifest_path")
    if not manifest_path:
        return

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"run_id": run_context.get("run_id")}

    manifest["stage"] = stage
    manifest["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    if artifacts:
        manifest_artifacts = manifest.setdefault("artifacts", {})
        for key, value in artifacts.items():
            manifest_artifacts[key] = _to_relpath(value) if isinstance(value, str) else value

    if summary:
        manifest["summary"] = summary

    _write_json(manifest_path, manifest)
