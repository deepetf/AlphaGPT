import copy
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - fallback path
    pq = None


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


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_manifest_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    project_root = _project_root()
    abs_path = os.path.abspath(path)
    try:
        common = os.path.commonpath([project_root, abs_path])
    except ValueError:
        common = None
    if common == project_root:
        return os.path.relpath(abs_path, project_root)
    return abs_path


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_cb_parquet_path(config: Dict[str, Any]) -> str:
    parquet_path = config.get("cb_parquet_path")
    if parquet_path:
        resolved = _resolve_path(str(parquet_path))
        if resolved:
            return resolved
    return _resolve_path(r"C:\Trading\Projects\AlphaGPT\data\cb_data.pq") or r"C:\Trading\Projects\AlphaGPT\data\cb_data.pq"


def _schema_hash_from_parquet(path: str) -> Tuple[str, Optional[int]]:
    if pq is None:
        return "", None
    parquet_file = pq.ParquetFile(path)
    schema_payload = []
    schema = parquet_file.schema_arrow
    for field in schema:
        schema_payload.append(
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": bool(field.nullable),
            }
        )
    return _stable_hash(schema_payload), parquet_file.metadata.num_rows


def _load_parquet_frame(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _extract_data_snapshot(
    *,
    config: Dict[str, Any],
    data_start_date: Optional[str],
) -> Dict[str, Any]:
    parquet_path = _resolve_cb_parquet_path(config)
    if not parquet_path or not os.path.exists(parquet_path):
        raise FileNotFoundError(f"训练数据文件不存在: {parquet_path}")

    schema_hash, row_count = _schema_hash_from_parquet(parquet_path)
    frame = _load_parquet_frame(parquet_path)
    if row_count is None:
        row_count = int(len(frame))

    date_col = "trade_date" if "trade_date" in frame.columns else ("date" if "date" in frame.columns else None)
    code_col = "code" if "code" in frame.columns else None
    max_date = None
    asset_count = None
    if date_col and not frame.empty:
        date_series = pd.to_datetime(frame[date_col], errors="coerce")
        if date_series.notna().any():
            max_date = date_series.max().strftime("%Y-%m-%d")
    if code_col and not frame.empty:
        asset_count = int(frame[code_col].nunique(dropna=True))

    feature_payload = {
        "input_features": config.get("input_features", []),
        "feature_normalization_overrides": config.get("feature_normalization_overrides", {}) or {},
    }

    return {
        "parquet_path": _to_manifest_path(parquet_path),
        "file_hash": _sha256_file(parquet_path),
        "schema_hash": schema_hash or _stable_hash(list(frame.columns)),
        "row_count": int(row_count),
        "asset_count": int(asset_count or 0),
        "max_date": max_date,
        "train_date_range": {
            "data_start_date": data_start_date or "2022-08-01",
            "train_test_split_date": ((config.get("robust_config") or {}).get("train_test_split_date")),
        },
        "feature_config_hash": _stable_hash(feature_payload),
        "code_commit": _resolve_git_commit(),
    }


def _resolve_git_commit() -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_project_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    value = (output or "").strip()
    return value or None


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    resolved_manifest_path = _resolve_path(manifest_path) or manifest_path
    if not os.path.exists(resolved_manifest_path):
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    return _read_json(resolved_manifest_path)


def validate_run_data_snapshot(
    manifest_path: str,
    *,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_manifest_path = _resolve_path(manifest_path) or manifest_path
    manifest = load_manifest(resolved_manifest_path)
    snapshot = manifest.get("data_snapshot") or {}
    if not snapshot:
        raise ValueError(f"manifest 缺少 data_snapshot: {resolved_manifest_path}")

    manifest_parquet_path = _resolve_path(snapshot.get("parquet_path")) or snapshot.get("parquet_path")
    if not manifest_parquet_path or not os.path.exists(manifest_parquet_path):
        raise FileNotFoundError(f"manifest 绑定的数据文件不存在: {snapshot.get('parquet_path')}")

    if config_path:
        resolved_config_path = _resolve_path(config_path) or config_path
        config = _read_yaml(resolved_config_path)
        config_parquet_path = _resolve_cb_parquet_path(config)
        if os.path.abspath(config_parquet_path) != os.path.abspath(manifest_parquet_path):
            raise ValueError(
                "当前阶段提供的配置指向了不同数据文件: "
                f"manifest={manifest_parquet_path}, config={config_parquet_path}"
            )

    current_snapshot = _extract_data_snapshot(
        config={"cb_parquet_path": manifest_parquet_path},
        data_start_date=(snapshot.get("train_date_range") or {}).get("data_start_date"),
    )
    compare_fields = ("file_hash", "schema_hash", "row_count", "asset_count", "max_date")
    mismatches = []
    for field in compare_fields:
        if current_snapshot.get(field) != snapshot.get(field):
            mismatches.append((field, snapshot.get(field), current_snapshot.get(field)))
    if mismatches:
        details = "; ".join([f"{field}: expected={expected}, current={current}" for field, expected, current in mismatches])
        raise ValueError(
            "run 数据快照校验失败，当前数据文件与训练时不一致: "
            f"{details}"
        )

    return snapshot


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
        "data_snapshot": _extract_data_snapshot(
            config=config,
            data_start_date=data_start_date,
        ),
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
