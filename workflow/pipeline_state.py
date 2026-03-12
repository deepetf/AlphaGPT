import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


STAGE_ORDER = ["train", "select", "bundle", "verify", "sim"]


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root(), path))


def status_path_for_run(run_dir: str) -> str:
    return os.path.join(run_dir, "pipeline_status.json")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def init_pipeline_status(
    *,
    run_dir: str,
    run_id: str,
    command: str,
    requested_stages: List[str],
    resume: bool,
) -> Dict[str, Any]:
    path = status_path_for_run(run_dir)
    if os.path.exists(path):
        data = _read_json(path)
    else:
        data = {
            "run_id": run_id,
            "created_at": _now(),
            "pipeline": {
                "command": command,
                "resume_enabled": bool(resume),
                "requested_stages": requested_stages,
            },
            "stages": {
                stage: {
                    "status": "pending",
                    "attempts": 0,
                }
                for stage in STAGE_ORDER
            },
        }
    data.setdefault("pipeline", {})
    data["pipeline"]["command"] = command
    data["pipeline"]["resume_enabled"] = bool(resume)
    data["pipeline"]["requested_stages"] = requested_stages
    data["updated_at"] = _now()
    _write_json(path, data)
    return data


def load_pipeline_status(run_dir: str) -> Optional[Dict[str, Any]]:
    path = status_path_for_run(run_dir)
    if not os.path.exists(path):
        return None
    return _read_json(path)


def update_stage_status(
    *,
    run_dir: str,
    stage: str,
    status: str,
    command: Optional[str] = None,
    error: Optional[str] = None,
    outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = status_path_for_run(run_dir)
    data = _read_json(path) if os.path.exists(path) else {"stages": {}}
    stages = data.setdefault("stages", {})
    stage_info = stages.setdefault(stage, {"status": "pending", "attempts": 0})

    previous_status = stage_info.get("status")
    stage_info["status"] = status
    stage_info["updated_at"] = _now()
    if command:
        stage_info["command"] = command
    if outputs:
        stage_info["outputs"] = outputs
    if error:
        stage_info["error"] = error
    elif "error" in stage_info and status != "failed":
        stage_info.pop("error", None)

    if status == "running":
        stage_info["attempts"] = int(stage_info.get("attempts", 0)) + (0 if previous_status == "running" else 1)
        stage_info["started_at"] = _now()
    elif status in {"completed", "failed", "skipped"}:
        stage_info["ended_at"] = _now()

    data["updated_at"] = _now()
    _write_json(path, data)
    return data


def should_skip_stage(run_dir: str, stage: str, resume: bool) -> bool:
    if not resume:
        return False
    data = load_pipeline_status(run_dir)
    if not data:
        return False
    stage_info = (data.get("stages") or {}).get(stage) or {}
    return stage_info.get("status") == "completed"


def mark_pipeline_finished(run_dir: str, *, status: str, error: Optional[str] = None) -> Dict[str, Any]:
    path = status_path_for_run(run_dir)
    data = _read_json(path) if os.path.exists(path) else {}
    pipeline_info = data.setdefault("pipeline", {})
    pipeline_info["status"] = status
    pipeline_info["finished_at"] = _now()
    if error:
        pipeline_info["error"] = error
    elif "error" in pipeline_info and status != "failed":
        pipeline_info.pop("error", None)
    data["updated_at"] = _now()
    _write_json(path, data)
    return data
