import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import yaml

from .run_manifest import update_training_manifest


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_rel(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_project_root(), path))


def _to_rel(path: str) -> str:
    return os.path.relpath(path, _project_root())


def load_manifest(manifest_path: str) -> Tuple[Dict[str, Any], str]:
    manifest_abs = _resolve_rel(manifest_path)
    if not manifest_abs or not os.path.exists(manifest_abs):
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    manifest = _read_json(manifest_abs)
    run_dir = os.path.dirname(manifest_abs)
    return manifest, run_dir


def infer_selection_output(run_dir: str, explicit_path: Optional[str] = None) -> Optional[str]:
    if explicit_path:
        resolved = _resolve_rel(explicit_path)
        if not resolved or not os.path.exists(resolved):
            raise FileNotFoundError(f"selection output 不存在: {explicit_path}")
        return resolved

    candidates = [
        os.path.join(run_dir, "selection", "top_candidates.json"),
        os.path.join(run_dir, "selection", "top3_factors.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def infer_best_formula_path(manifest: Dict[str, Any], run_dir: str) -> str:
    artifacts = manifest.get("artifacts", {})
    best_formula_rel = artifacts.get("best_formula_path")
    if isinstance(best_formula_rel, str):
        path = _resolve_rel(best_formula_rel)
        if path and os.path.exists(path):
            return path

    candidates = [
        os.path.join(run_dir, "train", "best_cb_formula.json"),
        os.path.join(_project_root(), "model_core", "best_cb_formula.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("无法定位 best_cb_formula.json")


def pick_candidate(
    *,
    source: str,
    manifest: Dict[str, Any],
    run_dir: str,
    selection_output_path: Optional[str],
    candidate_rank: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if candidate_rank < 1:
        raise ValueError("candidate_rank 必须 >= 1")

    if source == "selected":
        if not selection_output_path:
            raise FileNotFoundError("source=selected 但未找到 selection output")
        selection_obj = _read_json(selection_output_path)
        selected = selection_obj.get("selected", [])
        idx = candidate_rank - 1
        if idx >= len(selected):
            raise IndexError(
                f"selected 候选不足: candidate_rank={candidate_rank}, available={len(selected)}"
            )
        candidate = dict(selected[idx])
        meta = {
            "source_type": "selected",
            "selection_output_path": selection_output_path,
            "candidate_rank": candidate_rank,
            "selection_score": candidate.get("selection_score"),
            "step": candidate.get("step"),
        }
        return candidate, meta

    if source == "best":
        best_path = infer_best_formula_path(manifest, run_dir)
        best_obj = _read_json(best_path)
        best = dict(best_obj.get("best") or {})
        formula = best.get("formula")
        if not isinstance(formula, list) or not formula:
            raise ValueError(f"best formula 格式错误: {best_path}")
        candidate = {
            "formula": formula,
            "readable": best.get("readable") or " ".join(formula),
            "raw_formula": best.get("raw_formula"),
            "raw_readable": best.get("raw_readable"),
            "score": best.get("score"),
            "sharpe": best.get("sharpe"),
            "annualized_ret": best.get("annualized_ret"),
        }
        meta = {
            "source_type": "best",
            "best_formula_path": best_path,
            "candidate_rank": 1,
        }
        return candidate, meta

    raise ValueError(f"未知 source: {source}")


def build_bundle(
    *,
    manifest_path: str,
    selection_output_path: Optional[str] = None,
    source: Optional[str] = None,
    candidate_rank: int = 1,
    strategy_id: Optional[str] = None,
    strategy_name: Optional[str] = None,
    top_k: Optional[int] = None,
    fee_rate: Optional[float] = None,
    take_profit: Optional[float] = None,
    initial_capital: Optional[float] = None,
    state_backend: str = "sql",
    replay_source: str = "sql_eod",
    replay_strict: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    manifest, run_dir = load_manifest(manifest_path)
    selection_output_abs = infer_selection_output(run_dir, explicit_path=selection_output_path)
    actual_source = source or ("selected" if selection_output_abs else "best")

    candidate, source_meta = pick_candidate(
        source=actual_source,
        manifest=manifest,
        run_dir=run_dir,
        selection_output_path=selection_output_abs,
        candidate_rank=candidate_rank,
    )

    training_cfg_rel = (
        manifest.get("training", {}).get("resolved_config_snapshot_path")
        or manifest.get("training", {}).get("config_path")
    )
    training_cfg_abs = _resolve_rel(training_cfg_rel)
    if not training_cfg_abs or not os.path.exists(training_cfg_abs):
        raise FileNotFoundError(f"无法定位训练配置快照: {training_cfg_rel}")
    training_cfg = _read_yaml(training_cfg_abs)
    robust_cfg = training_cfg.get("robust_config", {})

    run_id = str(manifest.get("run_id") or os.path.basename(run_dir))
    strategy_id_val = strategy_id or f"{run_id}_top{candidate_rank}"
    strategy_name_val = strategy_name or strategy_id_val
    bundle_dir = _resolve_rel(output_dir) if output_dir else os.path.join(run_dir, "bundle")
    os.makedirs(bundle_dir, exist_ok=True)

    formula = candidate.get("formula")
    readable = candidate.get("readable") or " ".join(formula)
    formula_path = os.path.join(bundle_dir, "formula_top1.json")
    formula_payload = {
        "formula": formula,
        "readable": readable,
        "source": source_meta,
        "run_id": run_id,
    }
    _write_json(formula_path, formula_payload)

    params = {
        "initial_capital": float(initial_capital if initial_capital is not None else 1_000_000.0),
        "top_k": int(top_k if top_k is not None else robust_cfg.get("top_k", 5)),
        "take_profit_ratio": float(
            take_profit if take_profit is not None else robust_cfg.get("take_profit", 0.0)
        ),
        "fee_rate": float(fee_rate if fee_rate is not None else robust_cfg.get("fee_rate", 0.001)),
        "replay_strict": bool(replay_strict),
        "replay_source": replay_source,
        "state_backend": state_backend,
    }

    bundle = {
        "bundle_version": "v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "strategy_id": strategy_id_val,
        "strategy_name": strategy_name_val,
        "formula_path": _to_rel(formula_path),
        "model_config_path": _to_rel(training_cfg_abs),
        "source": {
            "manifest_path": _to_rel(_resolve_rel(manifest_path)),
            **{k: _to_rel(v) if isinstance(v, str) and os.path.exists(v) else v for k, v in source_meta.items()},
        },
        "formula_summary": {
            "readable": readable,
            "score": candidate.get("score"),
            "selection_score": candidate.get("selection_score"),
            "sharpe": candidate.get("sharpe") or candidate.get("sharpe_all"),
            "annualized_ret": candidate.get("annualized_ret"),
        },
        "params": params,
    }
    bundle_path = os.path.join(bundle_dir, "strategy_bundle.json")
    _write_json(bundle_path, bundle)

    generated_strategy_config = {
        "global": {
            "data_source": "mini_qmt",
            "log_level": "INFO",
        },
        "defaults": params,
        "strategies": [
            {
                "id": strategy_id_val,
                "name": strategy_name_val,
                "enabled": True,
                "formula_path": _to_rel(formula_path),
                "params": params,
            }
        ],
    }
    generated_strategy_config_path = os.path.join(bundle_dir, "generated_strategy_config.json")
    _write_json(generated_strategy_config_path, generated_strategy_config)

    update_training_manifest(
        {"manifest_path": _resolve_rel(manifest_path), "run_id": run_id},
        stage="bundle_created",
        artifacts={
            "bundle_dir": bundle_dir,
            "strategy_bundle_path": bundle_path,
            "generated_strategy_config_path": generated_strategy_config_path,
            "formula_bundle_path": formula_path,
            "selection_output_path": selection_output_abs,
        },
        summary={
            **(manifest.get("summary") or {}),
            "bundle_strategy_id": strategy_id_val,
            "bundle_source": actual_source,
            "bundle_candidate_rank": int(candidate_rank),
        },
    )

    return {
        "bundle_path": bundle_path,
        "formula_path": formula_path,
        "generated_strategy_config_path": generated_strategy_config_path,
        "strategy_id": strategy_id_val,
        "source": actual_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strategy bundle from training manifest and candidate outputs.")
    parser.add_argument("--manifest", type=str, required=True, help="path to run manifest.json")
    parser.add_argument(
        "--selection-output",
        type=str,
        default=None,
        help="optional selection output path; default auto-detect under run_dir/selection",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["selected", "best"],
        default=None,
        help="candidate source; default selected if selection output exists else best",
    )
    parser.add_argument("--candidate-rank", type=int, default=1, help="1-based candidate rank")
    parser.add_argument("--strategy-id", type=str, default=None)
    parser.add_argument("--strategy-name", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--initial-capital", type=float, default=None)
    parser.add_argument("--state-backend", type=str, default="sql", choices=["sql", "json"])
    parser.add_argument("--replay-source", type=str, default="sql_eod", choices=["sql_eod", "parquet"])
    parser.add_argument(
        "--replay-strict",
        dest="replay_strict",
        action="store_true",
        default=True,
        help="enable replay_strict in generated params (default: enabled)",
    )
    parser.add_argument(
        "--no-replay-strict",
        dest="replay_strict",
        action="store_false",
        help="disable replay_strict in generated params",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="optional bundle output dir")
    args = parser.parse_args()

    result = build_bundle(
        manifest_path=args.manifest,
        selection_output_path=args.selection_output,
        source=args.source,
        candidate_rank=args.candidate_rank,
        strategy_id=args.strategy_id,
        strategy_name=args.strategy_name,
        top_k=args.top_k,
        fee_rate=args.fee_rate,
        take_profit=args.take_profit,
        initial_capital=args.initial_capital,
        state_backend=args.state_backend,
        replay_source=args.replay_source,
        replay_strict=args.replay_strict,
        output_dir=args.output_dir,
    )
    print(f"Saved bundle to: {result['bundle_path']}")
    print(f"Saved formula to: {result['formula_path']}")
    print(f"Saved generated strategy config to: {result['generated_strategy_config_path']}")
    print(f"Strategy ID: {result['strategy_id']} | source={result['source']}")


if __name__ == "__main__":
    main()
