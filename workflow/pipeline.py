import argparse
import os
import subprocess
import sys
from typing import List, Optional, Tuple

from .bundle_builder import load_manifest
from .run_manifest import generate_run_id
from .pipeline_state import (
    STAGE_ORDER,
    init_pipeline_status,
    load_pipeline_status,
    mark_pipeline_finished,
    should_skip_stage,
    update_stage_status,
)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_manifest_path(run_id: Optional[str], manifest_path: Optional[str], artifacts_root: Optional[str]) -> str:
    if manifest_path:
        if os.path.isabs(manifest_path):
            return manifest_path
        return os.path.abspath(os.path.join(_project_root(), manifest_path))
    if not run_id:
        raise ValueError("必须提供 --run-id 或 --manifest")
    root = artifacts_root or os.path.join("artifacts", "runs")
    if not os.path.isabs(root):
        root = os.path.abspath(os.path.join(_project_root(), root))
    return os.path.join(root, run_id, "manifest.json")


def _resolve_run_dir(run_id: Optional[str], manifest_path: Optional[str], artifacts_root: Optional[str]) -> Tuple[str, str]:
    resolved_manifest = _resolve_manifest_path(run_id, manifest_path, artifacts_root)
    _, run_dir = load_manifest(resolved_manifest)
    return resolved_manifest, run_dir


def _run_command(cmd: List[str]) -> None:
    printable = _command_str(cmd)
    print(f"[pipeline] running: {printable}")
    subprocess.run(cmd, cwd=_project_root(), check=True)


def _command_str(cmd: List[str]) -> str:
    return " ".join([f'"{x}"' if " " in x else x for x in cmd])


def _selection_paths(run_dir: str) -> Tuple[str, str]:
    selection_dir = os.path.join(run_dir, "selection")
    os.makedirs(selection_dir, exist_ok=True)
    return (
        os.path.join(selection_dir, "top3_factors.json"),
        os.path.join(selection_dir, "top3_factors_report.md"),
    )


def _bundle_path(run_dir: str) -> str:
    return os.path.join(run_dir, "bundle", "strategy_bundle.json")


def _requested_e2e_stages(args) -> List[str]:
    stages = ["train", "select", "bundle"]
    if args.verify_start and args.verify_end:
        stages.append("verify")
    if args.sim_date or (args.sim_start_date and args.sim_end_date):
        stages.append("sim")
    return stages


def _build_train_cmd(args) -> Tuple[List[str], str]:
    run_id = args.run_id or generate_run_id(config_path=args.config, explicit_run_id=None)
    cmd = [sys.executable, "-m", "model_core.engine", "--run-id", run_id]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.data_start_date:
        cmd.extend(["--data-start-date", args.data_start_date])
    if args.artifacts_root:
        cmd.extend(["--artifacts-root", args.artifacts_root])
    return cmd, run_id


def _build_select_cmd(args, manifest_path: str, run_dir: str) -> Tuple[List[str], str]:
    manifest, _ = load_manifest(manifest_path)
    train_dir = os.path.join(run_dir, "train")
    input_path = args.input or os.path.join(train_dir, "best_cb_formula.json")
    output_path, report_output = _selection_paths(run_dir)
    if args.output:
        output_path = args.output
    if args.report_output:
        report_output = args.report_output
    config_path = args.config or manifest.get("training", {}).get("resolved_config_snapshot_path")
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(_project_root(), config_path))

    cmd = [
        sys.executable,
        "-m",
        "model_core.select_top_factors",
        "--input",
        input_path,
        "--output",
        output_path,
        "--report-output",
        report_output,
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if args.selection_config:
        cmd.extend(["--selection-config", args.selection_config])
    if args.data_start_date:
        cmd.extend(["--data-start-date", args.data_start_date])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.enable_ai_review:
        cmd.append("--enable-ai-review")
    if args.ai_provider:
        cmd.extend(["--ai-provider", args.ai_provider])
    if args.ai_model:
        cmd.extend(["--ai-model", args.ai_model])
    if args.ai_max_candidates is not None:
        cmd.extend(["--ai-max-candidates", str(args.ai_max_candidates)])
    if args.ai_timeout_sec is not None:
        cmd.extend(["--ai-timeout-sec", str(args.ai_timeout_sec)])
    if args.ai_review_output:
        cmd.extend(["--ai-review-output", args.ai_review_output])
    return cmd, output_path


def _build_bundle_cmd(args, manifest_path: str, run_dir: str) -> Tuple[List[str], str]:
    selection_output = args.selection_output or os.path.join(run_dir, "selection", "top3_factors.json")
    cmd = [sys.executable, "-m", "workflow.bundle_builder", "--manifest", manifest_path]
    if args.source:
        cmd.extend(["--source", args.source])
    elif os.path.exists(selection_output):
        cmd.extend(["--source", "selected"])
    else:
        cmd.extend(["--source", "best"])
    if os.path.exists(selection_output):
        cmd.extend(["--selection-output", selection_output])
    if args.candidate_rank is not None:
        cmd.extend(["--candidate-rank", str(args.candidate_rank)])
    if args.strategy_id:
        cmd.extend(["--strategy-id", args.strategy_id])
    if args.strategy_name:
        cmd.extend(["--strategy-name", args.strategy_name])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.fee_rate is not None:
        cmd.extend(["--fee-rate", str(args.fee_rate)])
    if args.take_profit is not None:
        cmd.extend(["--take-profit", str(args.take_profit)])
    if args.initial_capital is not None:
        cmd.extend(["--initial-capital", str(args.initial_capital)])
    if args.state_backend:
        cmd.extend(["--state-backend", args.state_backend])
    if args.replay_source:
        cmd.extend(["--replay-source", args.replay_source])
    if args.no_replay_strict:
        cmd.append("--no-replay-strict")
    else:
        cmd.append("--replay-strict")
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    return cmd, _bundle_path(run_dir)


def _build_verify_cmd(args, run_dir: str) -> Tuple[List[str], str]:
    bundle_path = args.bundle or _bundle_path(run_dir)
    cmd = [
        sys.executable,
        "tests\\verify_strategy.py",
        "--bundle",
        bundle_path,
        "--start",
        args.start,
        "--end",
        args.end,
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.strategy_id:
        cmd.extend(["--strategy-id", args.strategy_id])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.fee_rate is not None:
        cmd.extend(["--fee-rate", str(args.fee_rate)])
    if args.initial_cash is not None:
        cmd.extend(["--initial-cash", str(args.initial_cash)])
    if args.take_profit is not None:
        cmd.extend(["--take-profit", str(args.take_profit)])
    if args.verify_no_cash_aware:
        cmd.append("--verify-no-cash-aware")
    return cmd, bundle_path


def _build_sim_cmd(args, run_dir: str) -> Tuple[List[str], str]:
    bundle_path = args.bundle or _bundle_path(run_dir)
    cmd = [
        sys.executable,
        "strategy_manager\\run_sim.py",
        "--bundle",
        bundle_path,
        "--mode",
        args.mode,
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.strategy_id:
        cmd.extend(["--strategy-id", args.strategy_id])
    if args.date:
        cmd.extend(["--date", args.date])
    if args.start_date:
        cmd.extend(["--start-date", args.start_date])
    if args.end_date:
        cmd.extend(["--end-date", args.end_date])
    if args.state_backend:
        cmd.extend(["--state-backend", args.state_backend])
    if args.live_quote_source:
        cmd.extend(["--live-quote-source", args.live_quote_source])
    if args.replay_source:
        cmd.extend(["--replay-source", args.replay_source])
    if args.schedule:
        cmd.append("--schedule")
    if args.hour is not None:
        cmd.extend(["--hour", str(args.hour)])
    if args.minute is not None:
        cmd.extend(["--minute", str(args.minute)])
    return cmd, bundle_path


def cmd_train(args) -> None:
    cmd, run_id = _build_train_cmd(args)
    _run_command(cmd)
    manifest_path = _resolve_manifest_path(run_id, None, args.artifacts_root)
    print(f"[pipeline] train completed: run_id={run_id}")
    print(f"[pipeline] manifest={manifest_path}")


def cmd_select(args) -> None:
    manifest_path, run_dir = _resolve_run_dir(args.run_id, args.manifest, args.artifacts_root)
    cmd, output_path = _build_select_cmd(args, manifest_path, run_dir)
    _run_command(cmd)
    print(f"[pipeline] selection output={output_path}")


def cmd_bundle(args) -> None:
    manifest_path, run_dir = _resolve_run_dir(args.run_id, args.manifest, args.artifacts_root)
    cmd, bundle_path = _build_bundle_cmd(args, manifest_path, run_dir)
    _run_command(cmd)
    print(f"[pipeline] bundle path={bundle_path}")


def cmd_verify(args) -> None:
    manifest_path, run_dir = _resolve_run_dir(args.run_id, args.manifest, args.artifacts_root)
    cmd, bundle_path = _build_verify_cmd(args, run_dir)
    _run_command(cmd)
    print(f"[pipeline] verify completed: bundle={bundle_path}")


def cmd_sim(args) -> None:
    manifest_path, run_dir = _resolve_run_dir(args.run_id, args.manifest, args.artifacts_root)
    cmd, bundle_path = _build_sim_cmd(args, run_dir)
    _run_command(cmd)
    print(f"[pipeline] sim completed: bundle={bundle_path}")


def cmd_e2e(args) -> None:
    run_id = args.run_id or generate_run_id(config_path=args.config, explicit_run_id=None)
    manifest_path = _resolve_manifest_path(run_id, None, args.artifacts_root)
    run_dir = os.path.dirname(manifest_path)
    requested_stages = _requested_e2e_stages(args)
    init_pipeline_status(
        run_dir=run_dir,
        run_id=run_id,
        command=_command_str(sys.argv),
        requested_stages=requested_stages,
        resume=args.resume,
    )

    try:
        stages_to_run = [stage for stage in requested_stages if STAGE_ORDER.index(stage) >= STAGE_ORDER.index(args.from_stage)]
        train_args = argparse.Namespace(
            config=args.config,
            data_start_date=args.data_start_date,
            run_id=run_id,
            artifacts_root=args.artifacts_root,
        )
        if "train" in stages_to_run:
            if should_skip_stage(run_dir, "train", args.resume):
                print("[pipeline] skip completed stage: train")
            else:
                train_cmd, _ = _build_train_cmd(train_args)
                update_stage_status(run_dir=run_dir, stage="train", status="running", command=_command_str(train_cmd))
                _run_command(train_cmd)
                update_stage_status(
                    run_dir=run_dir,
                    stage="train",
                    status="completed",
                    outputs={
                        "manifest_path": manifest_path,
                        "train_dir": os.path.join(run_dir, "train"),
                    },
                )

        select_args = argparse.Namespace(
            run_id=run_id,
            manifest=None,
            artifacts_root=args.artifacts_root,
            input=None,
            output=None,
            report_output=None,
            config=None,
            selection_config=args.selection_config,
            data_start_date=args.data_start_date,
            top_k=args.selection_top_k,
            enable_ai_review=args.enable_ai_review,
            ai_provider=args.ai_provider,
            ai_model=args.ai_model,
            ai_max_candidates=args.ai_max_candidates,
            ai_timeout_sec=args.ai_timeout_sec,
            ai_review_output=args.ai_review_output,
        )
        if "select" in stages_to_run:
            if should_skip_stage(run_dir, "select", args.resume):
                print("[pipeline] skip completed stage: select")
            else:
                select_cmd, output_path = _build_select_cmd(select_args, manifest_path, run_dir)
                update_stage_status(run_dir=run_dir, stage="select", status="running", command=_command_str(select_cmd))
                _run_command(select_cmd)
                update_stage_status(
                    run_dir=run_dir,
                    stage="select",
                    status="completed",
                    outputs={
                        "selection_output": output_path,
                        "selection_report": os.path.join(run_dir, "selection", "top3_factors_report.md"),
                    },
                )

        bundle_args = argparse.Namespace(
            run_id=run_id,
            manifest=None,
            artifacts_root=args.artifacts_root,
            selection_output=None,
            source=args.bundle_source,
            candidate_rank=args.candidate_rank,
            strategy_id=args.strategy_id,
            strategy_name=args.strategy_name,
            top_k=args.top_k,
            fee_rate=args.fee_rate,
            take_profit=args.take_profit,
            initial_capital=args.initial_capital,
            state_backend=args.state_backend,
            replay_source=args.replay_source,
            no_replay_strict=not args.replay_strict,
            output_dir=None,
        )
        if "bundle" in stages_to_run:
            if should_skip_stage(run_dir, "bundle", args.resume):
                print("[pipeline] skip completed stage: bundle")
            else:
                bundle_cmd, bundle_path = _build_bundle_cmd(bundle_args, manifest_path, run_dir)
                update_stage_status(run_dir=run_dir, stage="bundle", status="running", command=_command_str(bundle_cmd))
                _run_command(bundle_cmd)
                update_stage_status(
                    run_dir=run_dir,
                    stage="bundle",
                    status="completed",
                    outputs={
                        "bundle_path": bundle_path,
                        "bundle_dir": os.path.join(run_dir, "bundle"),
                    },
                )

        if args.verify_start and args.verify_end and "verify" in stages_to_run:
            verify_args = argparse.Namespace(
                run_id=run_id,
                manifest=None,
                artifacts_root=args.artifacts_root,
                bundle=None,
                start=args.verify_start,
                end=args.verify_end,
                config=None,
                strategy_id=args.strategy_id,
                top_k=args.top_k,
                fee_rate=args.fee_rate,
                initial_cash=args.initial_capital,
                take_profit=args.take_profit,
                verify_no_cash_aware=args.verify_no_cash_aware,
            )
            if should_skip_stage(run_dir, "verify", args.resume):
                print("[pipeline] skip completed stage: verify")
            else:
                verify_cmd, verify_bundle_path = _build_verify_cmd(verify_args, run_dir)
                update_stage_status(run_dir=run_dir, stage="verify", status="running", command=_command_str(verify_cmd))
                _run_command(verify_cmd)
                update_stage_status(
                    run_dir=run_dir,
                    stage="verify",
                    status="completed",
                    outputs={"bundle_path": verify_bundle_path, "artifacts_dir": os.path.join(_project_root(), "tests", "artifacts")},
                )

        if (args.sim_date or (args.sim_start_date and args.sim_end_date)) and "sim" in stages_to_run:
            sim_args = argparse.Namespace(
                run_id=run_id,
                manifest=None,
                artifacts_root=args.artifacts_root,
                bundle=None,
                mode=args.sim_mode,
                config=None,
                strategy_id=args.strategy_id,
                date=args.sim_date,
                start_date=args.sim_start_date,
                end_date=args.sim_end_date,
                state_backend=args.state_backend,
                live_quote_source=args.live_quote_source,
                replay_source=args.replay_source,
                schedule=False,
                hour=args.hour,
                minute=args.minute,
            )
            if should_skip_stage(run_dir, "sim", args.resume):
                print("[pipeline] skip completed stage: sim")
            else:
                sim_cmd, sim_bundle_path = _build_sim_cmd(sim_args, run_dir)
                update_stage_status(run_dir=run_dir, stage="sim", status="running", command=_command_str(sim_cmd))
                _run_command(sim_cmd)
                update_stage_status(
                    run_dir=run_dir,
                    stage="sim",
                    status="completed",
                    outputs={"bundle_path": sim_bundle_path, "mode": args.sim_mode},
                )

        mark_pipeline_finished(run_dir, status="completed")
        print(f"[pipeline] e2e completed: run_id={run_id}")
        print(f"[pipeline] status={os.path.join(run_dir, 'pipeline_status.json')}")
    except subprocess.CalledProcessError as exc:
        failed_stage = None
        status_obj = load_pipeline_status(run_dir) or {}
        for stage in requested_stages:
            stage_info = (status_obj.get("stages") or {}).get(stage) or {}
            if stage_info.get("status") == "running":
                failed_stage = stage
                break
        if failed_stage:
            update_stage_status(run_dir=run_dir, stage=failed_stage, status="failed", error=str(exc))
        mark_pipeline_finished(run_dir, status="failed", error=str(exc))
        print(f"[pipeline] e2e failed: run_id={run_id}")
        print(f"[pipeline] status={os.path.join(run_dir, 'pipeline_status.json')}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Straight-through pipeline for train/select/bundle/verify/sim.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="run model training and create run manifest")
    p_train.add_argument("--config", type=str, default=None)
    p_train.add_argument("--data-start-date", type=str, default=None)
    p_train.add_argument("--run-id", type=str, default=None)
    p_train.add_argument("--artifacts-root", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_select = subparsers.add_parser("select", help="run candidate post-selection into run_dir/selection")
    p_select.add_argument("--run-id", type=str, default=None)
    p_select.add_argument("--manifest", type=str, default=None)
    p_select.add_argument("--artifacts-root", type=str, default=None)
    p_select.add_argument("--input", type=str, default=None)
    p_select.add_argument("--output", type=str, default=None)
    p_select.add_argument("--report-output", type=str, default=None)
    p_select.add_argument("--config", type=str, default=None)
    p_select.add_argument("--selection-config", type=str, default=None)
    p_select.add_argument("--data-start-date", type=str, default=None)
    p_select.add_argument("--top-k", type=int, default=None)
    p_select.add_argument("--enable-ai-review", action="store_true")
    p_select.add_argument("--ai-provider", type=str, default=None)
    p_select.add_argument("--ai-model", type=str, default=None)
    p_select.add_argument("--ai-max-candidates", type=int, default=None)
    p_select.add_argument("--ai-timeout-sec", type=float, default=None)
    p_select.add_argument("--ai-review-output", type=str, default=None)
    p_select.set_defaults(func=cmd_select)

    p_bundle = subparsers.add_parser("bundle", help="build strategy bundle from run outputs")
    p_bundle.add_argument("--run-id", type=str, default=None)
    p_bundle.add_argument("--manifest", type=str, default=None)
    p_bundle.add_argument("--artifacts-root", type=str, default=None)
    p_bundle.add_argument("--selection-output", type=str, default=None)
    p_bundle.add_argument("--source", type=str, choices=["selected", "best"], default=None)
    p_bundle.add_argument("--candidate-rank", type=int, default=1)
    p_bundle.add_argument("--strategy-id", type=str, default=None)
    p_bundle.add_argument("--strategy-name", type=str, default=None)
    p_bundle.add_argument("--top-k", type=int, default=None)
    p_bundle.add_argument("--fee-rate", type=float, default=None)
    p_bundle.add_argument("--take-profit", type=float, default=None)
    p_bundle.add_argument("--initial-capital", type=float, default=None)
    p_bundle.add_argument("--state-backend", type=str, default="sql", choices=["sql", "json"])
    p_bundle.add_argument("--replay-source", type=str, default="sql_eod", choices=["sql_eod", "parquet"])
    p_bundle.add_argument("--replay-strict", action="store_true", default=True)
    p_bundle.add_argument("--no-replay-strict", action="store_true")
    p_bundle.add_argument("--output-dir", type=str, default=None)
    p_bundle.set_defaults(func=cmd_bundle)

    p_verify = subparsers.add_parser("verify", help="run verify directly from bundle")
    p_verify.add_argument("--run-id", type=str, default=None)
    p_verify.add_argument("--manifest", type=str, default=None)
    p_verify.add_argument("--artifacts-root", type=str, default=None)
    p_verify.add_argument("--bundle", type=str, default=None)
    p_verify.add_argument("--start", type=str, required=True)
    p_verify.add_argument("--end", type=str, required=True)
    p_verify.add_argument("--config", type=str, default=None)
    p_verify.add_argument("--strategy-id", type=str, default=None)
    p_verify.add_argument("--top-k", type=int, default=None)
    p_verify.add_argument("--fee-rate", type=float, default=None)
    p_verify.add_argument("--initial-cash", type=float, default=None)
    p_verify.add_argument("--take-profit", type=float, default=None)
    p_verify.add_argument("--verify-no-cash-aware", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_sim = subparsers.add_parser("sim", help="run sim directly from bundle")
    p_sim.add_argument("--run-id", type=str, default=None)
    p_sim.add_argument("--manifest", type=str, default=None)
    p_sim.add_argument("--artifacts-root", type=str, default=None)
    p_sim.add_argument("--bundle", type=str, default=None)
    p_sim.add_argument("--mode", type=str, default="strict_replay", choices=["live", "strict_replay"])
    p_sim.add_argument("--config", type=str, default=None)
    p_sim.add_argument("--strategy-id", type=str, default=None)
    p_sim.add_argument("--date", type=str, default=None)
    p_sim.add_argument("--start-date", type=str, default=None)
    p_sim.add_argument("--end-date", type=str, default=None)
    p_sim.add_argument("--state-backend", type=str, default=None, choices=["sql", "json"])
    p_sim.add_argument("--live-quote-source", type=str, default="dummy", choices=["dummy", "qmt"])
    p_sim.add_argument("--replay-source", type=str, default=None, choices=["sql_eod", "parquet"])
    p_sim.add_argument("--schedule", action="store_true")
    p_sim.add_argument("--hour", type=int, default=14)
    p_sim.add_argument("--minute", type=int, default=50)
    p_sim.set_defaults(func=cmd_sim)

    p_e2e = subparsers.add_parser("e2e", help="run train -> select -> bundle -> verify/sim")
    p_e2e.add_argument("--config", type=str, default=None)
    p_e2e.add_argument("--data-start-date", type=str, default=None)
    p_e2e.add_argument("--run-id", type=str, default=None)
    p_e2e.add_argument("--artifacts-root", type=str, default=None)
    p_e2e.add_argument("--selection-config", type=str, default=None)
    p_e2e.add_argument("--selection-top-k", type=int, default=None)
    p_e2e.add_argument("--enable-ai-review", action="store_true")
    p_e2e.add_argument("--ai-provider", type=str, default=None)
    p_e2e.add_argument("--ai-model", type=str, default=None)
    p_e2e.add_argument("--ai-max-candidates", type=int, default=None)
    p_e2e.add_argument("--ai-timeout-sec", type=float, default=None)
    p_e2e.add_argument("--ai-review-output", type=str, default=None)
    p_e2e.add_argument("--bundle-source", type=str, choices=["selected", "best"], default=None)
    p_e2e.add_argument("--candidate-rank", type=int, default=1)
    p_e2e.add_argument("--strategy-id", type=str, default=None)
    p_e2e.add_argument("--strategy-name", type=str, default=None)
    p_e2e.add_argument("--top-k", type=int, default=None)
    p_e2e.add_argument("--fee-rate", type=float, default=None)
    p_e2e.add_argument("--take-profit", type=float, default=None)
    p_e2e.add_argument("--initial-capital", type=float, default=None)
    p_e2e.add_argument("--state-backend", type=str, default="sql", choices=["sql", "json"])
    p_e2e.add_argument("--replay-source", type=str, default="sql_eod", choices=["sql_eod", "parquet"])
    p_e2e.add_argument("--replay-strict", action="store_true", default=True)
    p_e2e.add_argument("--verify-start", type=str, default=None)
    p_e2e.add_argument("--verify-end", type=str, default=None)
    p_e2e.add_argument("--verify-no-cash-aware", action="store_true")
    p_e2e.add_argument("--sim-mode", type=str, default="strict_replay", choices=["live", "strict_replay"])
    p_e2e.add_argument("--sim-date", type=str, default=None)
    p_e2e.add_argument("--sim-start-date", type=str, default=None)
    p_e2e.add_argument("--sim-end-date", type=str, default=None)
    p_e2e.add_argument("--live-quote-source", type=str, default="dummy", choices=["dummy", "qmt"])
    p_e2e.add_argument("--hour", type=int, default=14)
    p_e2e.add_argument("--minute", type=int, default=50)
    p_e2e.add_argument("--resume", action="store_true", help="resume and skip completed stages based on pipeline_status.json")
    p_e2e.add_argument(
        "--from-stage",
        type=str,
        default="train",
        choices=STAGE_ORDER,
        help="start e2e from a specific stage",
    )
    p_e2e.set_defaults(func=cmd_e2e)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
