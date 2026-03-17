import argparse

import workflow.pipeline as pipeline_module


def test_cmd_select_validates_snapshot(monkeypatch):
    calls = []

    monkeypatch.setattr(
        pipeline_module,
        "_resolve_run_dir",
        lambda run_id, manifest, artifacts_root: ("manifest.json", "run_dir"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_build_select_cmd",
        lambda args, manifest_path, run_dir: (["python", "-m", "dummy"], "selection.json"),
    )
    monkeypatch.setattr(pipeline_module, "_run_command", lambda cmd: calls.append(("run", cmd)))
    monkeypatch.setattr(
        pipeline_module,
        "_validate_stage_data_snapshot",
        lambda manifest_path, config_path=None: calls.append(("validate", manifest_path, config_path)),
    )

    args = argparse.Namespace(
        run_id="run_x",
        manifest=None,
        artifacts_root=None,
        config="model_core/default_config.yaml",
    )
    pipeline_module.cmd_select(args)

    assert calls[0] == ("validate", "manifest.json", "model_core/default_config.yaml")


def test_cmd_verify_validates_snapshot(monkeypatch):
    calls = []

    monkeypatch.setattr(
        pipeline_module,
        "_resolve_run_dir",
        lambda run_id, manifest, artifacts_root: ("manifest.json", "run_dir"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_build_verify_cmd",
        lambda args, run_dir: (["python", "tests\\verify_strategy.py"], "bundle.json"),
    )
    monkeypatch.setattr(pipeline_module, "_run_command", lambda cmd: calls.append(("run", cmd)))
    monkeypatch.setattr(
        pipeline_module,
        "_validate_stage_data_snapshot",
        lambda manifest_path, config_path=None: calls.append(("validate", manifest_path, config_path)),
    )

    args = argparse.Namespace(
        run_id="run_x",
        manifest=None,
        artifacts_root=None,
        config=None,
    )
    pipeline_module.cmd_verify(args)

    assert calls[0] == ("validate", "manifest.json", None)


def test_cmd_sim_validates_snapshot(monkeypatch):
    calls = []

    monkeypatch.setattr(
        pipeline_module,
        "_resolve_run_dir",
        lambda run_id, manifest, artifacts_root: ("manifest.json", "run_dir"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_build_sim_cmd",
        lambda args, run_dir: (["python", "strategy_manager\\run_sim.py"], "bundle.json"),
    )
    monkeypatch.setattr(pipeline_module, "_run_command", lambda cmd: calls.append(("run", cmd)))
    monkeypatch.setattr(
        pipeline_module,
        "_validate_stage_data_snapshot",
        lambda manifest_path, config_path=None: calls.append(("validate", manifest_path, config_path)),
    )

    args = argparse.Namespace(
        run_id="run_x",
        manifest=None,
        artifacts_root=None,
        config="bundle_config.yaml",
    )
    pipeline_module.cmd_sim(args)

    assert calls[0] == ("validate", "manifest.json", "bundle_config.yaml")
