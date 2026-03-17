import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from workflow.run_manifest import prepare_training_run, validate_run_data_snapshot


def _write_parquet(path: Path, rows):
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


def _write_yaml(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def test_prepare_training_run_writes_data_snapshot(tmp_path):
    parquet_path = tmp_path / "cb_data_snapshot.pq"
    _write_parquet(
        parquet_path,
        [
            {"trade_date": "2024-01-02", "code": "110001.SH", "close": 101.0},
            {"trade_date": "2024-01-03", "code": "110002.SH", "close": 102.0},
        ],
    )
    config = {
        "cb_parquet_path": str(parquet_path),
        "input_features": ["CLOSE", "DBLOW"],
        "feature_normalization_overrides": {"DBLOW": False},
        "robust_config": {"train_test_split_date": "2024-08-01"},
    }
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config)

    run_context = prepare_training_run(
        config=config,
        config_path=str(config_path),
        data_start_date="2022-08-01",
        run_id="snapshot_case",
        artifacts_root=str(tmp_path / "artifacts"),
    )

    with open(run_context["manifest_path"], "r", encoding="utf-8") as f:
        manifest = json.load(f)

    snapshot = manifest["data_snapshot"]
    assert snapshot["parquet_path"] == str(parquet_path)
    assert snapshot["row_count"] == 2
    assert snapshot["asset_count"] == 2
    assert snapshot["max_date"] == "2024-01-03"
    assert snapshot["train_date_range"]["data_start_date"] == "2022-08-01"
    assert snapshot["train_date_range"]["train_test_split_date"] == "2024-08-01"
    assert snapshot["file_hash"]
    assert snapshot["schema_hash"]
    assert snapshot["feature_config_hash"]


def test_validate_run_data_snapshot_detects_file_change(tmp_path):
    parquet_path = tmp_path / "cb_data_snapshot.pq"
    _write_parquet(
        parquet_path,
        [
            {"trade_date": "2024-01-02", "code": "110001.SH", "close": 101.0},
            {"trade_date": "2024-01-03", "code": "110002.SH", "close": 102.0},
        ],
    )
    config = {
        "cb_parquet_path": str(parquet_path),
        "input_features": ["CLOSE"],
        "robust_config": {"train_test_split_date": "2024-08-01"},
    }
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config)
    run_context = prepare_training_run(
        config=config,
        config_path=str(config_path),
        data_start_date="2022-08-01",
        run_id="snapshot_changed",
        artifacts_root=str(tmp_path / "artifacts"),
    )

    validate_run_data_snapshot(run_context["manifest_path"])

    _write_parquet(
        parquet_path,
        [
            {"trade_date": "2024-01-02", "code": "110001.SH", "close": 101.0},
            {"trade_date": "2024-01-03", "code": "110002.SH", "close": 102.0},
            {"trade_date": "2024-01-04", "code": "110003.SH", "close": 103.0},
        ],
    )

    with pytest.raises(ValueError, match="数据快照校验失败"):
        validate_run_data_snapshot(run_context["manifest_path"])


def test_validate_run_data_snapshot_rejects_mismatched_config_path(tmp_path):
    parquet_path = tmp_path / "cb_data_snapshot.pq"
    other_parquet_path = tmp_path / "cb_data_other.pq"
    _write_parquet(
        parquet_path,
        [{"trade_date": "2024-01-02", "code": "110001.SH", "close": 101.0}],
    )
    _write_parquet(
        other_parquet_path,
        [{"trade_date": "2024-01-02", "code": "110999.SH", "close": 88.0}],
    )
    base_config = {"cb_parquet_path": str(parquet_path), "input_features": ["CLOSE"]}
    other_config = {"cb_parquet_path": str(other_parquet_path), "input_features": ["CLOSE"]}
    base_config_path = tmp_path / "base.yaml"
    other_config_path = tmp_path / "other.yaml"
    _write_yaml(base_config_path, base_config)
    _write_yaml(other_config_path, other_config)

    run_context = prepare_training_run(
        config=base_config,
        config_path=str(base_config_path),
        data_start_date="2022-08-01",
        run_id="snapshot_config_mismatch",
        artifacts_root=str(tmp_path / "artifacts"),
    )

    with pytest.raises(ValueError, match="不同数据文件"):
        validate_run_data_snapshot(run_context["manifest_path"], config_path=str(other_config_path))
