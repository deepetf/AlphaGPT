"""
run_sim 状态准备逻辑测试。
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_manager import run_sim


def test_prepare_single_day_replay_sql_backend():
    """单日 strict_replay 在 SQL 后端下只清理当日并按 as-of 回灌。"""
    runner = MagicMock()
    runner.strategy_id = "king_v1"
    runner.sql_state_store = MagicMock()
    runner._hydrate_state_from_sql = MagicMock()

    run_sim._prepare_runner_state_for_single_day_replay(runner, "2025-02-17")

    runner.sql_state_store.reset_strategy_date.assert_called_once_with("king_v1", "2025-02-17")
    runner._hydrate_state_from_sql.assert_called_once_with(as_of_date="2025-02-17")


def test_run_once_strict_replay_uses_single_day_prepare(monkeypatch):
    """run_once(strict_replay + date) 应走单日准备流程，不走全量 reset。"""

    class FakeProvider:
        def close(self):
            return None

    fake_runner = MagicMock()
    fake_runner.strategy_id = "king_v1"

    class FakeMultiRunner:
        def __init__(self):
            self.runners = {"king_v1": fake_runner}

        def run_all_strategies(self, date, mode="auto"):
            return {"king_v1": {"status": "success", "date": date, "mode": mode}}

        def print_summary(self):
            return None

    prepare_mock = MagicMock()
    reset_mock = MagicMock()

    monkeypatch.setattr(run_sim, "RealtimeDataProvider", lambda: FakeProvider())
    monkeypatch.setattr(run_sim, "_get_warmup_start_date", lambda *_args, **_kwargs: "2024-11-01")
    monkeypatch.setattr(run_sim, "_build_runner", lambda **_kwargs: FakeMultiRunner())
    monkeypatch.setattr(run_sim, "_prepare_runner_state_for_single_day_replay", prepare_mock)
    monkeypatch.setattr(run_sim, "_reset_runner_state", reset_mock)

    result = run_sim.run_once(
        mode="strict_replay",
        date="2025-02-17",
        strategy_id="king_v1",
        state_backend="sql",
    )

    assert result["status"] == "success"
    prepare_mock.assert_called_once_with(fake_runner, "2025-02-17")
    reset_mock.assert_not_called()
