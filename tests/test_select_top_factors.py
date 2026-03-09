import json
from pathlib import Path

from model_core.select_top_factors import (
    apply_hard_filters,
    compute_selection_score,
    dedupe_candidates,
    load_candidates,
    select_diverse_top_k,
)


def test_load_candidates_reads_best_history_and_diverse(tmp_path):
    path = tmp_path / "best_cb_formula.json"
    payload = {
        "best": {
            "formula": ["PURE_VALUE", "REMAIN_SIZE", "MIN"],
            "readable": "PURE_VALUE REMAIN_SIZE MIN",
            "score": 1.0,
        },
        "history": [
            {
                "formula": ["A", "B", "MIN"],
                "readable": "A B MIN",
                "score": 2.0,
            }
        ],
        "diverse_top_50": [
            {
                "formula": ["X", "Y", "MAX"],
                "readable": "X Y MAX",
                "score": 3.0,
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    candidates = load_candidates(str(path), ["best", "history", "diverse_top_50"])

    assert len(candidates) == 3
    assert [c["source"] for c in candidates] == ["best", "history", "diverse_top_50"]


def test_dedupe_candidates_merges_canonical_equivalents():
    candidates = [
        {
            "source": "history",
            "formula": ["PURE_VALUE", "REMAIN_SIZE", "MIN"],
            "readable": "PURE_VALUE REMAIN_SIZE MIN",
            "original_score": 8.0,
            "original_sharpe_train": 2.0,
            "original_sharpe_val": 2.2,
        },
        {
            "source": "diverse_top_50",
            "formula": ["REMAIN_SIZE", "PURE_VALUE", "MIN"],
            "readable": "REMAIN_SIZE PURE_VALUE MIN",
            "original_score": 7.0,
        },
    ]

    deduped = dedupe_candidates(candidates)

    assert len(deduped) == 1
    assert deduped[0]["formula"] == ["PURE_VALUE", "REMAIN_SIZE", "MIN"]
    assert deduped[0]["source"] == "history"


def test_filters_and_diverse_selection_keep_balanced_candidates():
    selection_cfg = {
        "min_sharpe_all": 2.4,
        "min_sharpe_train": 2.1,
        "min_sharpe_val": 2.5,
        "min_balanced_sharpe": 2.1,
        "min_stability": -0.4,
        "max_drawdown": 0.20,
        "max_train_val_gap": 1.0,
    }
    weights = {
        "sharpe_all": 0.30,
        "balanced_sharpe": 0.25,
        "stability": 0.20,
        "annualized_ret": 0.15,
        "max_drawdown": -0.10,
        "train_val_gap": -0.20,
    }

    good_a = {
        "formula": ["PURE_VALUE", "REMAIN_SIZE", "MIN"],
        "readable": "PURE_VALUE REMAIN_SIZE MIN",
        "sharpe_all": 3.0,
        "sharpe_train": 2.5,
        "sharpe_val": 3.0,
        "balanced_sharpe": 2.5,
        "stability": 0.1,
        "annualized_ret": 1.0,
        "max_drawdown": 0.15,
        "train_val_gap": 0.5,
    }
    good_b = {
        "formula": ["PREM_Z", "VOL", "ADD"],
        "readable": "PREM_Z VOL ADD",
        "sharpe_all": 2.9,
        "sharpe_train": 2.4,
        "sharpe_val": 2.8,
        "balanced_sharpe": 2.4,
        "stability": -0.1,
        "annualized_ret": 0.9,
        "max_drawdown": 0.14,
        "train_val_gap": 0.4,
    }
    bad_gap = {
        "formula": ["PURE_VALUE", "PREM_Z", "MAX"],
        "readable": "PURE_VALUE PREM_Z MAX",
        "sharpe_all": 3.1,
        "sharpe_train": 2.1,
        "sharpe_val": 3.8,
        "balanced_sharpe": 2.1,
        "stability": 0.0,
        "annualized_ret": 1.1,
        "max_drawdown": 0.16,
        "train_val_gap": 1.7,
    }

    assert not apply_hard_filters(good_a, selection_cfg)
    assert not apply_hard_filters(good_b, selection_cfg)
    assert "max_train_val_gap" in apply_hard_filters(bad_gap, selection_cfg)

    for item in (good_a, good_b, bad_gap):
        item["selection_score"] = compute_selection_score(item, weights)

    passed = [item for item in (good_a, good_b, bad_gap) if not apply_hard_filters(item, selection_cfg)]

    selected = select_diverse_top_k(
        passed,
        top_k=2,
        jaccard_threshold=0.75,
    )

    assert len(selected) == 2
    assert bad_gap not in selected
