import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
import yaml

from .backtest import CBBacktest
from .config import ModelConfig, RobustConfig
from .config_loader import load_config
from .data_loader import CBDataLoader
from .factor_ai_review import render_markdown_report, review_candidates_with_ai
from .formula_simplifier import formula_to_canonical_key, simplify_formula
from .vm import StackVM


DEFAULT_INPUT_PATH = os.path.join(os.path.dirname(__file__), "best_cb_formula.json")
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "top3_factors.json")
DEFAULT_REPORT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "top3_factors_report.md")
DEFAULT_AI_REVIEW_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "top3_factor_ai_reviews.json",
)
DEFAULT_SELECTION_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "top_factor_config.yaml",
)

DEFAULT_SELECTION_CONFIG: Dict[str, Any] = {
    "selection": {
        "top_k": 3,
        "candidate_sources": ["best", "history", "diverse_top_50"],
        "min_sharpe_all": 2.4,
        "min_sharpe_train": 2.1,
        "min_sharpe_val": 2.5,
        "min_balanced_sharpe": 2.1,
        "min_stability": -0.4,
        "max_drawdown": 0.20,
        "max_train_val_gap": 1.0,
        "jaccard_threshold": 0.75,
        "weights": {
            "sharpe_all": 0.30,
            "balanced_sharpe": 0.25,
            "stability": 0.20,
            "annualized_ret": 0.15,
            "max_drawdown": -0.10,
            "train_val_gap": -0.20,
        },
    }
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_selection_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = dict(DEFAULT_SELECTION_CONFIG)
    target_path = config_path or DEFAULT_SELECTION_CONFIG_PATH
    if target_path and os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            custom = yaml.safe_load(f) or {}
        config = _deep_merge(config, custom)
    return config


def _coerce_formula(item: Dict[str, Any]) -> Optional[List[str]]:
    formula = item.get("formula")
    if isinstance(formula, list) and formula and isinstance(formula[0], str):
        return list(formula)
    raw_formula = item.get("raw_formula")
    if isinstance(raw_formula, list) and raw_formula and isinstance(raw_formula[0], str):
        return list(raw_formula)
    readable = item.get("readable")
    if isinstance(readable, str) and readable.strip():
        return readable.strip().split()
    raw_readable = item.get("raw_readable")
    if isinstance(raw_readable, str) and raw_readable.strip():
        return raw_readable.strip().split()
    return None


def load_candidates(input_path: str, candidate_sources: List[str]) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    candidates: List[Dict[str, Any]] = []
    for source in candidate_sources:
        if source == "best":
            best = obj.get("best")
            if isinstance(best, dict):
                formula = _coerce_formula(best)
                if formula:
                    candidates.append(
                        {
                            "source": "best",
                            "step": best.get("step"),
                            "formula": formula,
                            "raw_formula": best.get("raw_formula"),
                            "readable": best.get("readable"),
                            "raw_readable": best.get("raw_readable"),
                            "original_score": best.get("score"),
                            "original_sharpe": best.get("sharpe"),
                            "original_annualized_ret": best.get("annualized_ret"),
                        }
                    )
        else:
            items = obj.get(source, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                formula = _coerce_formula(item)
                if not formula:
                    continue
                candidates.append(
                    {
                        "source": source,
                        "step": item.get("step"),
                        "formula": formula,
                        "raw_formula": item.get("raw_formula"),
                        "readable": item.get("readable"),
                        "raw_readable": item.get("raw_readable"),
                        "original_score": item.get("score"),
                        "original_sharpe": item.get("sharpe"),
                        "original_sharpe_train": item.get("sharpe_train"),
                        "original_sharpe_val": item.get("sharpe_val"),
                        "original_stability": item.get("stability"),
                        "original_annualized_ret": item.get("annualized_ret"),
                        "original_max_drawdown": item.get("max_drawdown"),
                    }
                )
    return candidates


def _candidate_priority(candidate: Dict[str, Any]) -> tuple:
    metrics_count = sum(
        1
        for key in (
            "original_score",
            "original_sharpe",
            "original_sharpe_train",
            "original_sharpe_val",
            "original_stability",
            "original_annualized_ret",
            "original_max_drawdown",
        )
        if candidate.get(key) is not None
    )
    return (
        metrics_count,
        float(candidate.get("original_score") or float("-inf")),
        1 if candidate.get("source") == "history" else 0,
    )


def dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[tuple[str, ...], Dict[str, Any]] = {}
    for candidate in candidates:
        simplified = simplify_formula(candidate["formula"])
        canonical_key = formula_to_canonical_key(simplified)
        normalized = dict(candidate)
        normalized["formula"] = simplified
        normalized["readable"] = " ".join(simplified)
        normalized["canonical_key"] = list(canonical_key)
        if (
            canonical_key not in deduped
            or _candidate_priority(normalized) > _candidate_priority(deduped[canonical_key])
        ):
            deduped[canonical_key] = normalized
    return list(deduped.values())


def _build_take_profit_inputs(loader: CBDataLoader) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if RobustConfig.TAKE_PROFIT <= 0:
        return None, None, None
    if "OPEN" not in loader.raw_data_cache or "HIGH" not in loader.raw_data_cache:
        return None, None, None

    open_prices = torch.roll(loader.raw_data_cache["OPEN"], -1, dims=0).clone()
    high_prices = torch.roll(loader.raw_data_cache["HIGH"], -1, dims=0).clone()
    prev_close = loader.raw_data_cache["CLOSE"].clone()
    open_prices[-1] = 1e9
    high_prices[-1] = 1e9
    return open_prices, high_prices, prev_close


def build_eval_context(data_start_date: Optional[str]) -> Dict[str, Any]:
    loader = CBDataLoader()
    loader.load_data(start_date=data_start_date or "2022-08-01")
    vm = StackVM()
    bt = CBBacktest(top_k=RobustConfig.TOP_K, take_profit=RobustConfig.TAKE_PROFIT)
    open_prices, high_prices, prev_close = _build_take_profit_inputs(loader)
    return {
        "loader": loader,
        "vm": vm,
        "bt": bt,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "prev_close": prev_close,
    }


def reevaluate_candidate(candidate: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    loader: CBDataLoader = ctx["loader"]
    vm: StackVM = ctx["vm"]
    bt: CBBacktest = ctx["bt"]

    factors = vm.execute(candidate["formula"], loader.feat_tensor)
    if factors is None:
        return None

    metrics = bt.evaluate_robust(
        factors=factors,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask,
        split_idx=loader.split_idx,
        open_prices=ctx["open_prices"],
        high_prices=ctx["high_prices"],
        prev_close=ctx["prev_close"],
    )

    enriched = dict(candidate)
    enriched.update(
        {
            "sharpe_all": float(metrics.get("sharpe_all", 0.0)),
            "sharpe_train": float(metrics.get("sharpe_train", 0.0)),
            "sharpe_val": float(metrics.get("sharpe_val", 0.0)),
            "stability": float(metrics.get("stability_metric", 0.0)),
            "annualized_ret": float(metrics.get("annualized_ret", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "active_ratio": float(metrics.get("active_ratio", 0.0)),
            "valid_days_train": int(metrics.get("valid_days_train", 0)),
            "valid_days_val": int(metrics.get("valid_days_val", 0)),
            "valid_ic_days": int(metrics.get("valid_ic_days", 0)),
            "balanced_sharpe": float(
                min(metrics.get("sharpe_train", 0.0), metrics.get("sharpe_val", 0.0))
            ),
            "train_val_gap": float(
                abs(metrics.get("sharpe_train", 0.0) - metrics.get("sharpe_val", 0.0))
            ),
        }
    )
    return enriched


def apply_hard_filters(candidate: Dict[str, Any], selection_cfg: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if candidate["sharpe_all"] < float(selection_cfg["min_sharpe_all"]):
        reasons.append("min_sharpe_all")
    if candidate["sharpe_train"] < float(selection_cfg["min_sharpe_train"]):
        reasons.append("min_sharpe_train")
    if candidate["sharpe_val"] < float(selection_cfg["min_sharpe_val"]):
        reasons.append("min_sharpe_val")
    if candidate["balanced_sharpe"] < float(selection_cfg["min_balanced_sharpe"]):
        reasons.append("min_balanced_sharpe")
    if candidate["stability"] < float(selection_cfg["min_stability"]):
        reasons.append("min_stability")
    if candidate["max_drawdown"] > float(selection_cfg["max_drawdown"]):
        reasons.append("max_drawdown")
    if candidate["train_val_gap"] > float(selection_cfg["max_train_val_gap"]):
        reasons.append("max_train_val_gap")
    return reasons


def compute_selection_score(candidate: Dict[str, Any], weights: Dict[str, Any]) -> float:
    return (
        float(weights["sharpe_all"]) * candidate["sharpe_all"]
        + float(weights["balanced_sharpe"]) * candidate["balanced_sharpe"]
        + float(weights["stability"]) * candidate["stability"]
        + float(weights["annualized_ret"]) * candidate["annualized_ret"]
        + float(weights["max_drawdown"]) * candidate["max_drawdown"]
        + float(weights["train_val_gap"]) * candidate["train_val_gap"]
    )


def calculate_jaccard_similarity(formula_a: List[str], formula_b: List[str]) -> float:
    set_a = set(formula_a)
    set_b = set(formula_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def select_diverse_top_k(
    candidates: List[Dict[str, Any]],
    top_k: int,
    jaccard_threshold: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda x: x["selection_score"], reverse=True):
        if all(
            calculate_jaccard_similarity(candidate["formula"], chosen["formula"]) <= jaccard_threshold
            for chosen in selected
        ):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def run_selection(
    input_path: str,
    output_path: str,
    training_config_path: Optional[str],
    selection_config_path: Optional[str],
    data_start_date: Optional[str],
    top_k_override: Optional[int],
) -> Dict[str, Any]:
    load_config(training_config_path)
    selection_bundle = load_selection_config(selection_config_path)
    selection_cfg = selection_bundle["selection"]
    top_k = int(top_k_override or selection_cfg["top_k"])

    raw_candidates = load_candidates(input_path, list(selection_cfg["candidate_sources"]))
    deduped_candidates = dedupe_candidates(raw_candidates)
    ctx = build_eval_context(data_start_date)

    reevaluated: List[Dict[str, Any]] = []
    rejected_eval: List[Dict[str, Any]] = []
    for candidate in deduped_candidates:
        enriched = reevaluate_candidate(candidate, ctx)
        if enriched is None:
            rejected_eval.append(
                {
                    "readable": candidate["readable"],
                    "source": candidate["source"],
                    "reason": "vm_execute_none",
                }
            )
            continue
        fail_reasons = apply_hard_filters(enriched, selection_cfg)
        enriched["filter_reasons"] = fail_reasons
        enriched["selection_score"] = compute_selection_score(enriched, selection_cfg["weights"])
        reevaluated.append(enriched)

    passed = [c for c in reevaluated if not c["filter_reasons"]]
    selected = select_diverse_top_k(
        passed,
        top_k=top_k,
        jaccard_threshold=float(selection_cfg["jaccard_threshold"]),
    )

    result = {
        "input_path": os.path.abspath(input_path),
        "training_config_path": training_config_path or "default_config.yaml",
        "selection_config_path": selection_config_path or DEFAULT_SELECTION_CONFIG_PATH,
        "data_start_date": data_start_date or "2022-08-01",
        "selection_config": selection_cfg,
        "counts": {
            "raw_candidates": len(raw_candidates),
            "deduped_candidates": len(deduped_candidates),
            "reevaluated_candidates": len(reevaluated),
            "hard_filter_passed": len(passed),
            "selected_top_k": len(selected),
            "eval_rejected": len(rejected_eval),
        },
        "selected": selected,
        "shortlist": sorted(passed, key=lambda x: x["selection_score"], reverse=True)[: max(top_k * 3, 10)],
        "rejected_eval": rejected_eval,
        "filter_failures": [
            {
                "readable": c["readable"],
                "source": c["source"],
                "filter_reasons": c["filter_reasons"],
                "selection_score": c["selection_score"],
            }
            for c in reevaluated
            if c["filter_reasons"]
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top-K factors from best_cb_formula.json using quantitative post-screening."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", type=str, default=None, help="training config path")
    parser.add_argument(
        "--selection-config",
        type=str,
        default=None,
        help="selection config path",
    )
    parser.add_argument("--data-start-date", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--report-output", type=str, default=DEFAULT_REPORT_OUTPUT_PATH)
    parser.add_argument("--ai-review-output", type=str, default=DEFAULT_AI_REVIEW_OUTPUT_PATH)
    parser.add_argument("--enable-ai-review", action="store_true")
    parser.add_argument("--ai-provider", type=str, default=None)
    parser.add_argument("--ai-model", type=str, default=None)
    parser.add_argument("--ai-max-candidates", type=int, default=None)
    args = parser.parse_args()

    selection_bundle = load_selection_config(args.selection_config)
    ai_cfg = selection_bundle.get("ai_review", {})
    result = run_selection(
        input_path=args.input,
        output_path=args.output,
        training_config_path=args.config,
        selection_config_path=args.selection_config,
        data_start_date=args.data_start_date,
        top_k_override=args.top_k,
    )

    ai_reviews = None
    if args.enable_ai_review:
        ai_reviews = review_candidates_with_ai(
            result.get("shortlist", []),
            provider=str(args.ai_provider or ai_cfg.get("provider", "openai")),
            model=str(args.ai_model or ai_cfg.get("model", "gpt-5")),
            max_candidates=int(args.ai_max_candidates or ai_cfg.get("max_candidates", 10)),
        )
        with open(args.ai_review_output, "w", encoding="utf-8") as f:
            json.dump(ai_reviews, f, ensure_ascii=False, indent=2)

    report = render_markdown_report(result, ai_reviews=ai_reviews)
    with open(args.report_output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved top factors to: {args.output}")
    print(f"Saved report to: {args.report_output}")
    if ai_reviews is not None:
        print(f"Saved AI reviews to: {args.ai_review_output}")
        failed_ai_reviews = sum(1 for item in ai_reviews if item.get("error"))
        if failed_ai_reviews:
            print(
                f"AI review warnings: {failed_ai_reviews}/{len(ai_reviews)} candidates fell back "
                "to placeholder reviews due to provider errors."
            )
    print(
        "Counts: "
        f"raw={result['counts']['raw_candidates']}, "
        f"deduped={result['counts']['deduped_candidates']}, "
        f"passed={result['counts']['hard_filter_passed']}, "
        f"selected={result['counts']['selected_top_k']}"
    )
    for rank, candidate in enumerate(result["selected"], start=1):
        print(
            f"#{rank} "
            f"score={candidate['selection_score']:.4f} "
            f"Sharpe A/T/V={candidate['sharpe_all']:.2f}/"
            f"{candidate['sharpe_train']:.2f}/{candidate['sharpe_val']:.2f} "
            f"Stab={candidate['stability']:.3f} "
            f"AR={candidate['annualized_ret']:.2%} "
            f"MDD={candidate['max_drawdown']:.1%} "
            f"| {candidate['readable']}"
        )


if __name__ == "__main__":
    main()
