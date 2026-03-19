import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

import torch
import yaml

from .backtest import CBBacktest
from .config import ModelConfig, RobustConfig
from .config_loader import load_config
from .data_loader import CBDataLoader
from .factor_ai_review import (
    DEFAULT_AI_REVIEW_CONFIG,
    render_markdown_report,
    review_candidates_with_ai,
)
from .formula_simplifier import formula_to_canonical_key, simplify_formula
from .signal_utils import build_topk_weights, default_min_valid_count
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
        "similarity_mode": "hybrid",
        "jaccard_threshold": 0.75,
        "holding_jaccard_threshold": 0.80,
        "return_corr_threshold": 0.90,
        "return_corr_abs": True,
        "similarity_min_overlap_days": 20,
        "behavior_fallback_to_formula": True,
        "weights": {
            "sharpe_all": 0.30,
            "balanced_sharpe": 0.25,
            "stability": 0.20,
            "annualized_ret": 0.15,
            "max_drawdown": -0.10,
            "train_val_gap": -0.20,
        },
    },
    "ai_review": DEFAULT_AI_REVIEW_CONFIG,
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
    config = copy.deepcopy(DEFAULT_SELECTION_CONFIG)
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


def _compute_behavior_signature(factors: torch.Tensor, ctx: Dict[str, Any]) -> Dict[str, Any]:
    loader: CBDataLoader = ctx["loader"]
    bt: CBBacktest = ctx["bt"]
    t_count = int(factors.shape[0])

    weights, valid_trading_day, _, daily_holdings = build_topk_weights(
        factors=factors,
        valid_mask=loader.tradable_mask,
        top_k=bt.top_k,
        min_valid_count=default_min_valid_count(
            top_k=bt.top_k,
            override=RobustConfig.SIGNAL_MIN_VALID_COUNT,
            floor=RobustConfig.MIN_VALID_COUNT,
        ),
        clean_enabled=RobustConfig.SIGNAL_CLEAN_ENABLED,
        winsor_q=RobustConfig.SIGNAL_WINSOR_Q,
        clip_value=RobustConfig.SIGNAL_CLIP,
        rank_output=RobustConfig.SIGNAL_RANK_OUTPUT,
    )

    prev_weights = torch.roll(weights, 1, dims=0)
    prev_weights[0] = 0
    turnover = torch.abs(weights - prev_weights).sum(dim=1)
    tx_cost = turnover * bt.fee_rate * 2

    effective_ret = loader.target_ret.clone()
    tp_extra_cost = torch.zeros(t_count, device=factors.device, dtype=factors.dtype)
    if (
        bt.take_profit > 0
        and ctx["open_prices"] is not None
        and ctx["high_prices"] is not None
        and ctx["prev_close"] is not None
    ):
        open_prices = ctx["open_prices"]
        high_prices = ctx["high_prices"]
        prev_close = ctx["prev_close"]
        valid_price_mask = (
            (prev_close > 0)
            & (prev_close < 10000)
            & (open_prices > 0)
            & (open_prices < 10000)
            & (high_prices > 0)
            & (high_prices < 10000)
        )
        tp_trigger_price = prev_close * (1 + bt.take_profit)
        tp_holding_mask = prev_weights > 0
        open_gap_up = (open_prices >= tp_trigger_price) & valid_price_mask
        gap_up_mask = open_gap_up & tp_holding_mask
        intraday_tp = (high_prices >= tp_trigger_price) & (~open_gap_up) & valid_price_mask
        intra_tp_mask = intraday_tp & tp_holding_mask

        open_ret = (open_prices / prev_close) - 1.0
        effective_ret[gap_up_mask] = open_ret[gap_up_mask]
        effective_ret[intra_tp_mask] = bt.take_profit

        daily_k = tp_holding_mask.sum(dim=1).float()
        gap_up_count = gap_up_mask.sum(dim=1).float()
        intra_tp_count = intra_tp_mask.sum(dim=1).float()
        safe_k = torch.where(daily_k > 0, daily_k, torch.ones_like(daily_k))
        tp_extra_cost += (gap_up_count + intra_tp_count) * 2 * bt.fee_rate / safe_k

    gross_ret = (weights * effective_ret).sum(dim=1)
    net_ret = gross_ret - tx_cost - tp_extra_cost
    return {
        "daily_holdings": daily_holdings,
        "daily_net_ret": net_ret.detach().cpu().tolist(),
        "valid_trading_day": valid_trading_day.detach().cpu().tolist(),
    }


def _calc_holding_jaccard_similarity(sig_a: Dict[str, Any], sig_b: Dict[str, Any], min_overlap_days: int) -> Optional[float]:
    holdings_a = sig_a.get("daily_holdings") or []
    holdings_b = sig_b.get("daily_holdings") or []
    valid_a = sig_a.get("valid_trading_day") or []
    valid_b = sig_b.get("valid_trading_day") or []
    day_count = min(len(holdings_a), len(holdings_b), len(valid_a), len(valid_b))
    if day_count <= 0:
        return None

    scores: List[float] = []
    for day_idx in range(day_count):
        if not (bool(valid_a[day_idx]) and bool(valid_b[day_idx])):
            continue
        set_a = set(int(x) for x in holdings_a[day_idx])
        set_b = set(int(x) for x in holdings_b[day_idx])
        union_count = len(set_a | set_b)
        if union_count == 0:
            continue
        scores.append(float(len(set_a & set_b)) / float(union_count))

    if len(scores) < int(min_overlap_days):
        return None
    return float(sum(scores) / len(scores))


def _calc_return_corr(sig_a: Dict[str, Any], sig_b: Dict[str, Any], min_overlap_days: int, return_corr_abs: bool) -> Optional[float]:
    returns_a = sig_a.get("daily_net_ret") or []
    returns_b = sig_b.get("daily_net_ret") or []
    valid_a = sig_a.get("valid_trading_day") or []
    valid_b = sig_b.get("valid_trading_day") or []
    day_count = min(len(returns_a), len(returns_b), len(valid_a), len(valid_b))
    if day_count <= 0:
        return None

    aligned_a: List[float] = []
    aligned_b: List[float] = []
    for day_idx in range(day_count):
        if not (bool(valid_a[day_idx]) and bool(valid_b[day_idx])):
            continue
        aligned_a.append(float(returns_a[day_idx]))
        aligned_b.append(float(returns_b[day_idx]))

    if len(aligned_a) < int(min_overlap_days):
        return None

    x = torch.tensor(aligned_a, dtype=torch.float64)
    y = torch.tensor(aligned_b, dtype=torch.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_std = float(x_centered.std(unbiased=False).item())
    y_std = float(y_centered.std(unbiased=False).item())
    if x_std < 1e-12 or y_std < 1e-12:
        corr = 0.0
    else:
        cov = float((x_centered * y_centered).mean().item())
        corr = cov / (x_std * y_std)
        corr = max(-1.0, min(1.0, corr))
    return abs(corr) if return_corr_abs else corr


def _resolve_similarity_mode(selection_cfg: Dict[str, Any]) -> str:
    raw_mode = str(selection_cfg.get("similarity_mode", "hybrid")).strip().lower()
    if raw_mode in {"formula", "formula_only", "token_jaccard"}:
        return "formula"
    if raw_mode in {"behavior", "behavior_only"}:
        return "behavior"
    return "hybrid"


def _check_similarity_rejection(
    candidate: Dict[str, Any],
    chosen: Dict[str, Any],
    selection_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    similarity_mode = _resolve_similarity_mode(selection_cfg)
    formula_threshold = float(selection_cfg.get("jaccard_threshold", 0.75))
    holding_threshold = float(selection_cfg.get("holding_jaccard_threshold", 0.80))
    corr_threshold = float(selection_cfg.get("return_corr_threshold", 0.90))
    corr_abs = bool(selection_cfg.get("return_corr_abs", True))
    min_overlap_days = int(selection_cfg.get("similarity_min_overlap_days", 20))
    behavior_fallback_to_formula = bool(selection_cfg.get("behavior_fallback_to_formula", True))

    formula_jaccard = calculate_jaccard_similarity(candidate["formula"], chosen["formula"])
    formula_reject = formula_jaccard > formula_threshold

    holding_jaccard: Optional[float] = None
    return_corr: Optional[float] = None
    behavior_reject = False
    behavior_ready = False
    sig_a = candidate.get("_behavior_signature")
    sig_b = chosen.get("_behavior_signature")
    if isinstance(sig_a, dict) and isinstance(sig_b, dict):
        holding_jaccard = _calc_holding_jaccard_similarity(sig_a, sig_b, min_overlap_days)
        return_corr = _calc_return_corr(sig_a, sig_b, min_overlap_days, corr_abs)
        behavior_ready = holding_jaccard is not None or return_corr is not None
        behavior_reject = (
            (holding_jaccard is not None and holding_jaccard > holding_threshold)
            or (return_corr is not None and return_corr > corr_threshold)
        )

    if similarity_mode == "formula":
        reject = formula_reject
    elif similarity_mode == "behavior":
        reject = behavior_reject or (formula_reject and behavior_fallback_to_formula and not behavior_ready)
    else:
        reject = formula_reject or behavior_reject

    return {
        "reject": bool(reject),
        "similarity_mode": similarity_mode,
        "formula_jaccard": float(formula_jaccard),
        "formula_threshold": formula_threshold,
        "holding_jaccard": holding_jaccard,
        "holding_threshold": holding_threshold,
        "return_corr": return_corr,
        "return_corr_threshold": corr_threshold,
        "return_corr_abs": corr_abs,
        "behavior_ready": behavior_ready,
        "fallback_formula_applied": bool(
            similarity_mode == "behavior"
            and behavior_fallback_to_formula
            and not behavior_ready
            and formula_reject
        ),
    }


def _public_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def reevaluate_candidate(candidate: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    loader: CBDataLoader = ctx["loader"]
    vm: StackVM = ctx["vm"]
    bt: CBBacktest = ctx["bt"]

    factors = vm.execute(candidate["formula"], loader.feat_tensor, cs_mask=loader.cs_mask)
    if factors is None:
        return None

    metrics = bt.evaluate_robust(
        factors=factors,
        target_ret=loader.target_ret,
        valid_mask=loader.tradable_mask,
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
    enriched["_behavior_signature"] = _compute_behavior_signature(factors, ctx)
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
    selection_cfg: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected: List[Dict[str, Any]] = []
    rejected_similarity: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda x: x["selection_score"], reverse=True):
        blocked = False
        for chosen in selected:
            check = _check_similarity_rejection(candidate, chosen, selection_cfg)
            if check["reject"]:
                blocked = True
                rejected_similarity.append(
                    {
                        "readable": candidate.get("readable"),
                        "source": candidate.get("source"),
                        "selection_score": candidate.get("selection_score"),
                        "blocked_by": chosen.get("readable"),
                        "check": check,
                    }
                )
                break
        if not blocked:
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected, rejected_similarity


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
    total_candidates = len(deduped_candidates)
    print(
        f"Reevaluating {total_candidates} deduped candidates "
        f"(raw={len(raw_candidates)})..."
    )
    for candidate in deduped_candidates:
        idx = len(reevaluated) + len(rejected_eval) + 1
        if idx == 1 or idx % 10 == 0 or idx == total_candidates:
            print(
                f"[Selection] evaluating {idx}/{total_candidates} "
                f"| source={candidate['source']} step={candidate.get('step')}"
            )
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
    print(
        f"Selection pass summary: reevaluated={len(reevaluated)}, "
        f"passed={len(passed)}, eval_rejected={len(rejected_eval)}"
    )
    selected, rejected_similarity = select_diverse_top_k(
        passed,
        top_k=top_k,
        selection_cfg=selection_cfg,
    )
    shortlist = sorted(passed, key=lambda x: x["selection_score"], reverse=True)[: max(top_k * 3, 10)]

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
            "similarity_rejected": len(rejected_similarity),
        },
        "selected": [_public_candidate(c) for c in selected],
        "shortlist": [_public_candidate(c) for c in shortlist],
        "rejected_eval": rejected_eval,
        "rejected_similarity": rejected_similarity,
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
    parser.add_argument("--ai-base-url", type=str, default=None)
    parser.add_argument("--ai-max-candidates", type=int, default=None)
    parser.add_argument("--ai-timeout-sec", type=float, default=None)
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
        shortlist = result.get("shortlist", [])
        max_ai_candidates = int(args.ai_max_candidates or ai_cfg.get("max_candidates", 10))
        provider = str(args.ai_provider or ai_cfg.get("provider", "openai"))
        model = str(args.ai_model or ai_cfg.get("model", "gpt-5"))
        base_url = args.ai_base_url or ai_cfg.get("base_url")
        print(
            f"Starting AI review: provider={provider}, model={model}, "
            f"candidates={min(len(shortlist), max_ai_candidates)}"
        )
        ai_reviews = review_candidates_with_ai(
            shortlist,
            provider=provider,
            model=model,
            max_candidates=max_ai_candidates,
            base_url=base_url,
            timeout_sec=args.ai_timeout_sec,
            ai_review_config=ai_cfg,
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
