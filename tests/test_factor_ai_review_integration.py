import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from openai import APITimeoutError, OpenAI

from model_core.factor_ai_review import (
    build_review_payload,
    build_system_prompt,
    build_user_prompt,
    parse_review_response,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_CONFIG_PATH = PROJECT_ROOT / "model_core" / "top_factor_config.yaml"
DEFAULT_TOP_FACTORS_PATH = PROJECT_ROOT / "model_core" / "top3_factors.json"
DEFAULT_BEST_FORMULA_PATH = PROJECT_ROOT / "model_core" / "best_cb_formula.json"

DEFAULT_BASE_URL = "https://once.novai.su/v1"
DEFAULT_MODEL = "[次]gemini-3.1-pro-preview"
DEFAULT_TIMEOUT_SEC = 180.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_SLEEP_SEC = 3.0


def load_ai_review_config(
    selection_config_path: Path = DEFAULT_SELECTION_CONFIG_PATH,
) -> Dict[str, Any]:
    if not selection_config_path.exists():
        return {}
    with selection_config_path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    ai_review = obj.get("ai_review", {})
    return ai_review if isinstance(ai_review, dict) else {}


def load_real_king_factor(
    top_factors_path: Path = DEFAULT_TOP_FACTORS_PATH,
    best_formula_path: Path = DEFAULT_BEST_FORMULA_PATH,
) -> Dict[str, Any]:
    if top_factors_path.exists():
        with top_factors_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        selected = obj.get("selected", [])
        if isinstance(selected, list) and selected:
            return selected[0]

    with best_formula_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    best = obj.get("best", {})
    if not isinstance(best, dict) or not best:
        raise ValueError("No usable king factor found in top3_factors.json or best_cb_formula.json")

    return {
        "formula": best.get("formula", []),
        "readable": best.get("readable", ""),
        "selection_score": best.get("score"),
        "sharpe_all": best.get("sharpe"),
        "sharpe_train": best.get("sharpe"),
        "sharpe_val": best.get("sharpe"),
        "balanced_sharpe": best.get("sharpe"),
        "train_val_gap": 0.0,
        "stability": 0.0,
        "annualized_ret": best.get("annualized_ret"),
        "max_drawdown": None,
        "active_ratio": None,
        "valid_days_train": None,
        "valid_days_val": None,
    }


def run_real_ai_review(
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    selection_config_path: Path = DEFAULT_SELECTION_CONFIG_PATH,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    max_retries: int = DEFAULT_MAX_RETRIES,
    print_raw: bool = True,
) -> Dict[str, Any]:
    ai_review_config = load_ai_review_config(selection_config_path)
    candidate = load_real_king_factor()
    payload = build_review_payload(candidate)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(timeout_sec))
    system_prompt = build_system_prompt(ai_review_config=ai_review_config)
    user_prompt = build_user_prompt(candidate, ai_review_config=ai_review_config)

    response = None
    total_attempts = max(1, int(max_retries) + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            print(
                f"[AI Review Integration] attempt={attempt}/{total_attempts} "
                f"timeout={float(timeout_sec):.0f}s model={model}"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            break
        except APITimeoutError as exc:
            if attempt >= total_attempts:
                raise RuntimeError(
                    "AI review request timed out. "
                    f"Increase --timeout-sec above {float(timeout_sec):.0f}, "
                    "reduce model latency, or switch to a non-thinking model."
                ) from exc
            print(
                f"[AI Review Integration] timeout on attempt {attempt}/{total_attempts}, "
                f"sleep {DEFAULT_RETRY_SLEEP_SEC:.0f}s before retry"
            )
            time.sleep(DEFAULT_RETRY_SLEEP_SEC)

    if response is None:
        raise RuntimeError("AI review request failed without a response")

    text = response.choices[0].message.content if response.choices else ""
    review = parse_review_response(
        text,
        ai_review_config=ai_review_config,
        candidate=candidate,
    )

    if print_raw:
        print("=== King Factor ===")
        print(candidate.get("readable", ""))
        print("")
        print("=== Expanded Formula ===")
        print(payload.get("expanded_formula", ""))
        print("")
        print("=== Semantic Expanded Formula ===")
        print(payload.get("semantic_expanded_formula", ""))
        print("")
        print("=== Raw Response ===")
        print(text)
        print("")
        print("=== Parsed Review ===")
        print(json.dumps(review, ensure_ascii=False, indent=2))
        print("")
        print(f"model={getattr(response, 'model', model)}")
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        if total_tokens is not None:
            print(f"total_tokens={total_tokens}")

    return review


@pytest.mark.integration
def test_real_king_factor_ai_review_integration() -> None:
    if os.getenv("RUN_REAL_AI_REVIEW") != "1":
        pytest.skip("Set RUN_REAL_AI_REVIEW=1 to run the real API integration test.")

    api_key = (
        os.getenv("NOVAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("AI_REVIEW_API_KEY")
    )
    if not api_key:
        pytest.skip("Missing NOVAI_API_KEY / OPENAI_API_KEY / AI_REVIEW_API_KEY.")

    base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    model = os.getenv("AI_REVIEW_MODEL") or DEFAULT_MODEL
    timeout_sec = float(os.getenv("AI_REVIEW_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC)))
    max_retries = int(os.getenv("AI_REVIEW_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    review = run_real_ai_review(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        print_raw=True,
    )

    assert isinstance(review, dict)
    assert review["review_decision"] in {"keep", "watch", "drop"}
    assert isinstance(review["summary"], str) and review["summary"].strip()
    assert isinstance(review["theme_tags"], list)
    assert isinstance(review["logic_chain"], list)
    assert isinstance(review["risks"], list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real AI review integration call for the current king factor."
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--selection-config",
        type=str,
        default=str(DEFAULT_SELECTION_CONFIG_PATH),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    api_key = (
        args.api_key
        or os.getenv("NOVAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("AI_REVIEW_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "Missing API key. Set NOVAI_API_KEY / OPENAI_API_KEY / AI_REVIEW_API_KEY or pass --api-key."
        )

    run_real_ai_review(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        selection_config_path=Path(args.selection_config),
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        print_raw=not args.quiet,
    )


if __name__ == "__main__":
    main()
