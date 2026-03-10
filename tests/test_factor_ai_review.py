from model_core.factor_ai_review import (
    build_review_payload,
    parse_review_response,
    render_markdown_report,
    review_candidates_with_ai,
)


def test_build_review_payload_contains_metrics_and_context(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["PURE_VALUE", "PREM_Z"],
        raising=False,
    )

    candidate = {
        "formula": ["PURE_VALUE", "PREM_Z", "MIN"],
        "readable": "PURE_VALUE PREM_Z MIN",
        "selection_score": 1.23,
        "sharpe_all": 3.1,
        "sharpe_train": 2.8,
        "sharpe_val": 3.4,
        "balanced_sharpe": 2.8,
        "train_val_gap": 0.6,
        "stability": 0.2,
        "annualized_ret": 1.0,
        "max_drawdown": 0.15,
    }

    payload = build_review_payload(candidate)

    assert payload["formula"] == ["PURE_VALUE", "PREM_Z", "MIN"]
    assert payload["metrics"]["sharpe_train"] == 2.8
    assert payload["context"]["asset_type"] == "convertible_bond"
    assert payload["context"]["input_features"] == ["PURE_VALUE", "PREM_Z"]


def test_parse_review_response_extracts_json_object():
    text = """
    Here is the review:
    {
      "theme_tags": ["valuation"],
      "style_label": "defensive",
      "financial_coherence_score": 0.8,
      "interpretability_score": 0.7,
      "regime_dependency_score": 0.3,
      "redundancy_risk_score": 0.2,
      "implementation_confidence": 0.9,
      "summary": "逻辑清晰",
      "logic_chain": ["PURE_VALUE 提供估值锚"],
      "risks": ["验证偏强"],
      "review_decision": "keep"
    }
    """

    review = parse_review_response(text)

    assert review["theme_tags"] == ["valuation"]
    assert review["review_decision"] == "keep"
    assert review["summary"] == "逻辑清晰"


def test_render_markdown_report_includes_ai_section():
    selection_result = {
        "counts": {
            "raw_candidates": 10,
            "deduped_candidates": 8,
            "hard_filter_passed": 4,
            "selected_top_k": 3,
        },
        "selected": [
            {
                "selection_score": 1.5,
                "sharpe_all": 3.0,
                "sharpe_train": 2.8,
                "sharpe_val": 3.2,
                "stability": 0.1,
                "annualized_ret": 1.0,
                "max_drawdown": 0.15,
                "readable": "PURE_VALUE PREM_Z MIN",
            }
        ],
    }
    ai_reviews = [
        {
            "readable": "PURE_VALUE PREM_Z MIN",
            "review": {
                "theme_tags": ["valuation"],
                "style_label": "defensive",
                "financial_coherence_score": 0.8,
                "interpretability_score": 0.7,
                "regime_dependency_score": 0.3,
                "redundancy_risk_score": 0.2,
                "review_decision": "keep",
                "summary": "逻辑清晰",
                "logic_chain": ["PURE_VALUE 提供估值锚"],
                "risks": ["验证偏强"],
            },
        }
    ]

    report = render_markdown_report(selection_result, ai_reviews=ai_reviews)

    assert "## AI Reviews" in report
    assert "valuation" in report
    assert "PURE_VALUE PREM_Z MIN" in report


def test_review_candidates_with_ai_dispatches_openai(monkeypatch):
    candidate = {
        "formula": ["PURE_VALUE", "PREM_Z", "MIN"],
        "readable": "PURE_VALUE PREM_Z MIN",
        "selection_score": 1.23,
    }

    monkeypatch.setattr(
        "model_core.factor_ai_review.review_candidate_with_openai",
        lambda candidate, model, api_key=None, timeout_sec=None: {
            "review_decision": "keep",
            "summary": model,
        },
    )

    reviews = review_candidates_with_ai(
        [candidate],
        provider="openai",
        model="gpt-5",
        max_candidates=1,
    )

    assert len(reviews) == 1
    assert reviews[0]["provider"] == "openai"
    assert reviews[0]["model"] == "gpt-5"
    assert reviews[0]["review"]["review_decision"] == "keep"


def test_review_candidates_with_ai_dispatches_gemini(monkeypatch):
    candidate = {
        "formula": ["PURE_VALUE", "PREM_Z", "MIN"],
        "readable": "PURE_VALUE PREM_Z MIN",
        "selection_score": 1.23,
    }

    monkeypatch.setattr(
        "model_core.factor_ai_review.review_candidate_with_gemini",
        lambda candidate, model, api_key=None, timeout_sec=None: {
            "review_decision": "watch",
            "summary": model,
        },
    )

    reviews = review_candidates_with_ai(
        [candidate],
        provider="gemini",
        model="gemini-2.0-flash",
        max_candidates=1,
    )

    assert len(reviews) == 1
    assert reviews[0]["provider"] == "gemini"
    assert reviews[0]["model"] == "gemini-2.0-flash"
    assert reviews[0]["review"]["review_decision"] == "watch"


def test_review_candidates_with_ai_dispatches_glm5(monkeypatch):
    candidate = {
        "formula": ["PURE_VALUE", "PREM_Z", "MIN"],
        "readable": "PURE_VALUE PREM_Z MIN",
        "selection_score": 1.23,
    }

    monkeypatch.setattr(
        "model_core.factor_ai_review.review_candidate_with_glm5",
        lambda candidate, model, api_key=None, timeout_sec=None: {
            "review_decision": "keep",
            "summary": model,
        },
    )

    reviews = review_candidates_with_ai(
        [candidate],
        provider="glm5",
        model="ZhipuAI/GLM-5",
        max_candidates=1,
    )

    assert len(reviews) == 1
    assert reviews[0]["provider"] == "glm5"
    assert reviews[0]["model"] == "ZhipuAI/GLM-5"
    assert reviews[0]["review"]["review_decision"] == "keep"


def test_review_candidates_with_ai_falls_back_on_provider_error(monkeypatch):
    candidate = {
        "formula": ["PURE_VALUE", "PREM_Z", "MIN"],
        "readable": "PURE_VALUE PREM_Z MIN",
        "selection_score": 1.23,
    }

    def _raise_quota(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

    monkeypatch.setattr(
        "model_core.factor_ai_review.review_candidate_with_gemini",
        _raise_quota,
    )

    reviews = review_candidates_with_ai(
        [candidate],
        provider="gemini",
        model="gemini-2.0-flash",
        max_candidates=1,
    )

    assert len(reviews) == 1
    assert reviews[0]["error"] == "AI review skipped: provider quota exhausted"
    assert reviews[0]["review"]["review_decision"] == "watch"
    assert "quota exhausted" in reviews[0]["review"]["summary"]


def test_review_candidates_with_ai_rejects_unknown_provider():
    candidate = {
        "formula": ["PURE_VALUE", "PREM_Z", "MIN"],
        "readable": "PURE_VALUE PREM_Z MIN",
        "selection_score": 1.23,
    }

    try:
        review_candidates_with_ai(
            [candidate],
            provider="unknown",
            model="dummy",
            max_candidates=1,
        )
    except ValueError as exc:
        assert "Unsupported AI review provider" in str(exc)
    else:
        raise AssertionError("unknown provider should fail fast")
