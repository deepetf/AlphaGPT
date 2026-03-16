from model_core.factor_ai_review import (
    build_review_payload,
    build_system_prompt,
    build_user_prompt,
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
    assert payload["expanded_formula"] == "MIN(PURE_VALUE, PREM_Z)"
    assert payload["semantic_expanded_formula"] == "min(PURE_VALUE, PREM_Z)"
    assert payload["metrics"]["sharpe_train"] == 2.8
    assert payload["context"]["asset_type"] == "convertible_bond"
    assert payload["context"]["input_features"] == ["PURE_VALUE", "PREM_Z"]
    assert "structure_hints" in payload
    assert payload["feature_metadata"]["PURE_VALUE"]["dimension_hint"] == "level"
    assert payload["feature_metadata"]["PREM_Z"]["native_zscore_like"] is True
    assert payload["feature_metadata"]["PREM_Z"]["pipeline_time_normalized"] is False
    assert payload["operator_metadata"]["MIN"]["arity"] == 2


def test_parse_review_response_extracts_json_object():
    text = """
    Here is the review:
    {
      "theme_tags": ["valuation"],
      "style_label": "defensive",
      "financial_coherence_score": 0.8,
      "interpretability_score": 0.7,
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


def test_build_prompts_can_use_config_override():
    candidate = {
        "formula": ["PURE_VALUE", "PREM_Z", "MIN"],
        "readable": "PURE_VALUE PREM_Z MIN",
        "selection_score": 1.23,
    }
    ai_review_config = {
        "prompt": {
            "system": "自定义 system",
            "user_template": "Schema={schema_json}\nPayload={payload_json}",
        }
    }

    system_prompt = build_system_prompt(ai_review_config=ai_review_config)
    user_prompt = build_user_prompt(candidate, ai_review_config=ai_review_config)

    assert system_prompt == "自定义 system"
    assert "Schema=" in user_prompt
    assert "Payload=" in user_prompt
    assert '"formula"' in user_prompt
    assert '"expanded_formula"' in user_prompt
    assert '"semantic_expanded_formula"' in user_prompt
    assert '"structure_hints"' in user_prompt
    assert '"review_decision": "watch"' in user_prompt
    assert '"feature_metadata"' in user_prompt
    assert '"operator_metadata"' in user_prompt


def test_default_system_prompt_contains_rpn_constraints():
    system_prompt = build_system_prompt()

    assert "逆波兰表达式" in system_prompt
    assert "RPN 栈式顺序正确展开公式" in system_prompt
    assert "不得把 IF_POS 当成普通三元条件表达式" in system_prompt
    assert "semantic_expanded_formula" in system_prompt
    assert "structure_hints" in system_prompt


def test_build_review_payload_semantic_expansion_respects_if_pos_definition():
    candidate = {
        "formula": ["A", "B", "IF_POS"],
        "readable": "A B IF_POS",
    }

    payload = build_review_payload(candidate)

    assert payload["expanded_formula"] == "IF_POS(A, B)"
    assert payload["semantic_expanded_formula"] == "(A if A > 0 else B)"


def test_build_review_payload_includes_structure_hints_for_complex_formula():
    candidate = {
        "formula": [
            "PURE_VALUE",
            "TS_RET",
            "TS_STD20",
            "PURE_VALUE",
            "NEG",
            "ABS",
            "IV",
            "PCT_CHG_STK",
            "MIN",
            "IF_POS",
            "PURE_VALUE",
            "MIN",
            "IF_POS",
            "PURE_VALUE",
            "MIN",
        ],
        "readable": "",
    }

    payload = build_review_payload(candidate)
    hints = payload["structure_hints"]

    assert any("ABS(NEG(PURE_VALUE))" in item for item in hints)
    assert any("TS_STD20(TS_RET(PURE_VALUE))" in item for item in hints)
    assert any("PURE_VALUE 的上界截断" in item or "不高于 PURE_VALUE" in item for item in hints)


def test_build_user_prompt_reflects_custom_required_fields():
    candidate = {
        "formula": ["PURE_VALUE"],
        "readable": "PURE_VALUE",
    }
    ai_review_config = {
        "output_schema": {
            "required_fields": ["summary", "review_decision"],
        }
    }

    user_prompt = build_user_prompt(candidate, ai_review_config=ai_review_config)

    assert '"summary"' in user_prompt
    assert '"review_decision"' in user_prompt
    assert '"theme_tags"' not in user_prompt


def test_parse_review_response_rejects_missing_required_fields():
    text = """
    {
      "summary": "逻辑清晰",
      "review_decision": "keep"
    }
    """

    try:
        parse_review_response(text)
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("missing fields should fail validation")


def test_parse_review_response_rejects_invalid_score_range():
    text = """
    {
      "theme_tags": ["valuation"],
      "style_label": "defensive",
      "financial_coherence_score": 1.2,
      "interpretability_score": 0.7,
      "summary": "逻辑清晰",
      "logic_chain": ["PURE_VALUE 提供估值锚"],
      "risks": ["验证偏强"],
      "review_decision": "keep"
    }
    """

    try:
        parse_review_response(text)
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("invalid score range should fail validation")


def test_parse_review_response_normalizes_review_decision_by_rule():
    text = """
    {
      "theme_tags": ["valuation"],
      "style_label": "defensive",
      "financial_coherence_score": 0.82,
      "interpretability_score": 0.76,
      "summary": "逻辑清晰",
      "logic_chain": ["PURE_VALUE 提供估值锚"],
      "risks": ["验证窗口有限"],
      "review_decision": "drop"
    }
    """

    review = parse_review_response(text)

    assert review["review_decision"] == "keep"


def test_parse_review_response_normalizes_to_drop_when_score_too_low():
    text = """
    {
      "theme_tags": ["valuation"],
      "style_label": "defensive",
      "financial_coherence_score": 0.75,
      "interpretability_score": 0.20,
      "summary": "逻辑较弱",
      "logic_chain": ["结构复杂"],
      "risks": ["解释性差"],
      "review_decision": "keep"
    }
    """

    review = parse_review_response(text)

    assert review["review_decision"] == "drop"


def test_parse_review_response_normalizes_labels_to_canonical_enums():
    text = """
    {
      "theme_tags": ["债底保护", "多因子合成", "Momentum"],
      "style_label": "Momentum",
      "financial_coherence_score": 0.62,
      "interpretability_score": 0.51,
      "summary": "逻辑清晰",
      "logic_chain": ["PURE_VALUE 提供估值锚"],
      "risks": ["验证窗口有限"],
      "review_decision": "watch"
    }
    """

    review = parse_review_response(text)

    assert review["theme_tags"] == ["defensive", "hybrid", "momentum"]
    assert review["style_label"] == "trend"


def test_parse_review_response_softens_overconfident_language():
    text = """
    {
      "theme_tags": ["momentum"],
      "style_label": "trend",
      "financial_coherence_score": 0.35,
      "interpretability_score": 0.20,
      "summary": "该因子是一个严重过拟合的结构，最终等效于简单动量。",
      "logic_chain": ["它证明了公式必然退化为趋势信号。"],
      "risks": ["该因子严重过拟合。"],
      "review_decision": "drop"
    }
    """

    review = parse_review_response(text)

    assert "严重过拟合" not in review["summary"]
    assert "最终等效于" not in review["summary"]
    assert "证明了" not in review["logic_chain"][0]
    assert "必然" not in review["logic_chain"][0]
    assert any("存在明显过拟合风险" in item for item in review["risks"])


def test_parse_review_response_injects_formula_risk_when_risks_are_generic():
    text = """
    {
      "theme_tags": ["valuation"],
      "style_label": "valuation",
      "financial_coherence_score": 0.55,
      "interpretability_score": 0.40,
      "summary": "逻辑一般。",
      "logic_chain": ["结构较复杂。"],
      "risks": ["数据源可能存在滞后或估计误差。"],
      "review_decision": "watch"
    }
    """
    candidate = {
        "formula": ["PURE_VALUE", "IV", "PCT_CHG_STK", "MIN", "IF_POS"],
    }

    review = parse_review_response(text, candidate=candidate)

    assert any("PURE_VALUE" in item or "IV" in item or "PCT_CHG_STK" in item for item in review["risks"])


def test_parse_review_response_passes_formula_context_into_risks():
    text = """
    {
      "theme_tags": ["valuation"],
      "style_label": "valuation",
      "financial_coherence_score": 0.55,
      "interpretability_score": 0.40,
      "summary": "逻辑一般。",
      "logic_chain": ["结构较复杂。"],
      "risks": ["该因子对数据源稳定性较敏感。"],
      "review_decision": "watch"
    }
    """
    candidate = {
        "formula": ["IF_POS", "MIN", "PURE_VALUE"],
    }

    review = parse_review_response(text, candidate=candidate)

    assert review["risks"][0].startswith("公式直接依赖")
    assert "IF_POS" in review["risks"][0] or "MIN" in review["risks"][0]


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


def test_review_candidates_with_ai_dispatches_select(monkeypatch):
    candidate = {
        "formula": ["PURE_VALUE", "PREM_Z", "MIN"],
        "readable": "PURE_VALUE PREM_Z MIN",
        "selection_score": 1.23,
    }

    monkeypatch.setattr(
        "model_core.factor_ai_review.review_candidate_with_select",
        lambda candidate, model, api_key=None, base_url=None, timeout_sec=None: {
            "review_decision": "keep",
            "summary": base_url or model,
        },
    )

    reviews = review_candidates_with_ai(
        [candidate],
        provider="select",
        model="[次]gemini-3.1-pro-preview",
        base_url="https://once.novai.su/v1",
        max_candidates=1,
    )

    assert len(reviews) == 1
    assert reviews[0]["provider"] == "select"
    assert reviews[0]["model"] == "[次]gemini-3.1-pro-preview"
    assert reviews[0]["review"]["review_decision"] == "keep"
    assert reviews[0]["review"]["summary"] == "https://once.novai.su/v1"


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
