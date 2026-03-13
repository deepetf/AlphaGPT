import json
import os
from typing import Any, Dict, List, Optional

from .config import ModelConfig, RobustConfig


DEFAULT_AI_REVIEW_SCHEMA = {
    "theme_tags": [],
    "style_label": "",
    "financial_coherence_score": 0.0,
    "interpretability_score": 0.0,
    "regime_dependency_score": 0.0,
    "redundancy_risk_score": 0.0,
    "implementation_confidence": 0.0,
    "summary": "",
    "logic_chain": [],
    "risks": [],
    "review_decision": "watch",
}


def _resolve_request_timeout_sec(timeout_sec: Optional[float] = None) -> float:
    if timeout_sec is not None:
        return float(timeout_sec)
    raw = os.getenv("AI_REVIEW_TIMEOUT_SEC", "60")
    try:
        return float(raw)
    except ValueError:
        return 60.0


def _normalize_review_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    upper = message.upper()
    if "429" in upper or "RESOURCE_EXHAUSTED" in upper or "QUOTA" in upper:
        return "AI review skipped: provider quota exhausted"
    if (
        "API KEY" in upper
        or "OPENAI_API_KEY" in upper
        or "GEMINI_API_KEY" in upper
        or "MODELSCOPE_API_KEY" in upper
        or "GLM5_API_KEY" in upper
    ):
        return "AI review skipped: missing API key"
    if "JSON" in upper:
        return "AI review skipped: invalid JSON response"
    if message:
        return f"AI review skipped: {message}"
    return "AI review skipped: unknown provider error"


def _build_fallback_review(exc: Exception) -> Dict[str, Any]:
    review = dict(DEFAULT_AI_REVIEW_SCHEMA)
    error_message = _normalize_review_error(exc)
    review["summary"] = error_message
    review["risks"] = [error_message]
    review["review_decision"] = "watch"
    return review


def build_system_prompt() -> str:
    return (
        "你是可转债量化研究员。你的任务不是评价语言是否优美，而是判断一个因子是否具有"
        "清晰、可复核、具备金融逻辑的经济含义。\n"
        "请同时结合公式结构与量化指标进行判断，不要只看公式臆测。\n"
        "要求：\n"
        "1. 输出必须是 JSON。\n"
        "2. 不要输出 markdown。\n"
        "3. 若验证集显著强于训练集，应提高 regime_dependency_score。\n"
        "4. 若公式结构复杂但实质冗余，应提高 redundancy_risk_score。\n"
        "5. 若因子主要围绕估值、溢价、流动性、正股联动、波动过滤等清晰主题展开，应提高 financial_coherence_score。\n"
        "6. 若公式过多依赖 IF_POS/MAX/MIN 等门控拼接且逻辑难解释，应降低 interpretability_score。\n"
        "7. 所有分数使用 0~1，小数即可。"
    )


def build_review_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "formula": candidate.get("formula", []),
        "readable": candidate.get("readable", ""),
        "metrics": {
            "selection_score": candidate.get("selection_score"),
            "sharpe_all": candidate.get("sharpe_all"),
            "sharpe_train": candidate.get("sharpe_train"),
            "sharpe_val": candidate.get("sharpe_val"),
            "balanced_sharpe": candidate.get("balanced_sharpe"),
            "train_val_gap": candidate.get("train_val_gap"),
            "stability": candidate.get("stability"),
            "annualized_ret": candidate.get("annualized_ret"),
            "max_drawdown": candidate.get("max_drawdown"),
            "active_ratio": candidate.get("active_ratio"),
            "valid_days_train": candidate.get("valid_days_train"),
            "valid_days_val": candidate.get("valid_days_val"),
        },
        "context": {
            "asset_type": "convertible_bond",
            "top_k": RobustConfig.TOP_K,
            "split_date": RobustConfig.TRAIN_TEST_SPLIT_DATE,
            "take_profit": RobustConfig.TAKE_PROFIT,
            "input_features": list(ModelConfig.INPUT_FEATURES),
        },
    }


def build_user_prompt(candidate: Dict[str, Any]) -> str:
    payload = build_review_payload(candidate)
    return (
        "请评估以下可转债因子，并返回 JSON。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_review_response(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty AI review response")

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("AI review response does not contain JSON object")

    obj = json.loads(raw[start : end + 1])
    normalized = dict(DEFAULT_AI_REVIEW_SCHEMA)
    normalized.update(obj)
    return normalized


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                value = getattr(part, "text", None)
                if isinstance(value, str):
                    chunks.append(value)
        if chunks:
            return "\n".join(chunks)
    return ""


def _extract_gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        chunks: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not isinstance(parts, list):
                continue
            for part in parts:
                value = getattr(part, "text", None)
                if isinstance(value, str):
                    chunks.append(value)
        if chunks:
            return "\n".join(chunks)
    return ""


def review_candidate_with_openai(
    candidate: Dict[str, Any],
    *,
    model: str,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "AI review requires the 'openai' package. Install it and set OPENAI_API_KEY."
        ) from exc

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AI review requires OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, timeout=_resolve_request_timeout_sec(timeout_sec))
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(candidate)

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return parse_review_response(_extract_response_text(response))

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    text = completion.choices[0].message.content if completion.choices else ""
    return parse_review_response(text)


def review_candidate_with_gemini(
    candidate: Dict[str, Any],
    *,
    model: str,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("AI review requires GEMINI_API_KEY")

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(candidate)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        from google import genai  # type: ignore
    except Exception:
        genai = None

    if genai is not None:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
        )
        return parse_review_response(_extract_gemini_response_text(response))

    try:
        import google.generativeai as legacy_genai  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "AI review with Gemini requires 'google-genai' or 'google-generativeai', and GEMINI_API_KEY."
        ) from exc

    legacy_genai.configure(api_key=api_key)
    gemini_model = legacy_genai.GenerativeModel(model)
    response = gemini_model.generate_content(full_prompt)
    text = getattr(response, "text", "")
    return parse_review_response(text)


def review_candidate_with_glm5(
    candidate: Dict[str, Any],
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "AI review with GLM-5 requires the 'openai' package, and MODELSCOPE_API_KEY or GLM5_API_KEY."
        ) from exc

    api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("GLM5_API_KEY")
    if not api_key:
        raise RuntimeError("AI review requires MODELSCOPE_API_KEY or GLM5_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url or os.getenv("MODELSCOPE_BASE_URL") or "https://api-inference.modelscope.cn/v1",
        timeout=_resolve_request_timeout_sec(timeout_sec),
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(candidate)},
        ],
        response_format={"type": "json_object"},
    )
    text = completion.choices[0].message.content if completion.choices else ""
    return parse_review_response(text)


def review_candidate_with_provider(
    candidate: Dict[str, Any],
    *,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    provider_norm = str(provider or "openai").strip().lower()
    if provider_norm == "openai":
        return review_candidate_with_openai(
            candidate, model=model, api_key=api_key, timeout_sec=timeout_sec
        )
    if provider_norm == "gemini":
        return review_candidate_with_gemini(
            candidate, model=model, api_key=api_key, timeout_sec=timeout_sec
        )
    if provider_norm == "glm5":
        return review_candidate_with_glm5(
            candidate, model=model, api_key=api_key, timeout_sec=timeout_sec
        )
    raise ValueError(f"Unsupported AI review provider: {provider}")


def review_candidates_with_ai(
    candidates: List[Dict[str, Any]],
    *,
    provider: str,
    model: str,
    max_candidates: int,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    provider_norm = str(provider or "openai").strip().lower()
    if provider_norm not in {"openai", "gemini", "glm5"}:
        raise ValueError(f"Unsupported AI review provider: {provider}")

    reviews: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates[:max_candidates], start=1):
        print(
            f"[AI Review] {provider_norm} {idx}/{min(len(candidates), max_candidates)} "
            f"| score={float(candidate.get('selection_score') or 0.0):.4f}"
        )
        error_message: Optional[str] = None
        try:
            review = review_candidate_with_provider(
                candidate,
                provider=provider_norm,
                model=model,
                api_key=api_key,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            error_message = _normalize_review_error(exc)
            review = _build_fallback_review(exc)
            print(f"[AI Review] fallback: {error_message}")
        reviews.append(
            {
                "rank_hint": idx,
                "provider": provider_norm,
                "model": model,
                "formula": candidate.get("formula"),
                "readable": candidate.get("readable"),
                "selection_score": candidate.get("selection_score"),
                "review": review,
                "error": error_message,
            }
        )
    return reviews


def render_markdown_report(
    selection_result: Dict[str, Any],
    ai_reviews: Optional[List[Dict[str, Any]]] = None,
) -> str:
    lines: List[str] = []
    lines.append("# Top Factors Report")
    lines.append("")
    counts = selection_result.get("counts", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Raw candidates: {counts.get('raw_candidates', 0)}")
    lines.append(f"- Deduped candidates: {counts.get('deduped_candidates', 0)}")
    lines.append(f"- Hard-filter passed: {counts.get('hard_filter_passed', 0)}")
    lines.append(f"- Similarity rejected: {counts.get('similarity_rejected', 0)}")
    lines.append(f"- Selected top-k: {counts.get('selected_top_k', 0)}")
    lines.append("")
    lines.append("## Top 3")
    lines.append("")
    lines.append("| Rank | Score | Sharpe A/T/V | Stability | Annualized Return | MDD | Formula |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | --- |")
    for idx, candidate in enumerate(selection_result.get("selected", []), start=1):
        lines.append(
            f"| {idx} | {candidate.get('selection_score', 0):.4f} | "
            f"{candidate.get('sharpe_all', 0):.2f}/"
            f"{candidate.get('sharpe_train', 0):.2f}/"
            f"{candidate.get('sharpe_val', 0):.2f} | "
            f"{candidate.get('stability', 0):.3f} | "
            f"{candidate.get('annualized_ret', 0):.2%} | "
            f"{candidate.get('max_drawdown', 0):.1%} | "
            f"{candidate.get('readable', '')} |"
        )

    if ai_reviews:
        lines.append("")
        lines.append("## AI Reviews")
        lines.append("")
        for idx, item in enumerate(ai_reviews, start=1):
            review = item.get("review", {})
            lines.append(f"### Candidate {idx}")
            lines.append("")
            lines.append(f"- Formula: `{item.get('readable', '')}`")
            lines.append(f"- Theme Tags: {', '.join(review.get('theme_tags', [])) or 'N/A'}")
            lines.append(f"- Style Label: {review.get('style_label', 'N/A')}")
            lines.append(
                f"- Scores: coherence={review.get('financial_coherence_score', 0):.2f}, "
                f"interpretability={review.get('interpretability_score', 0):.2f}, "
                f"regime_dependency={review.get('regime_dependency_score', 0):.2f}, "
                f"redundancy={review.get('redundancy_risk_score', 0):.2f}"
            )
            lines.append(f"- Decision: {review.get('review_decision', 'watch')}")
            if item.get("error"):
                lines.append(f"- Error: {item.get('error')}")
            lines.append(f"- Summary: {review.get('summary', '')}")
            logic_chain = review.get("logic_chain", [])
            if logic_chain:
                lines.append("- Logic Chain:")
                for point in logic_chain:
                    lines.append(f"  - {point}")
            risks = review.get("risks", [])
            if risks:
                lines.append("- Risks:")
                for point in risks:
                    lines.append(f"  - {point}")
            lines.append("")
    else:
        lines.append("")
        lines.append("## AI Reviews")
        lines.append("")
        lines.append("AI review not enabled in this run.")
        lines.append("")

    return "\n".join(lines)
