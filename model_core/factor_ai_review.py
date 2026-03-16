import json
import os
from typing import Any, Dict, List, Optional

from .config import ModelConfig, RobustConfig
from .features_registry import get_feature_spec
from .formula_simplifier import (
    collect_structure_hints,
    expand_formula,
    expand_formula_semantic,
)
from .ops_registry import OpsRegistry


DEFAULT_SYSTEM_PROMPT = (
    "你是可转债量化研究员。你的任务不是评价语言是否优美，而是判断一个因子是否具有"
    "清晰、可复核、具备金融逻辑的经济含义。\n"
    "该因子公式是逆波兰表达式（RPN, Reverse Polish Notation），必须按栈式执行顺序理解，"
    "不能按普通中缀表达式或自然语言短语拼接理解。\n"
    "请同时结合公式结构与量化指标进行判断，不要只看公式臆测。\n"
    "要求：\n"
    "1. 若 payload 提供 structure_hints，必须优先参考这些程序化结构事实；其次基于 semantic_expanded_formula 理解结构，再参考 expanded_formula；原始 formula 仅作为 RPN 参考。\n"
    "2. 你的第一步必须是按 RPN 栈式顺序正确展开公式，再基于展开后的结构讨论金融含义。\n"
    "3. 必须在内部先明确公式的数学结构；若无法可靠展开，就降低 interpretability_score，并在 risks 中说明“RPN 结构复杂导致解释不稳定”。\n"
    "4. 不要把相邻 token 直接当作并列语义特征；必须优先服从 RPN 执行顺序。\n"
    "5. 不得把 IF_POS 当成普通三元条件表达式；必须按 payload 中 operator_metadata 给出的项目内定义理解。\n"
    "6. 若未完成可靠的逐步展开，不得声称公式“等效于”“退化为”某个更简单表达式；最多只能写成待数值验证的风险提示。\n"
    "7. 输出必须是 JSON。\n"
    "8. 不要输出 markdown。\n"
    "9. 必须完整返回 schema 中定义的所有字段。\n"
    "10. financial_coherence_score 用于评估该因子是否围绕清晰的可转债主题展开，例如估值、溢价、流动性、正股联动、波动过滤。\n"
    "11. interpretability_score 用于评估公式是否容易解释、是否存在明显的门控拼接或无经济意义的复杂组合。\n"
    "12. theme_tags 返回 1~3 个简短主题标签。\n"
    "13. style_label 返回一个简短风格标签，例如 valuation、defensive、trend、liquidity。\n"
    "14. summary 用 1~2 句话概括核心逻辑。\n"
    "15. logic_chain 返回 1~3 条简短因果链。\n"
    "16. risks 返回 1~3 条主要风险或局限。\n"
    "17. review_decision 只能取 keep、watch、drop 之一，它只表示研究复核优先级，不代表最终量化入选结果。\n"
    "18. review_decision 必须严格按以下规则返回：若 financial_coherence_score >= 0.70 且 interpretability_score >= 0.60 且 risks 条数 <= 2，则返回 keep；若 financial_coherence_score < 0.40 或 interpretability_score < 0.30，则返回 drop；其余情况一律返回 watch。\n"
    "19. theme_tags 只能从以下枚举中选择：valuation、premium、equity_linkage、volatility、liquidity、momentum、defensive、hybrid。\n"
    "20. style_label 只能从以下枚举中选择：valuation、defensive、trend、mean_reversion、liquidity、hybrid。\n"
    "21. 不得使用“最终等效于”“必然”“证明了”“严重过拟合”这类过强结论；若只是基于公式结构和回测指标推断，必须使用“可能”“倾向于”“显示出”“存在...风险”等审慎表述。\n"
    "22. 不得声称完成了严格数学化简、因果证明或样本外验证；若无法由输入直接推出，只能写成风险提示或工作假设。\n"
    "23. risks 必须优先写以下三类：公式结构风险、经济解释风险、样本稳定性风险；只有在公式明显依赖某个输入字段时，才允许写数据质量风险。\n"
    "24. risks 至少有 1 条必须直接点名当前公式中的算子或字段，例如 IF_POS、MIN、MAX、PURE_VALUE、IV、PCT_CHG_STK。\n"
    "25. 评审时必须优先参考 payload 中提供的 feature_metadata、operator_metadata 与 structure_hints，不要自行臆测字段含义、算子语义或分支激活特性。\n"
    "26. 若多个字段量纲不同，必须提醒量纲不可直接类比；若字段本身是 zscore-like 或训练管线会额外做 time-z，也应在解释中区分说明。\n"
    "27. 对于 structure_hints 中已经明确给出的代数化简、非负性或分支触发条件，不要再给出与其矛盾的解释。\n"
    "28. 所有分数使用 0~1 小数。"
)

DEFAULT_USER_PROMPT_TEMPLATE = (
    "请评估以下可转债因子，并返回 JSON。\n\n"
    "输出 schema:\n"
    "{schema_json}\n\n"
    "候选因子数据:\n"
    "{payload_json}"
)

DEFAULT_AI_REVIEW_SCHEMA = {
    "theme_tags": [],
    "style_label": "",
    "financial_coherence_score": 0.0,
    "interpretability_score": 0.0,
    "summary": "",
    "logic_chain": [],
    "risks": [],
    "review_decision": "watch",
}

FEATURE_MEANING_HINTS: Dict[str, str] = {
    "CLOSE": "可转债收盘价，价格水平类特征。",
    "VOL": "可转债成交量，成交活跃度特征。",
    "PREM": "转股溢价率，反映转债相对正股转换价值的偏离。",
    "DBLOW": "双低指标，综合价格与溢价的可转债选债指标。",
    "REMAIN_SIZE": "剩余规模，反映转债存量盘子大小。",
    "PCT_CHG": "可转债单日涨跌幅，收益率类特征。",
    "PCT_CHG_5": "可转债5日涨跌幅，短周期收益率类特征。",
    "VOLATILITY_STK": "正股波动率，波动属性特征。",
    "PCT_CHG_STK": "正股单日涨跌幅，正股联动收益率特征。",
    "PCT_CHG_5_STK": "正股5日涨跌幅，短周期正股联动收益率特征。",
    "CLOSE_STK": "正股收盘价，价格水平类特征。",
    "CONV_PRICE": "转股价，用于衡量转股价值与 moneyness。",
    "PURE_VALUE": "纯债价值，常作为债底或防御性估值锚。",
    "ALPHA_PCT_CHG_5": "转债相对正股的5日相对强弱特征。",
    "CAP_MV_RATE": "转债市值占比，规模结构特征。",
    "TURNOVER": "换手率，流动性与交易活跃度特征。",
    "IV": "隐含波动率，期权属性定价特征。",
    "VOL_STK_60": "正股60日波动率，中期波动特征。",
    "PREM_Z": "转股溢价率的标准化分数特征。",
    "LEFT_YRS": "剩余年限，期限结构特征。",
    "LOG_MONEYNESS": "log(S/K)，反映正股价格相对转股价的 moneyness。",
    "PURE_VALUE_CS_RANK": "PURE_VALUE 的横截面排名特征。",
    "PURE_VALUE_CS_ROBUST_Z": "PURE_VALUE 的横截面 robust z-score 特征。",
    "PREM_CS_RANK": "PREM 的横截面排名特征。",
    "PREM_CS_ROBUST_Z": "PREM 的横截面 robust z-score 特征。",
    "REMAIN_SIZE_CS_RANK": "REMAIN_SIZE 的横截面排名特征。",
    "CAP_MV_RATE_CS_RANK": "CAP_MV_RATE 的横截面排名特征。",
    "DBLOW_CS_RANK": "DBLOW 的横截面排名特征。",
    "DBLOW_CS_ROBUST_Z": "DBLOW 的横截面 robust z-score 特征。",
}

OPERATOR_MEANING_HINTS: Dict[str, str] = {
    "ADD": "二元加法。",
    "SUB": "二元减法。",
    "MUL": "二元乘法。",
    "DIV": "安全除法，分母过小时会做保护。",
    "NEG": "取相反数。",
    "ABS": "取绝对值。",
    "LOG": "安全对数。",
    "SQRT": "安全平方根。",
    "SIGN": "取符号。",
    "TS_DELAY": "滞后1期。",
    "TS_DELTA": "一阶差分。",
    "TS_RET": "单步收益率。",
    "TS_MOM10": "10期动量差分。",
    "TS_MOM20": "20期动量差分。",
    "TS_MEAN5": "5期滚动均值。",
    "TS_STD5": "5期滚动标准差。",
    "TS_STD20": "20期滚动标准差。",
    "TS_STD60": "60期滚动标准差。",
    "TS_MAX20": "20期滚动最大值。",
    "TS_MIN20": "20期滚动最小值。",
    "TS_BIAS5": "相对5期均线偏离率。",
    "CS_RANK": "横截面排名，通常压缩到 0~1。",
    "CS_DEMEAN": "横截面去均值。",
    "CS_ROBUST_Z": "横截面 robust z-score。",
    "MAX": "逐元素取较大值。",
    "MIN": "逐元素取较小值。",
    "IF_POS": "若第一个输入大于0则返回第一个输入，否则返回第二个输入。",
    "CUT_NEG": "将负值截断为0。",
    "CUT_HIGH": "对过高值做强惩罚截断。",
}

DEFAULT_AI_REVIEW_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "provider": "gemini",
    "model": "gemini-2.0-flash",
    "base_url": None,
    "api_key_env": None,
    "max_candidates": 10,
    "prompt": {
        "system": DEFAULT_SYSTEM_PROMPT,
        "user_template": DEFAULT_USER_PROMPT_TEMPLATE,
    },
    "output_schema": {
        "required_fields": list(DEFAULT_AI_REVIEW_SCHEMA.keys()),
        "decision_values": ["keep", "watch", "drop"],
        "score_fields": [
            "financial_coherence_score",
            "interpretability_score",
        ],
    },
    "decision_rule": {
        "keep": {
            "min_financial_coherence_score": 0.70,
            "min_interpretability_score": 0.60,
            "max_risks": 2,
        },
        "drop": {
            "max_financial_coherence_score": 0.40,
            "max_interpretability_score": 0.30,
        },
        "fallback": "watch",
    },
    "label_constraints": {
        "allowed_theme_tags": [
            "valuation",
            "premium",
            "equity_linkage",
            "volatility",
            "liquidity",
            "momentum",
            "defensive",
            "hybrid",
        ],
        "allowed_style_labels": [
            "valuation",
            "defensive",
            "trend",
            "mean_reversion",
            "liquidity",
            "hybrid",
        ],
        "max_theme_tags": 3,
    },
    "risk_constraints": {
        "max_risks": 3,
        "required_formula_reference": True,
        "generic_risk_markers": [
            "数据源",
            "滞后",
            "估计误差",
            "准确性",
            "稳定性",
        ],
    },
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
        or "NOVAI_API_KEY" in upper
        or "OPENAI_API_KEY" in upper
        or "AI_REVIEW_API_KEY" in upper
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


def _resolve_prompt_config(ai_review_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prompt = {}
    if isinstance(ai_review_config, dict):
        prompt = ai_review_config.get("prompt", {}) or {}
    return prompt if isinstance(prompt, dict) else {}


def _resolve_config_value(
    ai_review_config: Optional[Dict[str, Any]],
    key: str,
) -> Optional[str]:
    if not isinstance(ai_review_config, dict):
        return None
    value = ai_review_config.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _resolve_env_value(*env_names: str) -> Optional[str]:
    for env_name in env_names:
        value = os.getenv(env_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_output_schema_config(ai_review_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    schema = {}
    if isinstance(ai_review_config, dict):
        schema = ai_review_config.get("output_schema", {}) or {}
    if not isinstance(schema, dict):
        schema = {}
    return {
        "required_fields": list(
            schema.get("required_fields")
            or DEFAULT_AI_REVIEW_CONFIG["output_schema"]["required_fields"]
        ),
        "decision_values": list(
            schema.get("decision_values")
            or DEFAULT_AI_REVIEW_CONFIG["output_schema"]["decision_values"]
        ),
        "score_fields": list(
            schema.get("score_fields")
            or DEFAULT_AI_REVIEW_CONFIG["output_schema"]["score_fields"]
        ),
    }


def _resolve_decision_rule_config(ai_review_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rule = {}
    if isinstance(ai_review_config, dict):
        rule = ai_review_config.get("decision_rule", {}) or {}
    if not isinstance(rule, dict):
        rule = {}
    default_rule = DEFAULT_AI_REVIEW_CONFIG["decision_rule"]
    keep_rule = rule.get("keep", {}) if isinstance(rule.get("keep", {}), dict) else {}
    drop_rule = rule.get("drop", {}) if isinstance(rule.get("drop", {}), dict) else {}
    return {
        "keep": {
            "min_financial_coherence_score": float(
                keep_rule.get(
                    "min_financial_coherence_score",
                    default_rule["keep"]["min_financial_coherence_score"],
                )
            ),
            "min_interpretability_score": float(
                keep_rule.get(
                    "min_interpretability_score",
                    default_rule["keep"]["min_interpretability_score"],
                )
            ),
            "max_risks": int(keep_rule.get("max_risks", default_rule["keep"]["max_risks"])),
        },
        "drop": {
            "max_financial_coherence_score": float(
                drop_rule.get(
                    "max_financial_coherence_score",
                    default_rule["drop"]["max_financial_coherence_score"],
                )
            ),
            "max_interpretability_score": float(
                drop_rule.get(
                    "max_interpretability_score",
                    default_rule["drop"]["max_interpretability_score"],
                )
            ),
        },
        "fallback": str(rule.get("fallback", default_rule["fallback"])),
    }


def _resolve_label_constraints_config(
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    constraints = {}
    if isinstance(ai_review_config, dict):
        constraints = ai_review_config.get("label_constraints", {}) or {}
    if not isinstance(constraints, dict):
        constraints = {}
    default_constraints = DEFAULT_AI_REVIEW_CONFIG["label_constraints"]
    return {
        "allowed_theme_tags": list(
            constraints.get("allowed_theme_tags")
            or default_constraints["allowed_theme_tags"]
        ),
        "allowed_style_labels": list(
            constraints.get("allowed_style_labels")
            or default_constraints["allowed_style_labels"]
        ),
        "max_theme_tags": int(
            constraints.get("max_theme_tags", default_constraints["max_theme_tags"])
        ),
    }


def _resolve_risk_constraints_config(
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    constraints = {}
    if isinstance(ai_review_config, dict):
        constraints = ai_review_config.get("risk_constraints", {}) or {}
    if not isinstance(constraints, dict):
        constraints = {}
    default_constraints = DEFAULT_AI_REVIEW_CONFIG["risk_constraints"]
    return {
        "max_risks": int(constraints.get("max_risks", default_constraints["max_risks"])),
        "required_formula_reference": bool(
            constraints.get(
                "required_formula_reference",
                default_constraints["required_formula_reference"],
            )
        ),
        "generic_risk_markers": list(
            constraints.get("generic_risk_markers")
            or default_constraints["generic_risk_markers"]
        ),
    }


def build_system_prompt(ai_review_config: Optional[Dict[str, Any]] = None) -> str:
    prompt_config = _resolve_prompt_config(ai_review_config)
    system_prompt = prompt_config.get("system")
    if isinstance(system_prompt, str) and system_prompt.strip():
        return system_prompt
    return DEFAULT_SYSTEM_PROMPT


def _infer_dimension_hint(feature_name: str) -> str:
    name = str(feature_name or "").upper()
    if name.endswith("_CS_RANK") or name == "CS_RANK":
        return "cross_sectional_rank_0_1"
    if name.endswith("_CS_ROBUST_Z") or name.endswith("_Z") or "ROBUST_Z" in name:
        return "zscore_like"
    if "PCT_CHG" in name or "RET" in name or "BIAS" in name or "ALPHA" in name:
        return "return_ratio"
    if "VOLATILITY" in name or name.startswith("IV") or "STD" in name or "VOL_" in name:
        return "volatility_scale"
    if "TURNOVER" in name:
        return "turnover_ratio"
    if "RATE" in name or "PREM" in name:
        return "ratio"
    if "LEFT_YRS" in name:
        return "years"
    if "SIZE" in name or "CAP_MV" in name:
        return "size_or_scale"
    return "level"


def _infer_native_zscore(feature_name: str) -> bool:
    name = str(feature_name or "").upper()
    return name.endswith("_Z") or "ROBUST_Z" in name


def _infer_pipeline_time_normalization(feature_name: str) -> Optional[bool]:
    spec = get_feature_spec(str(feature_name or ""))
    if spec is None:
        return None
    return bool(spec.apply_time_normalization)


def _build_feature_metadata(formula: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(formula, list):
        return {}
    metadata: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for token in formula:
        if not isinstance(token, str) or token in seen:
            continue
        spec = get_feature_spec(token)
        if spec is None:
            continue
        seen.add(token)
        metadata[token] = {
            "meaning": FEATURE_MEANING_HINTS.get(
                token,
                f"{token} 特征，需结合项目注册定义理解其金融含义。",
            ),
            "kind": spec.kind,
            "raw_column": spec.raw_column,
            "deps": list(spec.deps),
            "dimension_hint": _infer_dimension_hint(token),
            "native_zscore_like": _infer_native_zscore(token),
            "pipeline_time_normalized": _infer_pipeline_time_normalization(token),
            "review_hint": (
                "比较该特征与其他 token 的组合时，应注意量纲是否一致，并区分"
                "“字段本身已标准化”与“训练管线额外做过 time-z”。"
            ),
        }
    return metadata


def _build_operator_metadata(formula: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(formula, list):
        return {}
    metadata: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for token in formula:
        if not isinstance(token, str) or token in seen:
            continue
        op_info = OpsRegistry.get_op(token)
        if op_info is None:
            continue
        seen.add(token)
        metadata[token] = {
            "arity": op_info.get("arity"),
            "registry_description": op_info.get("description", ""),
            "meaning": OPERATOR_MEANING_HINTS.get(
                token,
                f"{token} 算子，语义以项目 VM/registry 实现为准。",
            ),
        }
    return metadata


def _build_expanded_formula(formula: Any) -> str:
    if not isinstance(formula, list) or not formula:
        return ""
    try:
        return expand_formula([str(item) for item in formula if isinstance(item, str)])
    except Exception:
        return ""


def _build_semantic_expanded_formula(formula: Any) -> str:
    if not isinstance(formula, list) or not formula:
        return ""
    try:
        return expand_formula_semantic([str(item) for item in formula if isinstance(item, str)])
    except Exception:
        return ""


def _build_structure_hints(formula: Any) -> List[str]:
    if not isinstance(formula, list) or not formula:
        return []
    try:
        return collect_structure_hints([str(item) for item in formula if isinstance(item, str)])
    except Exception:
        return []


def build_review_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    formula = candidate.get("formula", [])
    return {
        "formula": formula,
        "expanded_formula": _build_expanded_formula(formula),
        "semantic_expanded_formula": _build_semantic_expanded_formula(formula),
        "structure_hints": _build_structure_hints(formula),
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
        "feature_metadata": _build_feature_metadata(formula),
        "operator_metadata": _build_operator_metadata(formula),
        "context": {
            "asset_type": "convertible_bond",
            "top_k": RobustConfig.TOP_K,
            "split_date": RobustConfig.TRAIN_TEST_SPLIT_DATE,
            "take_profit": RobustConfig.TAKE_PROFIT,
            "input_features": list(ModelConfig.INPUT_FEATURES),
        },
    }


def _build_prompt_schema(ai_review_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    schema_config = _resolve_output_schema_config(ai_review_config)
    schema: Dict[str, Any] = {}
    for field in schema_config["required_fields"]:
        schema[field] = DEFAULT_AI_REVIEW_SCHEMA.get(field, "")
    return schema


def build_user_prompt(
    candidate: Dict[str, Any],
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> str:
    payload = build_review_payload(candidate)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    schema_json = json.dumps(
        _build_prompt_schema(ai_review_config), ensure_ascii=False, indent=2
    )
    prompt_config = _resolve_prompt_config(ai_review_config)
    template = prompt_config.get("user_template")
    if not isinstance(template, str) or not template.strip():
        template = DEFAULT_USER_PROMPT_TEMPLATE
    return (
        template.replace("{payload_json}", payload_json).replace("{schema_json}", schema_json)
    )


def _validate_review_response(
    raw_obj: Dict[str, Any],
    normalized: Dict[str, Any],
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> None:
    schema_config = _resolve_output_schema_config(ai_review_config)
    required_fields = schema_config["required_fields"]
    missing_fields = [field for field in required_fields if field not in raw_obj]
    if missing_fields:
        raise ValueError(
            "AI review response missing required fields: " + ", ".join(missing_fields)
        )

    score_fields = schema_config["score_fields"]
    for field in score_fields:
        value = normalized.get(field)
        if not isinstance(value, (int, float)):
            raise ValueError(f"AI review field '{field}' must be numeric")
        if float(value) < 0.0 or float(value) > 1.0:
            raise ValueError(f"AI review field '{field}' must be between 0 and 1")

    for field in ("theme_tags", "logic_chain", "risks"):
        value = normalized.get(field)
        if not isinstance(value, list):
            raise ValueError(f"AI review field '{field}' must be a list")
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"AI review field '{field}' must contain only strings")

    for field in ("style_label", "summary", "review_decision"):
        value = normalized.get(field)
        if not isinstance(value, str):
            raise ValueError(f"AI review field '{field}' must be a string")

    decision_values = {str(item) for item in schema_config["decision_values"]}
    if normalized["review_decision"] not in decision_values:
        raise ValueError(
            "AI review field 'review_decision' must be one of: "
            + ", ".join(sorted(decision_values))
        )


def _derive_review_decision(
    normalized: Dict[str, Any],
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> str:
    decision_rule = _resolve_decision_rule_config(ai_review_config)
    coherence = float(normalized.get("financial_coherence_score", 0.0))
    interpretability = float(normalized.get("interpretability_score", 0.0))
    risks = normalized.get("risks", [])
    risk_count = len(risks) if isinstance(risks, list) else 0

    keep_rule = decision_rule["keep"]
    if (
        coherence >= keep_rule["min_financial_coherence_score"]
        and interpretability >= keep_rule["min_interpretability_score"]
        and risk_count <= keep_rule["max_risks"]
    ):
        return "keep"

    drop_rule = decision_rule["drop"]
    if (
        coherence < drop_rule["max_financial_coherence_score"]
        or interpretability < drop_rule["max_interpretability_score"]
    ):
        return "drop"

    return decision_rule["fallback"]


def _canonicalize_theme_tag(tag: str) -> Optional[str]:
    mapping = {
        "valuation": "valuation",
        "估值": "valuation",
        "premium": "premium",
        "溢价": "premium",
        "equity_linkage": "equity_linkage",
        "stock_linkage": "equity_linkage",
        "正股联动": "equity_linkage",
        "volatility": "volatility",
        "波动": "volatility",
        "波动率": "volatility",
        "波动率过滤": "volatility",
        "liquidity": "liquidity",
        "流动性": "liquidity",
        "momentum": "momentum",
        "动量": "momentum",
        "trend": "momentum",
        "defensive": "defensive",
        "防御": "defensive",
        "债底保护": "defensive",
        "hybrid": "hybrid",
        "多因子": "hybrid",
        "多因子合成": "hybrid",
        "multi_factor": "hybrid",
    }
    return mapping.get(str(tag).strip().lower()) or mapping.get(str(tag).strip())


def _canonicalize_style_label(label: str) -> str:
    mapping = {
        "valuation": "valuation",
        "估值": "valuation",
        "defensive": "defensive",
        "防御": "defensive",
        "trend": "trend",
        "momentum": "trend",
        "mean_reversion": "mean_reversion",
        "reversal": "mean_reversion",
        "反转": "mean_reversion",
        "liquidity": "liquidity",
        "流动性": "liquidity",
        "hybrid": "hybrid",
        "多因子": "hybrid",
        "multi_factor": "hybrid",
    }
    raw = str(label).strip()
    return mapping.get(raw.lower()) or mapping.get(raw) or "hybrid"


def _soften_assertive_language(text: str) -> str:
    softened = str(text or "")
    replacements = {
        "最终等效于": "可能主要体现为",
        "等效于": "可能接近于",
        "必然": "可能",
        "证明了": "提示",
        "证明": "提示",
        "严重过拟合": "存在明显过拟合风险",
    }
    for old, new in replacements.items():
        softened = softened.replace(old, new)
    return softened


def _collect_formula_reference_tokens(candidate: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(candidate, dict):
        return []
    formula = candidate.get("formula", [])
    if not isinstance(formula, list):
        return []
    preferred = [
        "IF_POS",
        "MIN",
        "MAX",
        "PURE_VALUE",
        "IV",
        "PCT_CHG_STK",
        "TS_RET",
        "TS_STD20",
    ]
    formula_tokens = [str(item).strip() for item in formula if isinstance(item, str)]
    selected = [token for token in preferred if token in formula_tokens]
    if selected:
        return selected[:3]
    deduped: List[str] = []
    for token in formula_tokens:
        if token and token not in deduped:
            deduped.append(token)
        if len(deduped) >= 3:
            break
    return deduped


def _risk_mentions_formula(risk: str, reference_tokens: List[str]) -> bool:
    upper_risk = str(risk or "").upper()
    return any(token.upper() in upper_risk for token in reference_tokens)


def _build_formula_structure_risk(candidate: Optional[Dict[str, Any]]) -> str:
    reference_tokens = _collect_formula_reference_tokens(candidate)
    if not reference_tokens:
        return "公式结构较复杂，经济含义复核成本较高。"
    joined = "、".join(reference_tokens)
    return f"公式直接依赖 {joined} 等字段或算子，结构复核与经济解释成本较高。"


def _normalize_review_output(
    normalized: Dict[str, Any],
    ai_review_config: Optional[Dict[str, Any]] = None,
    candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    constraints = _resolve_label_constraints_config(ai_review_config)
    allowed_theme_tags = set(constraints["allowed_theme_tags"])
    max_theme_tags = max(1, int(constraints["max_theme_tags"]))
    theme_tags: List[str] = []
    for item in normalized.get("theme_tags", []):
        canonical = _canonicalize_theme_tag(item)
        if canonical and canonical in allowed_theme_tags and canonical not in theme_tags:
            theme_tags.append(canonical)
        if len(theme_tags) >= max_theme_tags:
            break
    normalized["theme_tags"] = theme_tags

    style_label = _canonicalize_style_label(normalized.get("style_label", ""))
    allowed_style_labels = set(constraints["allowed_style_labels"])
    normalized["style_label"] = style_label if style_label in allowed_style_labels else "hybrid"

    normalized["summary"] = _soften_assertive_language(normalized.get("summary", ""))
    normalized["logic_chain"] = [
        _soften_assertive_language(item) for item in normalized.get("logic_chain", [])[:3]
    ]
    risk_constraints = _resolve_risk_constraints_config(ai_review_config)
    max_risks = max(1, int(risk_constraints["max_risks"]))
    reference_tokens = _collect_formula_reference_tokens(candidate)
    generic_markers = [str(item) for item in risk_constraints["generic_risk_markers"]]
    risks: List[str] = []
    for raw_risk in normalized.get("risks", []):
        risk = _soften_assertive_language(raw_risk)
        if not risk:
            continue
        if risk in risks:
            continue
        risks.append(risk)
        if len(risks) >= max_risks:
            break
    has_formula_risk = any(_risk_mentions_formula(risk, reference_tokens) for risk in risks)
    has_only_generic_risk = bool(risks) and all(
        any(marker in risk for marker in generic_markers) for risk in risks
    )
    if risk_constraints["required_formula_reference"] and (
        not has_formula_risk or has_only_generic_risk
    ):
        structure_risk = _build_formula_structure_risk(candidate)
        risks = [structure_risk] + [risk for risk in risks if risk != structure_risk]
    normalized["risks"] = risks[:max_risks]
    return normalized


def parse_review_response(
    text: str,
    ai_review_config: Optional[Dict[str, Any]] = None,
    candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty AI review response")

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("AI review response does not contain JSON object")

    obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("AI review response JSON must be an object")
    normalized = dict(DEFAULT_AI_REVIEW_SCHEMA)
    normalized.update(obj)
    _validate_review_response(obj, normalized, ai_review_config=ai_review_config)
    normalized = _normalize_review_output(
        normalized,
        ai_review_config=ai_review_config,
        candidate=candidate,
    )
    normalized["review_decision"] = _derive_review_decision(
        normalized, ai_review_config=ai_review_config
    )
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
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "AI review requires the 'openai' package. Install it and set OPENAI_API_KEY."
        ) from exc

    api_key = (
        api_key
        or _resolve_config_value(ai_review_config, "api_key")
        or _resolve_env_value(
            _resolve_config_value(ai_review_config, "api_key_env") or "",
            "OPENAI_API_KEY",
            "AI_REVIEW_API_KEY",
        )
    )
    if not api_key:
        raise RuntimeError("AI review requires OPENAI_API_KEY or AI_REVIEW_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=(
            base_url
            or _resolve_config_value(ai_review_config, "base_url")
            or _resolve_env_value("OPENAI_BASE_URL")
        ),
        timeout=_resolve_request_timeout_sec(timeout_sec),
    )
    system_prompt = build_system_prompt(ai_review_config=ai_review_config)
    user_prompt = build_user_prompt(candidate, ai_review_config=ai_review_config)

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return parse_review_response(
            _extract_response_text(response),
            ai_review_config=ai_review_config,
            candidate=candidate,
        )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    text = completion.choices[0].message.content if completion.choices else ""
    return parse_review_response(
        text,
        ai_review_config=ai_review_config,
        candidate=candidate,
    )


def review_candidate_with_select(
    candidate: Dict[str, Any],
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "AI review with Select requires the 'openai' package, and NOVAI_API_KEY or OPENAI_API_KEY."
        ) from exc

    api_key = (
        api_key
        or _resolve_config_value(ai_review_config, "api_key")
        or _resolve_env_value(
            _resolve_config_value(ai_review_config, "api_key_env") or "",
            "NOVAI_API_KEY",
            "OPENAI_API_KEY",
            "AI_REVIEW_API_KEY",
        )
    )
    if not api_key:
        raise RuntimeError(
            "AI review requires NOVAI_API_KEY or OPENAI_API_KEY or AI_REVIEW_API_KEY"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=(
            base_url
            or _resolve_config_value(ai_review_config, "base_url")
            or _resolve_env_value("NOVAI_BASE_URL", "OPENAI_BASE_URL")
            or "https://once.novai.su/v1"
        ),
        timeout=_resolve_request_timeout_sec(timeout_sec),
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(ai_review_config=ai_review_config)},
            {
                "role": "user",
                "content": build_user_prompt(candidate, ai_review_config=ai_review_config),
            },
        ],
        response_format={"type": "json_object"},
    )
    text = completion.choices[0].message.content if completion.choices else ""
    return parse_review_response(
        text,
        ai_review_config=ai_review_config,
        candidate=candidate,
    )


def review_candidate_with_gemini(
    candidate: Dict[str, Any],
    *,
    model: str,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("AI review requires GEMINI_API_KEY")

    system_prompt = build_system_prompt(ai_review_config=ai_review_config)
    user_prompt = build_user_prompt(candidate, ai_review_config=ai_review_config)
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
        return parse_review_response(
            _extract_gemini_response_text(response),
            ai_review_config=ai_review_config,
            candidate=candidate,
        )

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
    return parse_review_response(
        text,
        ai_review_config=ai_review_config,
        candidate=candidate,
    )


def review_candidate_with_glm5(
    candidate: Dict[str, Any],
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "AI review with GLM-5 requires the 'openai' package, and MODELSCOPE_API_KEY or GLM5_API_KEY."
        ) from exc

    api_key = (
        api_key
        or _resolve_config_value(ai_review_config, "api_key")
        or _resolve_env_value(
            _resolve_config_value(ai_review_config, "api_key_env") or "",
            "MODELSCOPE_API_KEY",
            "GLM5_API_KEY",
        )
    )
    if not api_key:
        raise RuntimeError("AI review requires MODELSCOPE_API_KEY or GLM5_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=(
            base_url
            or _resolve_config_value(ai_review_config, "base_url")
            or os.getenv("MODELSCOPE_BASE_URL")
            or "https://api-inference.modelscope.cn/v1"
        ),
        timeout=_resolve_request_timeout_sec(timeout_sec),
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(ai_review_config=ai_review_config)},
            {
                "role": "user",
                "content": build_user_prompt(candidate, ai_review_config=ai_review_config),
            },
        ],
        response_format={"type": "json_object"},
    )
    text = completion.choices[0].message.content if completion.choices else ""
    return parse_review_response(
        text,
        ai_review_config=ai_review_config,
        candidate=candidate,
    )


def review_candidate_with_provider(
    candidate: Dict[str, Any],
    *,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    provider_norm = str(provider or "openai").strip().lower()
    extra_kwargs: Dict[str, Any] = {}
    if ai_review_config is not None:
        extra_kwargs["ai_review_config"] = ai_review_config
    if base_url is not None:
        extra_kwargs["base_url"] = base_url
    if provider_norm == "openai":
        return review_candidate_with_openai(
            candidate,
            model=model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            **extra_kwargs,
        )
    if provider_norm == "select":
        return review_candidate_with_select(
            candidate,
            model=model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            **extra_kwargs,
        )
    if provider_norm == "gemini":
        return review_candidate_with_gemini(
            candidate,
            model=model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            **extra_kwargs,
        )
    if provider_norm == "glm5":
        return review_candidate_with_glm5(
            candidate,
            model=model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            **extra_kwargs,
        )
    raise ValueError(f"Unsupported AI review provider: {provider}")


def review_candidates_with_ai(
    candidates: List[Dict[str, Any]],
    *,
    provider: str,
    model: str,
    max_candidates: int,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    ai_review_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    provider_norm = str(provider or "openai").strip().lower()
    if provider_norm not in {"openai", "select", "gemini", "glm5"}:
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
                base_url=base_url,
                timeout_sec=timeout_sec,
                ai_review_config=ai_review_config,
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
                f"interpretability={review.get('interpretability_score', 0):.2f}"
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
