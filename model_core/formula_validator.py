"""
公式结构验证器 (Formula Structure Validator) V2

用于在进化搜索过程中过滤具有已知缺陷的公式结构，避免分数坍缩问题。

V2 更新:
- 新增 SIGN-LOG 距离检查 (解决 LOG → SIGN → LOG 问题)
- 扩展 LOG 上游黑名单 (CS_DEMEAN, CS_RANK, NEG, TS_BIAS5)
- 多重离散化检测 (末尾连续离散算子)
- 更严格的 TS 链阈值

使用方法:
    from model_core.formula_validator import validate_formula
    
    is_valid, penalty, reason = validate_formula(formula)
    if not is_valid:
        return penalty, None  # 直接拒绝
"""

from typing import List, Tuple, Set
from .ops_registry import OpsRegistry
from .config import RobustConfig

# ============================================================
# 算子分类 (Operator Categories)
# ============================================================

# 离散化算子 - 输出为有限离散值
DISCRETE_OPS: Set[str] = {"SIGN", "CUT_NEG", "CUT_HIGH"}

# 可能输出 0 或负值的算子
ZERO_NEGATIVE_OPS: Set[str] = {
    "SIGN",       # {-1, 0, 1}
    "CUT_NEG",    # 0 或正值
    "CUT_HIGH",   # 可能返回 -inf
    "NEG",        # 取反，可能为负
    "CS_DEMEAN",  # 均值=0，大量接近0的值
    "CS_RANK",    # [0, 1]，包含0
    "TS_DELTA",   # 差分，可能为负
    "TS_RET",     # 收益率，可能为负
    "TS_BIAS5",   # 乖离率，可能为负
    "TS_STD5",    # 波动为0时输出0
}

# 时序算子前缀
TS_PREFIX = "TS_"


# ============================================================
# 硬过滤规则 (Hard Filters) - 直接拒绝
# ============================================================

# 直接相邻的禁止序列
HARD_FORBIDDEN_SEQUENCES: Set[Tuple[str, str]] = {
    ("SIGN", "LOG"),      # LOG(SIGN(x)) → 常数
    ("SIGN", "SQRT"),     # SQRT(SIGN(x)) → {0, 1}
    ("ABS", "SIGN"),      # SIGN(ABS(x)) → {0, 1}
    ("CUT_HIGH", "LOG"),  # LOG(CUT_HIGH(x)) → CUT_HIGH 返回 -1e9
    ("NEG", "SQRT"),      # SQRT(NEG(x)) → 负数被 clamp
    ("NEG", "LOG"),       # LOG(NEG(x)) → 负数被 clamp
}

# 禁止作为公式末尾的算子
FORBIDDEN_TERMINAL_OPS: Set[str] = {"SIGN"}

# SIGN 与 LOG 的最大允许距离 (超过此距离仍视为危险)
MAX_SIGN_LOG_DISTANCE = 2


# ============================================================
# 软惩罚规则 (Soft Penalties) - 扣分但允许继续
# ============================================================
# V2.2: Scale-Aware 惩罚值 (考虑 RobustConfig.SCALE=5.0)
# - 高危 (数值爆炸风险): -3.0 (等效 Sharpe 损失 -0.6)
# - 中危 (信息丢失风险): -2.0 (等效 Sharpe 损失 -0.4)
# - 低危 (结构冗余风险): -1.0 (等效 Sharpe 损失 -0.2)

# 相邻序列软惩罚
SOFT_PENALTY_SEQUENCES = {
    # 高危: 数值爆炸风险 (-3.0)
    ("SIGN", "DIV"): -3.0,        # 分母 {-1, 0, 1}，极易除零
    ("CS_RANK", "DIV"): -3.0,     # 分母 [0,1]，接近 0 时爆炸
    
    # 中危: 信息丢失风险 (-2.0)
    ("CUT_NEG", "LOG"): -2.0,     # 大量 0 值导致 LOG 无区分度
    ("CUT_NEG", "DIV"): -2.0,     # 分母可能为 0
    ("CS_DEMEAN", "LOG"): -2.0,   # 均值=0，大量接近 0 的值
    ("CS_DEMEAN", "DIV"): -2.0,   # 分母接近 0
    ("SIGN", "CS_RANK"): -2.0,    # 离散 + 排名 = 信息丢失
    ("CS_RANK", "SIGN"): -2.0,    # 排名 + 离散 = 信息丢失
    ("MIN", "DIV"): -2.0,         # MIN 作为分母可能接近 0
    
    # 低危: 结构冗余风险 (-1.0)
    ("TS_STD5", "LOG"): -1.0,     # 波动为 0 时 LOG 变常数
    ("TS_DELTA", "LOG"): -1.0,    # 差分可能为负，被 clamp
    ("TS_DELTA", "SQRT"): -1.0,   # 同上
    ("TS_RET", "LOG"): -1.0,      # 收益率可能为负
    ("TS_RET", "SQRT"): -1.0,     # 同上
    ("CS_RANK", "LOG"): -1.0,     # 输出 [0,1]，0 处 LOG 塌缩
    ("TS_BIAS5", "LOG"): -1.0,    # 乖离率可能为负
    ("TS_BIAS5", "SQRT"): -1.0,   # 同上
    ("IF_POS", "LOG"): -1.0,      # 分支可能输出 0 或负值
    ("MIN", "LOG"): -1.0,         # MIN 可能取到被截断的 0/极小值
}

# 软惩罚的末尾算子 (中危: -2.0)
SOFT_TERMINAL_PENALTIES = {
    "CUT_NEG": -2.0,   # 所有负值资产得分相同
    "CUT_HIGH": -2.0,  # 高值被截断
}


# ============================================================
# 结构性惩罚 (Structural Penalties)
# ============================================================
# V2.2: 与序列惩罚对齐

MAX_CONSECUTIVE_LOG = 2       # 允许的最大连续 LOG 数量
LOG_EXCESS_PENALTY = -1.0     # 低危: 每超过一个 LOG 的惩罚

MAX_CONSECUTIVE_TS = 2        # 允许的最大连续 TS_* 数量
TS_EXCESS_PENALTY = -1.0      # 低危: 每超过一个的惩罚

MAX_TERMINAL_DISCRETE = 1     # 末尾允许的最大离散算子数量
TERMINAL_DISCRETE_PENALTY = -2.0  # 中危: 超过时的惩罚

MAX_TOTAL_DISCRETE = 3        # 公式中允许的最大离散算子总数
TOTAL_DISCRETE_PENALTY = -1.0 # 低危: 每超过一个的惩罚

# V2.3: 局部密度检测 (Local Density Check) - 解决“间接堆叠”问题
# V2.3: 局部密度检测 (Local Density Check) - 解决“间接堆叠”问题
# 参数现已移至 RobustConfig (config.py/yaml)


# 注: SIGN-LOG 距离检查使用硬过滤 (直接拒绝)，不使用软惩罚

# ============================================================
# RPN 堆栈模拟校验 (RPN Stack Simulation)
# ============================================================

# 模块级缓存
_CACHE_OPS_ARITY = None
_CACHE_FEATURES = None
_CACHE_FEAT_HASH = None

def _get_validation_cache():
    global _CACHE_OPS_ARITY, _CACHE_FEATURES, _CACHE_FEAT_HASH
    from .ops_registry import OpsRegistry
    from .config import ModelConfig
    
    # Init Ops Cache (Static)
    if _CACHE_OPS_ARITY is None:
        _CACHE_OPS_ARITY = {name: arity for name, _, arity in OpsRegistry.get_ops_config()}
        
    # Init/Update Features Cache (Dynamic)
    current_feats = ModelConfig.INPUT_FEATURES
    current_hash = hash(tuple(current_feats))
    
    if _CACHE_FEATURES is None or current_hash != _CACHE_FEAT_HASH:
        _CACHE_FEATURES = set(current_feats)
        _CACHE_FEAT_HASH = current_hash
        
    return _CACHE_OPS_ARITY, _CACHE_FEATURES


def validate_formula(formula: List[str]) -> Tuple[bool, float, str]:
    """
    验证公式结构，检测已知的反模式。
    
    Args:
        formula: 公式字符串列表 (如 ["PURE_VALUE", "SIGN", "LOG"])
    
    Returns:
        (is_valid, penalty, reason)
        - is_valid: 是否通过硬过滤 (False = 直接拒绝)
        - penalty: 累计软惩罚分数 (仅当 is_valid=True 时有意义)
        - reason: 拒绝/惩罚原因
    """
    if not isinstance(formula, list) or not formula:
        return False, -5.0, "Empty/Invalid formula"
    
    # [V6] RPN 堆栈模拟校验
    # 目的: 提前拦截无效 RPN，解决 EXEC_NONE 效率问题
    ops_arity, features = _get_validation_cache()
    
    stack_depth = 0
    for i, token in enumerate(formula):
        if not isinstance(token, str):
            return False, -5.0, "TYPE_ERROR"
            
        if token in features:
            stack_depth += 1
        elif token in ops_arity:
            arity = ops_arity[token]
            if stack_depth < arity:
                return False, -5.0, "RPN_UNDERFLOW"
            stack_depth = stack_depth - arity + 1
        else:
            # 统一归类未知 Token
            return False, -5.0, "UNKNOWN_TOKEN"
            
    if stack_depth != 1:
        return False, -5.0, f"RPN_LEFTOVER (d={stack_depth})"

    
    total_penalty = 0.0
    reasons = []
    
    # ==================== 硬过滤检查 ====================
    
    # 1. 检查直接相邻的禁止序列
    for i in range(len(formula) - 1):
        pair = (formula[i], formula[i + 1])
        if pair in HARD_FORBIDDEN_SEQUENCES:
            return False, -5.0, f"Forbidden sequence: {pair[0]} → {pair[1]}"
    
    # 2. 检查禁止的末尾算子
    if formula[-1] in FORBIDDEN_TERMINAL_OPS:
        return False, -5.0, f"Forbidden terminal operator: {formula[-1]}"
    
    # 3. [V2] 检查 SIGN 和 LOG 的距离 (解决 LOG → SIGN → LOG 问题)
    sign_positions = [i for i, t in enumerate(formula) if t == "SIGN"]
    log_positions = [i for i, t in enumerate(formula) if t == "LOG"]
    
    for sign_pos in sign_positions:
        for log_pos in log_positions:
            # 只检查 SIGN 在 LOG 之前的情况 (SIGN ... LOG)
            if log_pos > sign_pos:
                distance = log_pos - sign_pos
                if distance <= MAX_SIGN_LOG_DISTANCE:
                    return False, -5.0, f"SIGN-LOG proximity: distance={distance} (max={MAX_SIGN_LOG_DISTANCE})"
    
    # 4. [V2] 检查末尾连续离散算子
    terminal_discrete_count = 0
    for token in reversed(formula):
        if token in DISCRETE_OPS:
            terminal_discrete_count += 1
        else:
            break
    
    if terminal_discrete_count > MAX_TERMINAL_DISCRETE:
        return False, -5.0, f"Too many terminal discrete ops: {terminal_discrete_count}"
    
    # ==================== 软惩罚检查 ====================
    
    # 5. 检查软惩罚序列
    for i in range(len(formula) - 1):
        pair = (formula[i], formula[i + 1])
        if pair in SOFT_PENALTY_SEQUENCES:
            penalty = SOFT_PENALTY_SEQUENCES[pair]
            total_penalty += penalty
            reasons.append(f"{pair[0]}→{pair[1]} ({penalty})")
    
    # 6. 检查软惩罚末尾算子
    if formula[-1] in SOFT_TERMINAL_PENALTIES:
        penalty = SOFT_TERMINAL_PENALTIES[formula[-1]]
        total_penalty += penalty
        reasons.append(f"Terminal {formula[-1]} ({penalty})")
    
    # 7. 检查连续 LOG 数量
    log_chain = 0
    max_log_chain = 0
    for token in formula:
        if token == "LOG":
            log_chain += 1
            max_log_chain = max(max_log_chain, log_chain)
        else:
            log_chain = 0
    
    if max_log_chain > MAX_CONSECUTIVE_LOG:
        excess = max_log_chain - MAX_CONSECUTIVE_LOG
        penalty = excess * LOG_EXCESS_PENALTY
        total_penalty += penalty
        reasons.append(f"LOG chain={max_log_chain} ({penalty})")
    
    # 8. 检查连续 TS_* 算子数量
    ts_chain = 0
    max_ts_chain = 0
    for token in formula:
        if token.startswith(TS_PREFIX):
            ts_chain += 1
            max_ts_chain = max(max_ts_chain, ts_chain)
        else:
            ts_chain = 0
    
    if max_ts_chain > MAX_CONSECUTIVE_TS:
        excess = max_ts_chain - MAX_CONSECUTIVE_TS
        penalty = excess * TS_EXCESS_PENALTY
        total_penalty += penalty
        reasons.append(f"TS chain={max_ts_chain} ({penalty})")
    
    # 9. [V2] 检查离散算子总数
    discrete_count = sum(1 for t in formula if t in DISCRETE_OPS)
    if discrete_count > MAX_TOTAL_DISCRETE:
        excess = discrete_count - MAX_TOTAL_DISCRETE
        penalty = excess * TOTAL_DISCRETE_PENALTY
        total_penalty += penalty
        reasons.append(f"Total discrete={discrete_count} ({penalty})")
    
    # 10. [V2.3] 局部密度检测 (Local Density Check)
    # 检查滑动窗口内的 TS_* 算子密度，打击间接堆叠
    density_window = RobustConfig.DENSITY_WINDOW
    max_ts = RobustConfig.MAX_TS_IN_WINDOW
    density_penalty = RobustConfig.DENSITY_PENALTY
    
    if len(formula) >= density_window:
        for i in range(len(formula) - density_window + 1):
            window = formula[i : i + density_window]
            ts_count = sum(1 for t in window if t.startswith(TS_PREFIX))
            
            if ts_count > max_ts:
                penalty = density_penalty * (ts_count - max_ts)
                total_penalty += penalty
                reasons.append(f"High TS Density @ {i} ({penalty})")
                # 只在第一次触发时惩罚，避免重复惩罚
                break
    
    # 构建原因字符串
    reason_str = "; ".join(reasons) if reasons else "OK"
    
    return True, total_penalty, reason_str


def get_validation_summary() -> str:
    """返回当前验证规则的摘要 (用于日志/调试)"""
    lines = [
        "=== Formula Validator Rules (V2) ===",
        f"Hard Forbidden Sequences: {len(HARD_FORBIDDEN_SEQUENCES)}",
        f"Soft Penalty Sequences: {len(SOFT_PENALTY_SEQUENCES)}",
        f"Max SIGN-LOG Distance: {MAX_SIGN_LOG_DISTANCE}",
        f"Max Consecutive LOG: {MAX_CONSECUTIVE_LOG}",
        f"Max Consecutive TS_*: {MAX_CONSECUTIVE_TS}",
        f"Max Total Discrete Ops: {MAX_TOTAL_DISCRETE}",
    ]
    return "\n".join(lines)
