"""
鍏紡缁撴瀯楠岃瘉鍣?(Formula Structure Validator) V2

鐢ㄤ簬鍦ㄨ繘鍖栨悳绱㈣繃绋嬩腑杩囨护鍏锋湁宸茬煡缂洪櫡鐨勫叕寮忕粨鏋勶紝閬垮厤鍒嗘暟鍧嶇缉闂銆?

V2 鏇存柊:
- 鏂板 SIGN-LOG 璺濈妫€鏌?(瑙ｅ喅 LOG 鈫?SIGN 鈫?LOG 闂)
- 鎵╁睍 LOG 涓婃父榛戝悕鍗?(CS_DEMEAN, CS_RANK, NEG, TS_BIAS5)
- 澶氶噸绂绘暎鍖栨娴?(鏈熬杩炵画绂绘暎绠楀瓙)
- 鏇翠弗鏍肩殑 TS 閾鹃槇鍊?

浣跨敤鏂规硶:
    from model_core.formula_validator import validate_formula
    
    is_valid, penalty, reason = validate_formula(formula)
    if not is_valid:
        return penalty, None  # 鐩存帴鎷掔粷
"""

from typing import List, Tuple, Set
from .ops_registry import OpsRegistry
from .config import RobustConfig

# ============================================================
# 绠楀瓙鍒嗙被 (Operator Categories)
# ============================================================

# 绂绘暎鍖栫畻瀛?- 杈撳嚭涓烘湁闄愮鏁ｅ€?
DISCRETE_OPS: Set[str] = {"SIGN", "CUT_NEG", "CUT_HIGH"}

# 鍙兘杈撳嚭 0 鎴栬礋鍊肩殑绠楀瓙
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
    "TS_MOM10",   # 动量差分，可能为负
    "TS_MOM20",   # 动量差分，可能为负
    "TS_STD20",   # 波动可能为0
    "TS_STD60",   # 波动可能为0
}

# 鏃跺簭绠楀瓙鍓嶇紑
TS_PREFIX = "TS_"


# ============================================================
# 纭繃婊よ鍒?(Hard Filters) - 鐩存帴鎷掔粷
# ============================================================

# 鐩存帴鐩搁偦鐨勭姝㈠簭鍒?
HARD_FORBIDDEN_SEQUENCES: Set[Tuple[str, str]] = {
    ("SIGN", "LOG"),      # LOG(SIGN(x)) 鈫?甯告暟
    ("SIGN", "SQRT"),     # SQRT(SIGN(x)) 鈫?{0, 1}
    ("ABS", "SIGN"),      # SIGN(ABS(x)) 鈫?{0, 1}
    ("CUT_HIGH", "LOG"),  # LOG(CUT_HIGH(x)) 鈫?CUT_HIGH 杩斿洖 -1e9
    ("NEG", "SQRT"),      # SQRT(NEG(x)) 鈫?璐熸暟琚?clamp
    ("NEG", "LOG"),       # LOG(NEG(x)) 鈫?璐熸暟琚?clamp
}

# 绂佹浣滀负鍏紡鏈熬鐨勭畻瀛?
FORBIDDEN_TERMINAL_OPS: Set[str] = {"SIGN"}

# SIGN 涓?LOG 鐨勬渶澶у厑璁歌窛绂?(瓒呰繃姝よ窛绂讳粛瑙嗕负鍗遍櫓)
MAX_SIGN_LOG_DISTANCE = 2


# ============================================================
# 杞儵缃氳鍒?(Soft Penalties) - 鎵ｅ垎浣嗗厑璁哥户缁?
# ============================================================
# V2.2: Scale-Aware 鎯╃綒鍊?(鑰冭檻 RobustConfig.SCALE=5.0)
# - 楂樺嵄 (鏁板€肩垎鐐搁闄?: -3.0 (绛夋晥 Sharpe 鎹熷け -0.6)
# - 涓嵄 (淇℃伅涓㈠け椋庨櫓): -2.0 (绛夋晥 Sharpe 鎹熷け -0.4)
# - 浣庡嵄 (缁撴瀯鍐椾綑椋庨櫓): -1.0 (绛夋晥 Sharpe 鎹熷け -0.2)

# 鐩搁偦搴忓垪杞儵缃?
SOFT_PENALTY_SEQUENCES = {
    # 高危: 数值爆炸风险 (-3.0)
    ("SIGN", "DIV"): -3.0,        # 分母 {-1, 0, 1}
    ("CS_RANK", "DIV"): -3.0,     # 分母 [0,1]
    ("TS_MOM10", "DIV"): -3.0,    # 分母可能接近 0
    ("TS_MOM20", "DIV"): -3.0,    # 分母可能接近 0
    ("TS_DELTA", "DIV"): -3.0,    # 分母可能接近 0
    ("TS_STD5", "DIV"): -3.0,     # 波动可能接近 0
    ("TS_STD20", "DIV"): -3.0,    # 波动可能接近 0
    ("TS_STD60", "DIV"): -3.0,    # 波动可能接近 0

    # 中危: 信息丢失/不稳定风险 (-2.0)
    ("CUT_NEG", "LOG"): -2.0,
    ("CUT_NEG", "DIV"): -2.0,
    ("CS_DEMEAN", "LOG"): -2.0,
    ("CS_DEMEAN", "DIV"): -2.0,
    ("CS_ROBUST_Z", "DIV"): -2.0,
    ("SIGN", "CS_RANK"): -2.0,
    ("CS_RANK", "SIGN"): -2.0,
    ("MIN", "DIV"): -2.0,

    # 低危: 结构冗余风险 (-1.0)
    ("TS_STD5", "LOG"): -1.0,
    ("TS_DELTA", "LOG"): -1.0,
    ("TS_DELTA", "SQRT"): -1.0,
    ("TS_RET", "LOG"): -1.0,
    ("TS_RET", "SQRT"): -1.0,
    ("CS_RANK", "LOG"): -1.0,
    ("TS_BIAS5", "LOG"): -1.0,
    ("TS_BIAS5", "SQRT"): -1.0,
    ("TS_MOM10", "LOG"): -1.0,
    ("TS_MOM10", "SQRT"): -1.0,
    ("TS_MOM20", "LOG"): -1.0,
    ("TS_MOM20", "SQRT"): -1.0,
    ("TS_STD5", "TS_MAX20"): -1.0,
    ("TS_DELTA", "TS_MAX20"): -1.0,
    ("TS_STD5", "TS_MIN20"): -1.0,
    ("TS_DELTA", "TS_MIN20"): -1.0,
    ("IF_POS", "LOG"): -1.0,
    ("MIN", "LOG"): -1.0,
}

# 杞儵缃氱殑鏈熬绠楀瓙 (涓嵄: -2.0)
SOFT_TERMINAL_PENALTIES = {
    "CUT_NEG": -2.0,   # 鎵€鏈夎礋鍊艰祫浜у緱鍒嗙浉鍚?
    "CUT_HIGH": -2.0,  # 楂樺€艰鎴柇
}


# ============================================================
# 缁撴瀯鎬ф儵缃?(Structural Penalties)
# ============================================================
# V2.2: 涓庡簭鍒楁儵缃氬榻?

MAX_CONSECUTIVE_LOG = 2       # 鍏佽鐨勬渶澶ц繛缁?LOG 鏁伴噺
LOG_EXCESS_PENALTY = -1.0     # 浣庡嵄: 姣忚秴杩囦竴涓?LOG 鐨勬儵缃?

MAX_CONSECUTIVE_TS = 2        # 鍏佽鐨勬渶澶ц繛缁?TS_* 鏁伴噺
TS_EXCESS_PENALTY = -1.0      # 浣庡嵄: 姣忚秴杩囦竴涓殑鎯╃綒

MAX_TERMINAL_DISCRETE = 1     # 鏈熬鍏佽鐨勬渶澶х鏁ｇ畻瀛愭暟閲?
TERMINAL_DISCRETE_PENALTY = -2.0  # 涓嵄: 瓒呰繃鏃剁殑鎯╃綒

MAX_TOTAL_DISCRETE = 3        # 鍏紡涓厑璁哥殑鏈€澶х鏁ｇ畻瀛愭€绘暟
TOTAL_DISCRETE_PENALTY = -1.0 # 浣庡嵄: 姣忚秴杩囦竴涓殑鎯╃綒

MAX_TOTAL_DIV = 2             # 鍏紡涓厑璁哥殑鏈€澶?DIV 鏁伴噺
DIV_EXCESS_PENALTY = -1.0     # 浣庡嵄: 姣忚秴杩囦竴涓殑鎯╃綒

# V2.3: 灞€閮ㄥ瘑搴︽娴?(Local Density Check) - 瑙ｅ喅鈥滈棿鎺ュ爢鍙犫€濋棶棰?
# V2.3: 灞€閮ㄥ瘑搴︽娴?(Local Density Check) - 瑙ｅ喅鈥滈棿鎺ュ爢鍙犫€濋棶棰?
# 鍙傛暟鐜板凡绉昏嚦 RobustConfig (config.py/yaml)


# 娉? SIGN-LOG 璺濈妫€鏌ヤ娇鐢ㄧ‖杩囨护 (鐩存帴鎷掔粷)锛屼笉浣跨敤杞儵缃?

# ============================================================
# RPN 鍫嗘爤妯℃嫙鏍￠獙 (RPN Stack Simulation)
# ============================================================

# 妯″潡绾х紦瀛?
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
    楠岃瘉鍏紡缁撴瀯锛屾娴嬪凡鐭ョ殑鍙嶆ā寮忋€?
    
    Args:
        formula: 鍏紡瀛楃涓插垪琛?(濡?["PURE_VALUE", "SIGN", "LOG"])
    
    Returns:
        (is_valid, penalty, reason)
        - is_valid: 鏄惁閫氳繃纭繃婊?(False = 鐩存帴鎷掔粷)
        - penalty: 绱杞儵缃氬垎鏁?(浠呭綋 is_valid=True 鏃舵湁鎰忎箟)
        - reason: 鎷掔粷/鎯╃綒鍘熷洜
    """
    if not isinstance(formula, list) or not formula:
        return False, -5.0, "Empty/Invalid formula"
    
    # [V6] RPN 鍫嗘爤妯℃嫙鏍￠獙
    # 鐩殑: 鎻愬墠鎷︽埅鏃犳晥 RPN锛岃В鍐?EXEC_NONE 鏁堢巼闂
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
            # 缁熶竴褰掔被鏈煡 Token
            return False, -5.0, "UNKNOWN_TOKEN"
            
    if stack_depth != 1:
        return False, -5.0, f"RPN_LEFTOVER (d={stack_depth})"

    
    total_penalty = 0.0
    reasons = []
    
    # ==================== 纭繃婊ゆ鏌?====================
    
    # 1. 妫€鏌ョ洿鎺ョ浉閭荤殑绂佹搴忓垪
    for i in range(len(formula) - 1):
        pair = (formula[i], formula[i + 1])
        if pair in HARD_FORBIDDEN_SEQUENCES:
            return False, -5.0, f"Forbidden sequence: {pair[0]} 鈫?{pair[1]}"
    
    # 2. 妫€鏌ョ姝㈢殑鏈熬绠楀瓙
    if formula[-1] in FORBIDDEN_TERMINAL_OPS:
        return False, -5.0, f"Forbidden terminal operator: {formula[-1]}"
    
    # 3. [V2] 妫€鏌?SIGN 鍜?LOG 鐨勮窛绂?(瑙ｅ喅 LOG 鈫?SIGN 鈫?LOG 闂)
    sign_positions = [i for i, t in enumerate(formula) if t == "SIGN"]
    log_positions = [i for i, t in enumerate(formula) if t == "LOG"]
    
    for sign_pos in sign_positions:
        for log_pos in log_positions:
            # 鍙鏌?SIGN 鍦?LOG 涔嬪墠鐨勬儏鍐?(SIGN ... LOG)
            if log_pos > sign_pos:
                distance = log_pos - sign_pos
                if distance <= MAX_SIGN_LOG_DISTANCE:
                    return False, -5.0, f"SIGN-LOG proximity: distance={distance} (max={MAX_SIGN_LOG_DISTANCE})"
    
    # 4. [V2] 妫€鏌ユ湯灏捐繛缁鏁ｇ畻瀛?
    terminal_discrete_count = 0
    for token in reversed(formula):
        if token in DISCRETE_OPS:
            terminal_discrete_count += 1
        else:
            break
    
    if terminal_discrete_count > MAX_TERMINAL_DISCRETE:
        return False, -5.0, f"Too many terminal discrete ops: {terminal_discrete_count}"
    
    # ==================== 杞儵缃氭鏌?====================
    
    # 5. 妫€鏌ヨ蒋鎯╃綒搴忓垪
    for i in range(len(formula) - 1):
        pair = (formula[i], formula[i + 1])
        if pair in SOFT_PENALTY_SEQUENCES:
            penalty = SOFT_PENALTY_SEQUENCES[pair]
            total_penalty += penalty
            reasons.append(f"{pair[0]}->{pair[1]} ({penalty})")
    
    # 6. 妫€鏌ヨ蒋鎯╃綒鏈熬绠楀瓙
    if formula[-1] in SOFT_TERMINAL_PENALTIES:
        penalty = SOFT_TERMINAL_PENALTIES[formula[-1]]
        total_penalty += penalty
        reasons.append(f"Terminal {formula[-1]} ({penalty})")
    
    # 7. 妫€鏌ヨ繛缁?LOG 鏁伴噺
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
    
    # 8. 妫€鏌ヨ繛缁?TS_* 绠楀瓙鏁伴噺
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
    
    # 9. [V2] 妫€鏌ョ鏁ｇ畻瀛愭€绘暟
    discrete_count = sum(1 for t in formula if t in DISCRETE_OPS)
    if discrete_count > MAX_TOTAL_DISCRETE:
        excess = discrete_count - MAX_TOTAL_DISCRETE
        penalty = excess * TOTAL_DISCRETE_PENALTY
        total_penalty += penalty
        reasons.append(f"Total discrete={discrete_count} ({penalty})")

    # 9.5 DIV 杩囧瘑鎯╃綒锛堥伩鍏嶆繁閾炬潯鏁板€兼斁澶э級
    div_count = sum(1 for t in formula if t == "DIV")
    if div_count > MAX_TOTAL_DIV:
        excess = div_count - MAX_TOTAL_DIV
        penalty = excess * DIV_EXCESS_PENALTY
        total_penalty += penalty
        reasons.append(f"Total DIV={div_count} ({penalty})")
    
    # 10. [V2.3] 灞€閮ㄥ瘑搴︽娴?(Local Density Check)
    # 妫€鏌ユ粦鍔ㄧ獥鍙ｅ唴鐨?TS_* 绠楀瓙瀵嗗害锛屾墦鍑婚棿鎺ュ爢鍙?
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
                # 鍙湪绗竴娆¤Е鍙戞椂鎯╃綒锛岄伩鍏嶉噸澶嶆儵缃?
                break
    
    # 鏋勫缓鍘熷洜瀛楃涓?
    reason_str = "; ".join(reasons) if reasons else "OK"
    
    return True, total_penalty, reason_str


def get_validation_summary() -> str:
    """杩斿洖褰撳墠楠岃瘉瑙勫垯鐨勬憳瑕?(鐢ㄤ簬鏃ュ織/璋冭瘯)"""
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


