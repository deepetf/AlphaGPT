"""
算子注册中心 (Operator Registry)

使用方法:
    from model_core.ops_registry import OpsRegistry, register_op

    @register_op(name='MY_OP', arity=1, description='我的自定义算子')
    def my_custom_op(x):
        return x * 2

然后 AlphaGPT 就能自动使用这个新算子了。
"""
import torch
from typing import Callable, Dict, Any, List, Tuple


class OpsRegistry:
    """算子注册表 - 单例模式"""
    _ops: Dict[str, Dict[str, Any]] = {}
    _frozen = False  # 冻结后不再接受新注册
    
    @classmethod
    def register(cls, name: str, arity: int, description: str = ''):
        """
        注册算子的装饰器
        
        Args:
            name: 算子名称 (会成为 AlphaGPT 词表的一部分)
            arity: 算子需要的参数数量 (1=一元算子, 2=二元算子)
            description: 算子描述 (用于调试和文档)
        """
        if cls._frozen:
            raise RuntimeError(f"Cannot register new op '{name}': registry is frozen.")
        
        def decorator(func: Callable):
            if name in cls._ops:
                print(f"Warning: Op '{name}' already registered, overwriting.")
            cls._ops[name] = {
                'func': func,
                'arity': arity,
                'description': description
            }
            return func
        return decorator
    
    @classmethod
    def get_ops_config(cls) -> List[Tuple[str, Callable, int]]:
        """
        返回与旧版 OPS_CONFIG 兼容的格式
        [(name, func, arity), ...]
        """
        return [(name, info['func'], info['arity']) for name, info in cls._ops.items()]
    
    @classmethod
    def get_op(cls, name: str) -> Dict[str, Any]:
        """获取单个算子信息"""
        return cls._ops.get(name)
    
    @classmethod
    def list_ops(cls) -> List[str]:
        """列出所有已注册算子名称"""
        return list(cls._ops.keys())
    
    @classmethod
    def freeze(cls):
        """冻结注册表，防止训练过程中动态添加算子"""
        cls._frozen = True
    
    @classmethod
    def clear(cls):
        """清空注册表 (仅用于测试)"""
        cls._ops = {}
        cls._frozen = False


# 方便使用的别名
register_op = OpsRegistry.register


# ============================================================
# 预注册的核心算子 (Core Operators)
# ============================================================

# --- 基础数学算子 ---
@register_op('ADD', 2, '加法')
def op_add(a, b):
    return a + b

@register_op('SUB', 2, '减法')
def op_sub(a, b):
    return a - b

@register_op('MUL', 2, '乘法')
def op_mul(a, b):
    return a * b

@register_op('DIV', 2, '除法 (安全)')
def op_div(a, b):
    return a / (b + 1e-9)

@register_op('NEG', 1, '取负')
def op_neg(x):
    return -x

@register_op('ABS', 1, '绝对值')
def op_abs(x):
    return torch.abs(x)

@register_op('LOG', 1, '对数 (安全)')
def op_log(x):
    return torch.log(torch.clamp(x, min=1e-9))

@register_op('SQRT', 1, '平方根 (安全)')
def op_sqrt(x):
    return torch.sqrt(torch.clamp(x, min=0))

@register_op('SIGN', 1, '符号函数')
def op_sign(x):
    return torch.sign(x)


# --- 时序算子 (Time-Series Operators) ---
# 注意: 所有 TS_* 算子操作在 dim=0 (Time) 上，对每个 Asset 独立计算
# 必须将边界值设为 0，避免 torch.roll 的循环行为导致未来函数

@register_op('TS_DELAY', 1, '滞后1期')
def op_ts_delay(x):
    result = torch.roll(x, 1, dims=0)
    result[0] = 0  # 第一天无前一天数据，设为 0
    return result

@register_op('TS_DELTA', 1, '一阶差分')
def op_ts_delta(x):
    prev = torch.roll(x, 1, dims=0)
    delta = x - prev
    delta[0] = 0  # 第一天无法计算差分
    return delta

@register_op('TS_RET', 1, '收益率')
def op_ts_ret(x):
    prev = torch.roll(x, 1, dims=0)
    ret = (x - prev) / (prev + 1e-9)
    ret[0] = 0  # 第一天无法计算收益率
    return ret

@register_op('TS_MEAN5', 1, '5日均值')
def op_ts_mean5(x):
    """滚动5日均值，仅使用过去数据"""
    if x.dim() == 2:
        T, N = x.shape
        if T < 5:
            return torch.zeros_like(x)
        # 用第一行重复填充，确保不引入未来数据
        padding = x[0].unsqueeze(0).repeat(4, 1)
        padded = torch.cat([padding, x], dim=0)  # [T+4, N]
        unfolded = padded.unfold(0, 5, 1)  # [T, N, 5]
        result = unfolded.mean(dim=-1)
        # 前4天数据不可靠（因为是用重复值填充的），设为 0
        result[:4] = 0
        return result
    return torch.zeros_like(x)  # 非 2D 输入返回全零，与 TS_STD5 保持一致

@register_op('TS_STD5', 1, '5日标准差')
def op_ts_std5(x):
    """滚动5日标准差，仅使用过去数据"""
    if x.dim() == 2:
        T, N = x.shape
        if T < 5:
            return torch.zeros_like(x)
        padding = x[0].unsqueeze(0).repeat(4, 1)
        padded = torch.cat([padding, x], dim=0)
        unfolded = padded.unfold(0, 5, 1)
        result = unfolded.std(dim=-1)
        result[:4] = 0  # 前4天不可靠
        return result
    return torch.zeros_like(x)


# --- 截面算子 (Cross-Sectional Operators) ---
@register_op('CS_RANK', 1, '截面排名 (0~1)')
def op_cs_rank(x):
    """对每个时间点，计算各资产的排名分位数"""
    if x.dim() == 2:
        # x: [Time, Assets]
        ranks = x.argsort(dim=1).argsort(dim=1).float()
        n = x.shape[1]
        return ranks / (n - 1 + 1e-9)
    return x

@register_op('CS_DEMEAN', 1, '截面去均值')
def op_cs_demean(x):
    """对每个时间点，减去截面均值"""
    if x.dim() == 2:
        mean = x.mean(dim=1, keepdim=True)
        return x - mean
    return x

@register_op('CS_ROBUST_Z', 1, '截面稳健标准化 (Median/MAD)')
def op_cs_robust_z(x):
    """使用中位数和绝对中位差进行标准化，对异常值更鲁棒"""
    if x.dim() == 2:
        median = x.median(dim=1, keepdim=True).values
        mad = (x - median).abs().median(dim=1, keepdim=True).values + 1e-9
        z = (x - median) / (mad * 1.4826)  # 1.4826 使 MAD 近似标准差
        return torch.clamp(z, -5, 5)
    return x


# --- 逻辑算子 (Logic Operators) ---
@register_op('MAX', 2, '取较大值')
def op_max(a, b):
    return torch.maximum(a, b)

@register_op('MIN', 2, '取较小值')
def op_min(a, b):
    return torch.minimum(a, b)

@register_op('IF_POS', 2, '条件选择: 若a>0则a, 否则b')
def op_if_pos(a, b):
    """如果 a > 0, 返回 a; 否则返回 b"""
    return torch.where(a > 0, a, b)

@register_op('CUT_NEG', 1, '负值截断为0')
def op_cut_neg(x):
    return torch.clamp(x, min=0)

@register_op('CUT_HIGH', 1, '高值惩罚 (超过阈值返回负无穷)')
def op_cut_high(x):
    """如果值 > 2 (sigma), 返回 -inf，用于排除异常高的因子"""
    return torch.where(x > 2, torch.tensor(-1e9, device=x.device), x)
