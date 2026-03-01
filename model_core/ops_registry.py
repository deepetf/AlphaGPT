"""
Operator registry and built-in operators.
"""

import torch
from typing import Any, Callable, Dict, List, Tuple


class OpsRegistry:
    """Operator registry singleton."""

    _ops: Dict[str, Dict[str, Any]] = {}
    _frozen = False

    @classmethod
    def register(cls, name: str, arity: int, description: str = ""):
        if cls._frozen:
            raise RuntimeError(f"Cannot register new op '{name}': registry is frozen.")

        def decorator(func: Callable):
            if name in cls._ops:
                print(f"Warning: Op '{name}' already registered, overwriting.")
            cls._ops[name] = {
                "func": func,
                "arity": arity,
                "description": description,
            }
            return func

        return decorator

    @classmethod
    def get_ops_config(cls) -> List[Tuple[str, Callable, int]]:
        return [(name, info["func"], info["arity"]) for name, info in cls._ops.items()]

    @classmethod
    def get_op(cls, name: str) -> Dict[str, Any]:
        return cls._ops.get(name)

    @classmethod
    def list_ops(cls) -> List[str]:
        return list(cls._ops.keys())

    @classmethod
    def list_ts_ops(cls) -> List[str]:
        return [name for name in cls._ops.keys() if name.startswith("TS_")]

    @classmethod
    def freeze(cls):
        cls._frozen = True

    @classmethod
    def clear(cls):
        cls._ops = {}
        cls._frozen = False


register_op = OpsRegistry.register


def _ts_nan_to_num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _ts_lag(x: torch.Tensor, n: int) -> torch.Tensor:
    """Lag helper with roll + zero boundary to avoid lookahead leakage."""
    if x.dim() != 2:
        return torch.zeros_like(x)
    if n <= 0:
        return _ts_nan_to_num(x)

    t = x.shape[0]
    if t <= n:
        return torch.zeros_like(x)

    result = torch.roll(x, n, dims=0)
    result[:n] = 0
    return _ts_nan_to_num(result)


def _ts_rolling_reduce(x: torch.Tensor, window: int, reducer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Rolling helper using left padding + unfold; first window-1 rows are zeroed."""
    if x.dim() != 2:
        return torch.zeros_like(x)

    t = x.shape[0]
    if t < window:
        return torch.zeros_like(x)

    padding = x[0].unsqueeze(0).repeat(window - 1, 1)
    padded = torch.cat([padding, x], dim=0)
    unfolded = padded.unfold(0, window, 1)
    out = reducer(unfolded)
    out[: window - 1] = 0
    return _ts_nan_to_num(out)


# --- Arithmetic operators ---
@register_op("ADD", 2, "addition")
def op_add(a, b):
    return a + b


@register_op("SUB", 2, "subtraction")
def op_sub(a, b):
    return a - b


@register_op("MUL", 2, "multiplication")
def op_mul(a, b):
    return a * b

@register_op("DIV", 2, "safe division")
def op_div(a, b):
    return a / (b + 1e-9)


@register_op("NEG", 1, "negation")
def op_neg(x):
    return -x


@register_op("ABS", 1, "absolute")
def op_abs(x):
    return torch.abs(x)


@register_op("LOG", 1, "safe log")
def op_log(x):
    return torch.log(torch.clamp(x, min=1e-9))


@register_op("SQRT", 1, "safe sqrt")
def op_sqrt(x):
    return torch.sqrt(torch.clamp(x, min=0))


@register_op("SIGN", 1, "sign")
def op_sign(x):
    return torch.sign(x)


# --- Time-series operators ---
@register_op("TS_DELAY", 1, "lag 1")
def op_ts_delay(x):
    return _ts_lag(x, 1)


@register_op("TS_DELTA", 1, "first difference")
def op_ts_delta(x):
    prev = _ts_lag(x, 1)
    out = x - prev
    if out.dim() == 2 and out.shape[0] > 0:
        out[0] = 0
    return _ts_nan_to_num(out)


@register_op("TS_RET", 1, "one-step return")
def op_ts_ret(x):
    prev = _ts_lag(x, 1)
    out = (x - prev) / (prev + 1e-9)
    if out.dim() == 2 and out.shape[0] > 0:
        out[0] = 0
    return _ts_nan_to_num(out)


@register_op("TS_MOM10", 1, "10-day momentum difference")
def op_ts_mom10(x):
    if x.dim() != 2 or x.shape[0] <= 10:
        return torch.zeros_like(x)
    prev = _ts_lag(x, 10)
    out = x - prev
    out[:10] = 0
    return _ts_nan_to_num(out)


@register_op("TS_MOM20", 1, "20-day momentum difference")
def op_ts_mom20(x):
    if x.dim() != 2 or x.shape[0] <= 20:
        return torch.zeros_like(x)
    prev = _ts_lag(x, 20)
    out = x - prev
    out[:20] = 0
    return _ts_nan_to_num(out)


@register_op("TS_MEAN5", 1, "5-day rolling mean")
def op_ts_mean5(x):
    return _ts_rolling_reduce(x, 5, lambda u: u.mean(dim=-1))


@register_op("TS_STD5", 1, "5-day rolling std")
def op_ts_std5(x):
    return _ts_rolling_reduce(x, 5, lambda u: u.std(dim=-1, unbiased=True))


@register_op("TS_STD20", 1, "20-day rolling std")
def op_ts_std20(x):
    return _ts_rolling_reduce(x, 20, lambda u: u.std(dim=-1, unbiased=True))


@register_op("TS_STD60", 1, "60-day rolling std")
def op_ts_std60(x):
    return _ts_rolling_reduce(x, 60, lambda u: u.std(dim=-1, unbiased=True))


@register_op("TS_MAX20", 1, "20-day rolling max")
def op_ts_max20(x):
    return _ts_rolling_reduce(x, 20, lambda u: u.max(dim=-1).values)


@register_op("TS_MIN20", 1, "20-day rolling min")
def op_ts_min20(x):
    return _ts_rolling_reduce(x, 20, lambda u: u.min(dim=-1).values)


@register_op("TS_BIAS5", 1, "(x - MA5) / MA5")
def op_ts_bias5(x):
    if x.dim() != 2:
        return torch.zeros_like(x)
    t = x.shape[0]
    if t < 5:
        return torch.zeros_like(x)

    ma5 = _ts_rolling_reduce(x, 5, lambda u: u.mean(dim=-1))
    out = (x - ma5) / (ma5 + 1e-9)
    out[:4] = 0
    return _ts_nan_to_num(out)


# --- Cross-sectional operators ---
@register_op("CS_RANK", 1, "cross-sectional rank")
def op_cs_rank(x):
    if x.dim() == 2:
        ranks = x.argsort(dim=1).argsort(dim=1).float()
        n = x.shape[1]
        return ranks / (n - 1 + 1e-9)
    return x


@register_op("CS_DEMEAN", 1, "cross-sectional demean")
def op_cs_demean(x):
    if x.dim() == 2:
        mean = x.mean(dim=1, keepdim=True)
        return x - mean
    return x


@register_op("CS_ROBUST_Z", 1, "cross-sectional robust z-score")
def op_cs_robust_z(x):
    if x.dim() == 2:
        median = x.median(dim=1, keepdim=True).values
        mad = (x - median).abs().median(dim=1, keepdim=True).values + 1e-9
        z = (x - median) / (mad * 1.4826)
        return torch.clamp(z, -5, 5)
    return x


# --- Logical / gating operators ---
@register_op("MAX", 2, "elementwise max")
def op_max(a, b):
    return torch.maximum(a, b)


@register_op("MIN", 2, "elementwise min")
def op_min(a, b):
    return torch.minimum(a, b)


@register_op("IF_POS", 2, "if a>0 then a else b")
def op_if_pos(a, b):
    return torch.where(a > 0, a, b)


@register_op("CUT_NEG", 1, "clamp negatives to 0")
def op_cut_neg(x):
    return torch.clamp(x, min=0)


@register_op("CUT_HIGH", 1, "penalize large values")
def op_cut_high(x):
    return torch.where(x > 2, torch.tensor(-1e9, device=x.device), x)
