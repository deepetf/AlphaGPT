import torch
from .ops_registry import OpsRegistry
from .config import ModelConfig
from typing import List, Optional


class StackVM:
    """
    栈式虚拟机 - 执行 RPN (逆波兰表达式) 格式的因子公式
    
    只接受字符串列表: ["CLOSE", "VOL", "DIV"]
    """
    def __init__(self):
        # 构建算子名称到函数的映射
        self.ops_map = {}
        self.arity_map = {}
        for name, func, arity in OpsRegistry.get_ops_config():
            self.ops_map[name] = func
            self.arity_map[name] = arity
        
        # 特征列表
        self.features = ModelConfig.INPUT_FEATURES
        self._cs_ops = {"CS_RANK", "CS_DEMEAN", "CS_ROBUST_Z"}

    def execute(self, formula: List[str], feat_tensor, cs_mask: Optional[torch.Tensor] = None):
        """
        执行公式
        
        Args:
            formula: 公式字符串列表 (如 ["CLOSE", "NEG"])
            feat_tensor: [Time, Assets, Features] 格式的输入张量
        
        Returns:
            result: [Time, Assets] 的因子值张量，或 None (如果公式非法)
        """
        # 强制要求字符串列表，不再兼容整数 token
        if not formula or not isinstance(formula[0], str):
            raise TypeError("formula must be a list of strings, not token IDs")
        
        stack = []
        try:
            for token_name in formula:
                if token_name in self.features:
                    # Token 是特征名
                    feat_idx = self.features.index(token_name)
                    stack.append(feat_tensor[:, :, feat_idx])
                    
                elif token_name in self.ops_map:
                    # Token 是算子名
                    arity = self.arity_map[token_name]
                    if len(stack) < arity:
                        return None  # 栈中参数不足
                    
                    # 弹出参数
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()  # 恢复参数顺序
                    
                    # 执行算子
                    if token_name in self._cs_ops:
                        if cs_mask is None:
                            raise ValueError(f"cross-sectional op '{token_name}' requires cs_mask")
                        res = self._apply_masked_cs(token_name, args[0], cs_mask)
                    else:
                        func = self.ops_map[token_name]
                        res = func(*args)
                    
                    stack.append(res)
                else:
                    # 未知 Token
                    return None
            
            # 公式执行完毕后，栈中应该只剩一个结果
            if len(stack) == 1:
                return stack[0]
            else:
                return None
                
        except (TypeError, ValueError):
            raise
        except Exception:
            return None

    def _resolve_cs_mask(self, x: torch.Tensor, cs_mask: torch.Tensor) -> torch.Tensor:
        """
        将 cs_mask 统一为与 x 对齐的 bool 掩码 [Time, Assets]。
        支持:
        - [Assets]：扩展到所有时点
        - [Time, Assets]：直接使用

        运行时真实 CS 掩码 = 基础 universe mask × 当前 operand 有效性。
        """
        if not isinstance(cs_mask, torch.Tensor):
            raise TypeError(f"cs_mask must be torch.Tensor, got {type(cs_mask)}")

        mask = cs_mask.to(device=x.device, dtype=torch.bool)
        if x.dim() != 2:
            raise ValueError(f"masked CS operators require 2D tensor [T, A], got {tuple(x.shape)}")

        t, a = x.shape
        if mask.dim() == 1:
            if mask.numel() != a:
                raise ValueError(f"cs_mask length mismatch: mask={mask.numel()}, assets={a}")
            mask = mask.unsqueeze(0).expand(t, a)
        elif mask.dim() == 2:
            if tuple(mask.shape) != (t, a):
                raise ValueError(f"cs_mask shape mismatch: mask={tuple(mask.shape)}, x={(t, a)}")
        else:
            raise ValueError(f"cs_mask must be 1D/2D, got dim={mask.dim()}")
        return mask & torch.isfinite(x)

    def _apply_masked_cs(self, op_name: str, x: torch.Tensor, cs_mask: torch.Tensor) -> torch.Tensor:
        mask = self._resolve_cs_mask(x, cs_mask)

        if op_name == "CS_RANK":
            return self._masked_cs_rank(x, mask)
        if op_name == "CS_DEMEAN":
            return self._masked_cs_demean(x, mask)
        if op_name == "CS_ROBUST_Z":
            return self._masked_cs_robust_z(x, mask)

        # 理论上不会到这里，保留兜底
        return self.ops_map[op_name](x)

    @staticmethod
    def _masked_cs_rank(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(x, float("nan"))
        t, _ = x.shape
        eps = 1e-9
        for i in range(t):
            row_mask = mask[i]
            n = int(row_mask.sum().item())
            if n <= 1:
                continue
            row = x[i, row_mask]
            order = torch.argsort(row)
            ranks = torch.empty(n, device=x.device, dtype=torch.float32)
            ranks[order] = torch.arange(n, device=x.device, dtype=torch.float32)
            out[i, row_mask] = ranks / (n - 1 + eps)
        return out

    @staticmethod
    def _masked_cs_demean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(x, float("nan"))
        t, _ = x.shape
        for i in range(t):
            row_mask = mask[i]
            if not torch.any(row_mask):
                continue
            row = x[i, row_mask]
            out[i, row_mask] = row - row.mean()
        return out

    @staticmethod
    def _masked_cs_robust_z(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(x, float("nan"))
        t, _ = x.shape
        for i in range(t):
            row_mask = mask[i]
            if not torch.any(row_mask):
                continue
            row = x[i, row_mask]
            median = row.median()
            mad = (row - median).abs().median() + 1e-9
            z = (row - median) / (mad * 1.4826)
            out[i, row_mask] = torch.clamp(z, -5.0, 5.0)
        return out
