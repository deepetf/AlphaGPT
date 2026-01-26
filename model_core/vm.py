import torch
from .ops_registry import OpsRegistry
from .config import ModelConfig
from typing import List


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

    def execute(self, formula: List[str], feat_tensor):
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
                    func = self.ops_map[token_name]
                    res = func(*args)
                    
                    # 处理异常值
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    stack.append(res)
                else:
                    # 未知 Token
                    return None
            
            # 公式执行完毕后，栈中应该只剩一个结果
            if len(stack) == 1:
                return stack[0]
            else:
                return None
                
        except Exception:
            return None