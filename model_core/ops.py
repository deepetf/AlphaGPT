"""
算子模块 (Operators)

为了向后兼容，保留 OPS_CONFIG 变量。
新代码应使用 ops_registry.py 的注册机制。

使用示例:
    from model_core.ops import get_ops_config
    config = get_ops_config()
"""
from .ops_registry import OpsRegistry


def get_ops_config():
    """
    获取所有已注册算子的配置列表
    
    Returns:
        List of (name, func, arity) tuples
    """
    return OpsRegistry.get_ops_config()


# 向后兼容: 旧代码可能直接 import OPS_CONFIG
# 注意: 这是一个动态属性，在 ops_registry 加载后才有值
OPS_CONFIG = get_ops_config()