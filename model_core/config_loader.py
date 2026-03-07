"""
配置加载模块

提供动态加载 YAML 配置文件的功能，支持通过命令行参数覆盖默认配置。
"""
import os
import yaml
from typing import Optional, Dict, Any

# 全局配置缓存
_config_cache: Optional[Dict[str, Any]] = None
_loaded_config_path: Optional[str] = None

# 默认配置文件路径
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'default_config.yaml')


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数:
        config_path: 自定义配置文件路径，如果为 None 则使用默认配置
        
    返回:
        合并后的配置字典
        
    说明:
        - 首先加载默认配置
        - 如果提供了自定义配置路径，则用自定义配置覆盖默认配置
        - 配置会被缓存，后续调用 get_config() 获取
    """
    global _config_cache
    
    # 1. 加载默认配置
    if not os.path.exists(_DEFAULT_CONFIG_PATH):
        raise FileNotFoundError(f"默认配置文件不存在: {_DEFAULT_CONFIG_PATH}")
    
    with open(_DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 如果提供了自定义配置，则合并
    if config_path:
        target_path = config_path
        if not os.path.exists(target_path):
            # 尝试相对于模块目录查找
            module_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(module_dir, config_path)
            if os.path.exists(potential_path):
                target_path = potential_path
            else:
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(target_path, 'r', encoding='utf-8') as f:
            custom_config = yaml.safe_load(f)
        
        if custom_config:
            config = _deep_merge(config, custom_config)
    
    # 3. 验证配置
    _validate_config(config)
    
    # 4. 缓存配置
    _config_cache = config
    global _loaded_config_path
    _loaded_config_path = config_path  # 记录原始路径
    
    return config


def get_loaded_config_path() -> Optional[str]:
    """获取最后加载的自定义配置路径"""
    return _loaded_config_path


def get_config() -> Dict[str, Any]:
    """
    获取当前配置
    
    返回:
        配置字典，如果尚未加载则自动加载默认配置
    """
    global _config_cache
    
    if _config_cache is None:
        load_config()
    
    return _config_cache


def get_input_features() -> list:
    """
    获取输入特征列表
    
    返回:
        INPUT_FEATURES 列表
    """
    return get_config().get('input_features', [])


def get_robust_config() -> Dict[str, Any]:
    """
    获取稳健性配置
    
    返回:
        RobustConfig 字典
    """
    return get_config().get('robust_config', {})


def get_config_val(key: str, default: Any = None) -> Any:
    """
    获取顶层配置项
    """
    return get_config().get(key, default)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典
    
    override 中的值会覆盖 base 中的值，
    对于嵌套字典会递归合并。
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def _validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置的合法性
    
    检查必要字段是否存在，数值是否在合理范围内。
    """
    # 验证 input_features
    if 'input_features' not in config or not config['input_features']:
        raise ValueError("配置缺少 'input_features' 或列表为空")
    
    # 验证 robust_config
    if 'robust_config' not in config:
        raise ValueError("配置缺少 'robust_config'")
    
    batch_size = config.get('batch_size', 512)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    rc = config['robust_config']
    required_fields = [
        'train_test_split_date', 'rolling_window', 'stability_k',
        'min_sharpe_val', 'min_active_ratio', 'min_valid_days',
        'top_k', 'fee_rate', 'train_weight', 'val_weight',
        'stability_w', 'ret_w', 'mdd_w', 'len_w', 'scale',
        'entropy_beta_start', 'entropy_beta_end', 'diversity_pool_size'
    ]
    
    for field in required_fields:
        if field not in rc:
            raise ValueError(f"robust_config 缺少字段: {field}")
    
    # 验证数值范围
    if rc['top_k'] < 1:
        raise ValueError("top_k 必须 >= 1")
    
    if rc['fee_rate'] < 0 or rc['fee_rate'] > 0.1:
        raise ValueError("fee_rate 应在 0 ~ 0.1 范围内")
    
    if rc['train_weight'] + rc['val_weight'] != 1.0:
        raise ValueError("train_weight + val_weight 应等于 1.0")


def reset_config() -> None:
    """
    重置配置缓存
    
    用于测试或需要重新加载配置的场景。
    """
    global _config_cache
    _config_cache = None
