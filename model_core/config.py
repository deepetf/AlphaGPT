"""
AlphaGPT 配置模块

静态配置 (设备、数据路径、基础因子) 保留在此文件中。
动态配置 (INPUT_FEATURES, RobustConfig) 从 YAML 文件加载。
"""
import torch
import os

# 尝试导入 Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False


class ConfigMeta(type):
    """
    配置元类
    
    用于支持类属性的动态访问 (Class Property)。
    解决了 @property 只能用于实例而无法用于类本身的问题。
    """
    
    # ========== ModelConfig 动态属性 ==========
    @property
    def INPUT_FEATURES(cls) -> list:
        from .config_loader import get_input_features
        return get_input_features()
    
    @property
    def INPUT_DIM(cls) -> int:
        return len(cls.INPUT_FEATURES)
    
    @property
    def TRAIN_STEPS(cls) -> int:
        from .config_loader import get_config_val
        return get_config_val('train_steps', 100)

    # ========== RobustConfig 动态属性 ==========
    @property
    def _rc(cls):
        from .config_loader import get_robust_config
        return get_robust_config()

    @property
    def TRAIN_TEST_SPLIT_DATE(cls) -> str:
        return cls._rc.get('train_test_split_date', '2024-05-01')
    
    @property
    def ROLLING_WINDOW(cls) -> int:
        return cls._rc.get('rolling_window', 60)
    
    @property
    def STABILITY_K(cls) -> float:
        return cls._rc.get('stability_k', 1.5)
    
    @property
    def MIN_SHARPE_VAL(cls) -> float:
        return cls._rc.get('min_sharpe_val', 0.2)
    
    @property
    def MIN_ACTIVE_RATIO(cls) -> float:
        return cls._rc.get('min_active_ratio', 0.3)
    
    @property
    def MIN_VALID_DAYS(cls) -> int:
        return cls._rc.get('min_valid_days', 20)
    
    @property
    def MIN_VALID_COUNT(cls) -> int:
        return cls._rc.get('min_valid_count', 30)
    
    @property
    def TOP_K(cls) -> int:
        return cls._rc.get('top_k', 10)
    
    @property
    def FEE_RATE(cls) -> float:
        return cls._rc.get('fee_rate', 0.0005)
    
    @property
    def TRAIN_WEIGHT(cls) -> float:
        return cls._rc.get('train_weight', 0.4)
    
    @property
    def VAL_WEIGHT(cls) -> float:
        return cls._rc.get('val_weight', 0.6)
    
    @property
    def STABILITY_W(cls) -> float:
        return cls._rc.get('stability_w', 0.5)
    
    @property
    def RET_W(cls) -> float:
        return cls._rc.get('ret_w', 6.0)
    
    @property
    def MDD_W(cls) -> float:
        return cls._rc.get('mdd_w', 12.0)
    
    @property
    def LEN_W(cls) -> float:
        return cls._rc.get('len_w', 0.1)
    
    @property
    def SCALE(cls) -> float:
        return cls._rc.get('scale', 5.0)
    
    @property
    def ENTROPY_BETA_START(cls) -> float:
        return cls._rc.get('entropy_beta_start', 0.04)
    
    @property
    def ENTROPY_BETA_END(cls) -> float:
        return cls._rc.get('entropy_beta_end', 0.005)
    
    @property
    def DIVERSITY_POOL_SIZE(cls) -> int:
        return cls._rc.get('diversity_pool_size', 50)
    
    @property
    def JACCARD_THRESHOLD(cls) -> float:
        return cls._rc.get('jaccard_threshold', 0.8)
    
    @property
    def DENSITY_WINDOW(cls) -> int:
        return cls._rc.get('density_window', 6)
    
    @property
    def MAX_TS_IN_WINDOW(cls) -> int:
        return cls._rc.get('max_ts_in_window', 3)
    
    @property
    def DENSITY_PENALTY(cls) -> float:
        return cls._rc.get('density_penalty', -2.0)


class ModelConfig(metaclass=ConfigMeta):
    """
    模型静态配置
    
    包含与硬件、数据源相关的不可变配置。
    INPUT_FEATURES 等可调参数已移至 default_config.yaml。
    """
    # ========== 设备检测 ==========
    # Intel XPU > NVIDIA CUDA > CPU
    if HAS_IPEX and torch.xpu.is_available():
        DEVICE = torch.device("xpu")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    
    # ========== 数据路径 ==========
    CB_PARQUET_PATH = r"C:\Trading\Projects\AlphaGPT\data\cb_data.pq"
    
    # ========== 训练参数 ==========
    BATCH_SIZE = 512

    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0  # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005  # 基础费率 0.5% (Swap + Gas + Jito Tip)
    MIN_SCORE_IMPROVEMENT = 1e-4  # New King 最小提升阈值
    
    # ========== 灵活因子配置 ==========
    # 基础因子: (内部名称, Parquet列名, 填充方法)
    # 填充方法: 'ffill' = 前值填充(适合价格), 'zero' = 零填充(适合成交量)
    BASIC_FACTORS = [
        # 转债行情
        ('OPEN', 'open', 'ffill'),
        ('HIGH', 'high', 'ffill'),
        ('LOW', 'low', 'ffill'),
        ('CLOSE', 'close', 'ffill'),
        ('VOL', 'vol', 'zero'),
        ('AMOUNT', 'amount', 'zero'),
        # 可转债特有指标
        ('PREM', 'conv_prem', 'ffill'),       # 转股溢价率
        ('DBLOW', 'dblow', 'ffill'),          # 双低值
        ('YTM', 'ytm', 'ffill'),              # 到期收益率
        ('LEFT_YRS', 'left_years', 'ffill'),  # 剩余年限
        ('REMAIN_SIZE', 'remain_size', 'ffill'),  # 剩余规模
        ('PCT_CHG', 'pct_chg', 'ffill'),        # 涨跌幅
        ('PCT_CHG_5', 'pct_chg_5', 'ffill'),    # 五日涨跌幅
        ('PURE_VALUE', 'pure_value', 'ffill'),    # 纯债价值
        ('ALPHA_PCT_CHG_5', 'alpha_pct_chg_5', 'ffill'),    # 五日涨跌幅差
        ('CAP_MV_RATE', 'cap_mv_rate', 'ffill'),    # 转债市占比
        ('TURNOVER', 'turnover', 'ffill'),    # 换手率
        
        # 正股行情
        ('STK_CLOSE', 'close_stk', 'ffill'),
        ('STK_VOL', 'vol_stk', 'zero'),
        ('VOLATILITY_STK', 'volatility_stk', 'ffill'),  # 正股波动率
        ('PCT_CHG_STK', 'pct_chg_stk', 'ffill'),        # 涨跌幅
        ('PCT_CHG_5_STK', 'pct_chg_5_stk', 'ffill'),    # 五日涨跌幅
        
        # 新增因子 (Added via Request)
        ('IV', 'IV', 'ffill'),                        # 隐含波动率
        ('VOL_STK_60', 'stock_vol60d', 'ffill'),      # 正股60日波动率
        ('PREM_Z', 'convprem_zscore', 'ffill'),       # 溢价率 Z-Score
    ]
    
    # ========== 动态配置属性 ==========
    # INPUT_FEATURES 和 INPUT_DIM 现由 ConfigMeta 动态提供
    
    @classmethod
    def get_input_features(cls) -> list:
        """获取输入特征列表 (保留方法访问方式)"""
        return cls.INPUT_FEATURES
    
    @classmethod
    def get_input_dim(cls) -> int:
        """获取输入维度 (保留方法访问方式)"""
        return cls.INPUT_DIM


class RobustConfig(metaclass=ConfigMeta):
    """
    稳健性增强配置 (Robustness Enhancement Config)
    
    所有参数现在从 YAML 配置文件动态加载。
    通过 metaclss 实现参数作为类属性直接访问，如 RobustConfig.TOP_K。
    """
    pass