import os
import torch

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

class ConfigMeta(type):
    """
    元类：支持动态从 config_loader 获取最新的配置值
    """
    _loader = None
    
    @property
    def _rc(cls):
        if cls._loader is None:
            from .config_loader import get_config
            cls._loader = get_config
        return cls._loader().get('robust_config', {})

    @property
    def PENALTY_FLIP_COEF(cls) -> float:
        return cls._rc.get('penalty_flip_coef', 2.0)
    
    @property
    def PENALTY_STRUCT(cls) -> float:
        return cls._rc.get('penalty_struct', -8.0)

    @property
    def PENALTY_EXEC(cls) -> float:
        return cls._rc.get('penalty_exec', -10.0)

    @property
    def PENALTY_LOWVAR(cls) -> float:
        return cls._rc.get('penalty_lowvar', -6.5)

    @property
    def PENALTY_METRIC_MAX(cls) -> float:
        # 指标失败惩罚上限 (最好情况)
        return cls._rc.get('penalty_metric_max', -4.0)

    @property
    def PENALTY_METRIC_MIN(cls) -> float:
        # 指标失败惩罚下限 (最坏情况)
        return cls._rc.get('penalty_metric_min', -6.0)

    @property
    def PENALTY_SIM(cls) -> float:
        # 相似度拒绝惩罚
        return cls._rc.get('penalty_sim', -1.5)

    @property
    def MAX_STACK_DEPTH(cls) -> int:
        return cls._rc.get('max_stack_depth', 7)

    @property
    def CACHE_MAX_SIZE(cls) -> int:
        # 会话级缓存上限，防止内存膨胀
        return cls._rc.get('cache_max_size', 100000)

    # --- [V4.1 Fix] 恢复所有缺失的 RobustConfig 属性 ---
    @property
    def TRAIN_TEST_SPLIT_DATE(cls) -> str:
        return cls._rc.get('train_test_split_date', "2024-05-01")

    @property
    def TOP_K(cls) -> int:
        return cls._rc.get('top_k', 10)

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
        return cls._rc.get('ret_w', 8.0)

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
        return cls._rc.get('entropy_beta_end', 0.015)

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
        return cls._rc.get('max_ts_in_window', 4)

    @property
    def DENSITY_PENALTY(cls) -> float:
        return cls._rc.get('density_penalty', -2.0)
    
    @property
    def FEE_RATE(cls) -> float:
        return cls._rc.get('fee_rate', 0.0005)

    @property
    def TAKE_PROFIT(cls) -> float:
        # 止盈涨幅阈值，0 表示不止盈
        return cls._rc.get('take_profit', 0.0)


class RobustConfig(metaclass=ConfigMeta):
    """
    稳健性评估配置 (通过元类动态获取)
    """
    pass


class ModelConfig:
    """
    模型静态配置 (部分由 config_loader 动态覆盖)
    """
    _loader = None
    
    @classmethod
    def _get_conf(cls):
        if cls._loader is None:
            from .config_loader import get_config
            cls._loader = get_config
        return cls._loader()

    # 原有的静态部分
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 512

    # RPN 最大 Token 长度 (对外兼容字段名)
    MAX_FORMULA_LEN = 15
    # 一些历史代码使用 MAX_LEN
    MAX_LEN = MAX_FORMULA_LEN

    # 动态属性 (通过 property 或类方法保持兼容)
    @classproperty
    def INPUT_FEATURES(cls):
        return cls._get_conf().get('input_features', [])


    @classproperty
    def INPUT_DIM(cls):
        return len(cls.INPUT_FEATURES)

    @classproperty
    def TRAIN_STEPS(cls):
        return cls._get_conf().get('train_steps', 2000)

    @classproperty
    def FEE_RATE(cls):
        return cls._get_conf().get('fee_rate', 0.0003)

    @classproperty
    def WARMUP_DAYS(cls):
        """预热天数（自然日），加载训练起始日前的额外数据用于特征标准化预热"""
        return cls._get_conf().get('warmup_days', 85)

    # 训练/评估比例
    TRAIN_RATIO = 0.7     # 70% 训练, 30% 验证
    
    # 网络超参
    D_MODEL = 256
    N_HEAD = 8
    N_LAYER = 4
    LR = 1e-4

    @classproperty
    def CB_PARQUET_PATH(cls):
        # 默认使用项目标准数据路径
        return cls._get_conf().get('cb_parquet_path', r"C:\Trading\Projects\AlphaGPT\data\cb_data.pq")

    # 基础因子定义 (InternalName, ParquetColumn, FillMethod)
    # 这是数据加载器解析 Parquet 文件的字典
    BASIC_FACTORS = [
        ('CLOSE', 'close', 'ffill'),
        ('OPEN', 'open', 'ffill'),       # 开盘价（用于止盈）
        ('HIGH', 'high', 'ffill'),       # 最高价（用于止盈）
        ('VOL', 'vol', 'zero'),
        ('PREM', 'conv_prem', 'ffill'),
        ('DBLOW', 'dblow', 'ffill'),
        ('REMAIN_SIZE', 'remain_size', 'ffill'),
        ('PCT_CHG', 'pct_chg', 'zero'),
        ('PCT_CHG_5', 'pct_chg_5', 'zero'),
        ('VOLATILITY_STK', 'volatility_stk', 'ffill'),
        ('PCT_CHG_STK', 'pct_chg_stk', 'zero'),
        ('PCT_CHG_5_STK', 'pct_chg_5_stk', 'zero'),
        ('CLOSE_STK', 'close_stk', 'ffill'),
        ('CONV_PRICE', 'conv_price', 'ffill'),
        ('PURE_VALUE', 'pure_value', 'ffill'),
        ('ALPHA_PCT_CHG_5', 'alpha_pct_chg_5', 'zero'),
        ('CAP_MV_RATE', 'cap_mv_rate', 'ffill'),
        ('TURNOVER', 'turnover', 'zero'),
        ('IV', 'IV', 'ffill'),
        ('VOL_STK_60', 'stock_vol60d', 'ffill'),
        ('PREM_Z', 'convprem_zscore', 'ffill'),
        ('LEFT_YRS', 'left_years', 'ffill') # 必须有，用于过滤临期债
    ]

    # 进化控制
    MIN_SCORE_IMPROVEMENT = 0.01

