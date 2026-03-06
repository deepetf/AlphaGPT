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
    def METRIC_FAIL_REWARD_MODE(cls) -> str:
        # hard: 旧版 clamp; soft: 连续 fail reward
        return str(cls._rc.get('metric_fail_reward_mode', 'hard')).lower()

    @property
    def METRIC_GAP_W(cls) -> float:
        # fail gap 惩罚权重
        return cls._rc.get('metric_gap_w', 8.0)

    @property
    def METRIC_FAIL_REWARD_CAP(cls) -> float:
        # fail reward 上限，防止失败样本变正激励
        return cls._rc.get('metric_fail_reward_cap', 0.0)

    @property
    def METRIC_FAIL_REWARD_FLOOR(cls) -> float:
        # fail reward 下限，防止极值爆点
        return cls._rc.get('metric_fail_reward_floor', -10.0)

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
    def DECODE_TS_DENSITY_SOFT_ENABLED(cls) -> bool:
        return cls._rc.get('decode_ts_density_soft_enabled', True)

    @property
    def DECODE_TS_DENSITY_PENALTY_L1(cls) -> float:
        return cls._rc.get('decode_ts_density_penalty_l1', 0.3)

    @property
    def DECODE_TS_DENSITY_PENALTY_L2(cls) -> float:
        return cls._rc.get('decode_ts_density_penalty_l2', 0.8)

    @property
    def DECODE_TS_DENSITY_PENALTY_L3(cls) -> float:
        return cls._rc.get('decode_ts_density_penalty_l3', 1.5)

    @property
    def DECODE_REACHABILITY_ENABLED(cls) -> bool:
        return cls._rc.get('decode_reachability_enabled', True)

    @property
    def DECODE_LOOKAHEAD_ENABLED(cls) -> bool:
        return cls._rc.get('decode_lookahead_enabled', True)
    
    @property
    def FEE_RATE(cls) -> float:
        return cls._rc.get('fee_rate', 0.0005)

    @property
    def TAKE_PROFIT(cls) -> float:
        # 止盈涨幅阈值，0 表示不止盈
        return cls._rc.get('take_profit', 0.0)

    @property
    def MIN_VALID_COUNT(cls) -> int:
        return cls._rc.get('min_valid_count', 30)

    @property
    def SIGNAL_CLEAN_ENABLED(cls) -> bool:
        return cls._rc.get('signal_clean_enabled', True)

    @property
    def SIGNAL_WINSOR_Q(cls) -> float:
        return cls._rc.get('signal_winsor_q', 0.01)

    @property
    def SIGNAL_CLIP(cls) -> float:
        return cls._rc.get('signal_clip', 5.0)

    @property
    def SIGNAL_RANK_OUTPUT(cls) -> bool:
        return cls._rc.get('signal_rank_output', True)

    @property
    def SIGNAL_MIN_VALID_COUNT(cls) -> int:
        return cls._rc.get('signal_min_valid_count', cls.MIN_VALID_COUNT)

    @property
    def SIM_MASKED_CS_ENABLED(cls) -> bool:
        return cls._rc.get('sim_masked_cs_enabled', True)

    @property
    def SIM_CS_REQUIRE_PRESENT(cls) -> bool:
        return cls._rc.get('sim_cs_require_present', True)

    @property
    def SIM_EXEC_PRICE_RAW_PRIORITY(cls) -> bool:
        return cls._rc.get('sim_exec_price_raw_priority', True)

    @property
    def REWARD_STD_FLOOR(cls) -> float:
        return cls._rc.get('reward_std_floor', 0.12)

    @property
    def REWARD_STD_PATIENCE(cls) -> int:
        return cls._rc.get('reward_std_patience', 20)

    @property
    def STAGNATION_PATIENCE(cls) -> int:
        return cls._rc.get('stagnation_patience', 80)

    @property
    def STAGNATION_ENTROPY_BOOST(cls) -> float:
        return cls._rc.get('stagnation_entropy_boost', 0.02)

    @property
    def COLLAPSE_ENTROPY_BOOST(cls) -> float:
        return cls._rc.get('collapse_entropy_boost', 0.015)

    @property
    def MIN_SCORE_IMPROVEMENT(cls) -> float:
        return cls._rc.get('min_score_improvement', 0.01)

    @property
    def ENTROPY_CONTROLLER_MODE(cls) -> str:
        return str(cls._rc.get('entropy_controller_mode', 'hard')).lower()

    @property
    def ENTROPY_LOCK_BETA(cls) -> float:
        return cls._rc.get('entropy_lock_beta', 0.06)

    @property
    def ENTROPY_LOCK_STEPS(cls) -> int:
        return cls._rc.get('entropy_lock_steps', 10)

    @property
    def ENTROPY_WARN_BOOST(cls) -> float:
        return cls._rc.get('entropy_warn_boost', 0.01)

    @property
    def CONTROLLER_HARD_FLOOR_RATE(cls) -> float:
        return cls._rc.get('controller_hard_floor_rate', 0.01)

    @property
    def CONTROLLER_HARD_FLOOR_ABS(cls) -> float:
        return cls._rc.get('controller_hard_floor_abs', 1.0)

    @property
    def CONTROLLER_HARD_WARN_RATE(cls) -> float:
        return cls._rc.get('controller_hard_warn_rate', 0.05)

    @property
    def CONTROLLER_METRIC_FLOOR_RATE(cls) -> float:
        return cls._rc.get('controller_metric_floor_rate', 0.002)

    @property
    def CONTROLLER_METRIC_FLOOR_ABS(cls) -> float:
        return cls._rc.get('controller_metric_floor_abs', 1.0)

    @property
    def CONTROLLER_METRIC_WARN_RATE(cls) -> float:
        return cls._rc.get('controller_metric_warn_rate', 0.01)

    @property
    def CONTROLLER_SIM_FLOOR_RATE(cls) -> float:
        return cls._rc.get('controller_sim_floor_rate', 0.002)

    @property
    def CONTROLLER_SIM_FLOOR_ABS(cls) -> float:
        return cls._rc.get('controller_sim_floor_abs', 1.0)

    @property
    def CONTROLLER_SIM_WARN_RATE(cls) -> float:
        return cls._rc.get('controller_sim_warn_rate', 0.01)

    @property
    def CONTROLLER_POOL_FLOOR_RATE(cls) -> float:
        return cls._rc.get('controller_pool_floor_rate', 0.1)

    @property
    def CONTROLLER_POOL_FLOOR_ABS(cls) -> float:
        return cls._rc.get('controller_pool_floor_abs', 0.5)

    @property
    def CONTROLLER_POOL_WARN_RATE(cls) -> float:
        return cls._rc.get('controller_pool_warn_rate', 0.2)

    @property
    def CONTROLLER_POOL_STAGNATION_PATIENCE(cls) -> int:
        return cls._rc.get('controller_pool_stagnation_patience', 6)

    @property
    def ADV_NOISE_STD(cls) -> float:
        return cls._rc.get('adv_noise_std', 0.05)


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

    @classproperty
    def LR(cls):
        return cls._get_conf().get('lr', 1e-4)

    @classproperty
    def WEIGHT_DECAY(cls):
        return cls._get_conf().get('weight_decay', 0.01)

    @classproperty
    def GRAD_CLIP_NORM(cls):
        return cls._get_conf().get('grad_clip_norm', 1.0)

    @classproperty
    def GRAD_NORM_LOG_INTERVAL(cls):
        return cls._get_conf().get('grad_norm_log_interval', 20)

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

