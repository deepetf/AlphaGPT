import torch
import os

# 尝试导入 Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

class ModelConfig:
    # 设备检测: Intel XPU > NVIDIA CUDA > CPU
    if HAS_IPEX and torch.xpu.is_available():
        DEVICE = torch.device("xpu")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    # DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    CB_PARQUET_PATH = r"C:\Trading\Projects\AlphaGPT\data\cb_data.pq"
    BATCH_SIZE = 512
    TRAIN_STEPS = 500
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    MIN_SCORE_IMPROVEMENT = 1e-4 # New King 最小提升阈值
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
        ('TURNOVER', 'turnover', 'ffill'),    # 转债市占比
        
        
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
    
    # AlphaGPT 输入特征 (这些会成为公式的"原子")
    # 从 BASIC_FACTORS 中选取用于模型输入的特征

    INPUT_FEATURES = ['CLOSE', 'VOL', 'PREM', 'DBLOW','REMAIN_SIZE','PCT_CHG','PCT_CHG_5','VOLATILITY_STK','PCT_CHG_STK','PCT_CHG_5_STK','PURE_VALUE','ALPHA_PCT_CHG_5','CAP_MV_RATE','TURNOVER', 'IV', 'VOL_STK_60', 'PREM_Z']
    
    # 动态计算 INPUT_DIM
    INPUT_DIM = len(INPUT_FEATURES)


class RobustConfig:
    """
    稳健性增强配置 (Robustness Enhancement Config)
    
    控制分段验证、稳定性惩罚、回撤惩罚、可交易性约束等参数。
    """
    # ========== 分段验证 (Split Validation) ==========
    TRAIN_TEST_SPLIT_DATE = '2024-06-01'  # 训练/验证切分日期
    
    # ========== 滚动稳定性 (Rolling Stability) ==========
    ROLLING_WINDOW = 60       # 滚动窗口天数
    STABILITY_K = 1.5         # 稳定性系数: Stability = Mean - K * Std
    
    # ========== 硬淘汰阈值 (Hard Thresholds) ==========
    MIN_SHARPE_VAL = 0.2      # 验证集最低 Sharpe，低于此直接淘汰
    MIN_ACTIVE_RATIO = 0.5    # 最低持仓满足率 (实际持仓数 / top_k)
    MIN_VALID_DAYS = 20       # 最少有效交易天数
    MIN_VALID_DAYS = 20       # 最少有效交易天数
    MIN_VALID_COUNT = 30      # 实盘最少有效标的数量 (熔断阈值)
    TOP_K = 10                # 策略选股数量
    
    # ========== 软评分权重 (Soft Scoring Weights) ==========
    # 基础分权重
    TRAIN_WEIGHT = 0.4        # 训练集 Sharpe 占比
    VAL_WEIGHT = 0.6          # 验证集 Sharpe 占比
    
    # 惩罚/奖励项权重
    STABILITY_W = 0.5         # 稳定性得分权重 (正向加分)
    MDD_W = 20.0              # 回撤惩罚权重 (例: 0.3 MDD -> 6 分扣除)
    LEN_W = 0.2               # 长度惩罚权重 (略微降低，让位给稳健性)
    
    # 总分缩放
    SCALE = 5.0               # 最终分数缩放系数