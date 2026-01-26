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
    TRAIN_STEPS = 100
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
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
        
    ]
    
    # AlphaGPT 输入特征 (这些会成为公式的"原子")
    # 从 BASIC_FACTORS 中选取用于模型输入的特征

    INPUT_FEATURES = ['CLOSE', 'VOL', 'PREM', 'DBLOW','REMAIN_SIZE','PCT_CHG','PCT_CHG_5','VOLATILITY_STK','PCT_CHG_STK','PCT_CHG_5_STK','PURE_VALUE','ALPHA_PCT_CHG_5','CAP_MV_RATE','TURNOVER']
    
    # 动态计算 INPUT_DIM
    INPUT_DIM = len(INPUT_FEATURES)