import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # PostgreSQL 配置 (原有)
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "crypto_quant")
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # 数据源配置: "api" | "local_db"
    DATA_SOURCE = os.getenv("DATA_SOURCE", "api")
    
    # CB_HISTORY MySQL 配置 (本地数据库)
    CB_DB_HOST = os.getenv("CB_DB_HOST", "192.168.8.78")
    CB_DB_PORT = os.getenv("CB_DB_PORT", "3306")
    CB_DB_USER = os.getenv("CB_DB_USER", "root")
    CB_DB_PASSWORD = os.getenv("CB_DB_PASSWORD", "Happy$4ever")
    CB_DB_NAME = os.getenv("CB_DB_NAME", "CB_HISTORY")
    CB_DB_DSN = f"mysql+pymysql://{CB_DB_USER}:{CB_DB_PASSWORD}@{CB_DB_HOST}:{CB_DB_PORT}/{CB_DB_NAME}"
    
    # API 配置 (原有)
    CHAIN = "solana"
    TIMEFRAME = "1m" # 也支持 15min
    MIN_LIQUIDITY_USD = 500000.0  
    MIN_FDV = 10000000.0            
    MAX_FDV = float('inf') 
    BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
    BIRDEYE_IS_PAID = True
    USE_DEXSCREENER = False
    CONCURRENCY = 20
    HISTORY_DAYS = 7
