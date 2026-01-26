import pandas as pd
import torch
import numpy as np
from .config import ModelConfig
from .factors import FeatureEngineer

class CBDataLoader:
    def __init__(self):
        self.raw_data_cache = None
        self.feat_tensor = None
        self.target_ret = None
        self.valid_mask = None # [Time, Assets] 标记是否可交易，用于过滤退市/停牌
        
    def load_data(self):
        print(f"Loading Parquet from: {ModelConfig.CB_PARQUET_PATH}")
        
        # 1. Load Parquet
        try:
            df = pd.read_parquet(ModelConfig.CB_PARQUET_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet file: {e}")

        print(f"Parquet columns: {list(df.columns)[:20]}...")
        print(f"Parquet shape: {df.shape}")
        print(f"Parquet index names: {df.index.names}")
        
        # 如果 code/trade_date 在 index 中，需要 reset_index
        if 'code' not in df.columns or 'trade_date' not in df.columns:
            print("Resetting index to access code/trade_date...")
            df = df.reset_index()
            print(f"After reset - columns: {list(df.columns)[:10]}...")

        # Ensure datetime index
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
        # Filter by Date (2022-08-01 ~ Now)
        start_date = '2022-08-01'
        print(f"Filtering data from {start_date}...")
        df = df[df['trade_date'] >= start_date]
        print(f"Filtered shape: {df.shape}")
        
        # 2. Pivot to Wide Format [Time, Assets]
        # 使用配置驱动的因子列表
        raw_tensors = {}
        assets = df['code'].unique()
        assets = sorted(assets)
        self.assets_list = assets
        
        print(f"Pivoting data for {len(assets)} assets...")
        
        # Pivot helper - 一次性 pivot 所有列
        pivot_df = df.pivot(index='trade_date', columns='code')
        
        # 保存日期列表 (用于交易细节输出)
        self.dates_list = pivot_df.index.strftime('%Y-%m-%d').tolist()
        
        # 构建 code -> name 映射 (用于交易细节输出)
        if 'name' in df.columns:
            name_df = df[['code', 'name']].drop_duplicates()
            self.names_dict = dict(zip(name_df['code'], name_df['name']))
        else:
            print("Warning: 'name' column not found, using code as name.")
            self.names_dict = {code: code for code in assets}
        
        for internal_name, parquet_col, fill_method in ModelConfig.BASIC_FACTORS:
            if parquet_col not in df.columns:
                print(f"Warning: Column '{parquet_col}' not found in parquet, skipping '{internal_name}'.")
                continue
                
            # Extract sub-dataframe
            sub_df = pivot_df[parquet_col].copy()
            
            # Fill NaN based on method
            if fill_method == 'ffill':
                sub_df = sub_df.ffill()  # pandas 2.x 兼容
            elif fill_method == 'zero':
                sub_df = sub_df.fillna(0.0)
            
            # Convert to Tensor [Time, Assets]
            # fillna(0) for remaining NaNs (e.g. before listing)
            data_np = sub_df.fillna(0.0).values.astype(np.float32)
            raw_tensors[internal_name] = torch.tensor(data_np, device=ModelConfig.DEVICE)
        
        self.raw_data_cache = raw_tensors
        print(f"Loaded {len(raw_tensors)} factors: {list(raw_tensors.keys())}")
        
        # 3. Construct Valid Mask (Filter)
        # 必须同时满足: 有收盘价(已上市) & 成交量>0(没停牌) & 剩余年限>0.5年(排除临期债)
        has_price = raw_tensors['CLOSE'] > 0
        is_trading = raw_tensors['VOL'] > 0
        not_expiring = raw_tensors['LEFT_YRS'] > 0.5  # 排除剩余年限不足半年的转债
        self.valid_mask = has_price & is_trading & not_expiring
        print(f"Valid mask: {self.valid_mask.sum().item()} / {self.valid_mask.numel()} valid samples")
        
        # 4. Compute Features (Feature Engineer)
        print("Computing features...")
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        
        # 5. Compute Target Return (T+1)
        # 我们预测的是 T+1 的收益率 (close_T+1 / close_T - 1)
        close = raw_tensors['CLOSE']
        ret_1d = (torch.roll(close, -1, dims=0) / (close + 1e-9)) - 1.0
        # 对无效数据的收益率置 0
        ret_1d[~self.valid_mask] = 0.0
        # 最后一行也是无效的
        ret_1d[-1] = 0.0
        
        self.target_ret = ret_1d
        
        print(f"Data Ready. Tensor Shape: {self.feat_tensor.shape}")