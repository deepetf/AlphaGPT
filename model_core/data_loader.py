import pandas as pd
import torch
import numpy as np
from .config import ModelConfig, RobustConfig
from .factors import FeatureEngineer

class CBDataLoader:
    def __init__(self):
        self.raw_data_cache = None
        self.feat_tensor = None
        self.target_ret = None
        self.valid_mask = None  # [Time, Assets] 兼容字段，当前等同于 tradable_mask
        self.listed_mask = None
        self.data_mask = None
        self.tradable_mask = None
        self.cs_mask = None
        self.feature_valid_tensor = None
        self.split_idx = None   # 训练/验证切分索引 (基于 RobustConfig)
        
    def load_data(self, start_date: str = '2022-08-01'):
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

        # 日期范围校验
        raw_min_date = df['trade_date'].min()
        raw_max_date = df['trade_date'].max()
        if pd.isna(raw_min_date) or pd.isna(raw_max_date):
            raise RuntimeError("trade_date contains no valid datetime values")

        requested_start = pd.to_datetime(start_date)
        effective_start = requested_start
        if requested_start < raw_min_date:
            print(
                f"Warning: requested start_date {requested_start.strftime('%Y-%m-%d')} "
                f"earlier than data min {raw_min_date.strftime('%Y-%m-%d')}, clamped to data min."
            )
            effective_start = raw_min_date
        if requested_start > raw_max_date:
            raise ValueError(
                f"start_date {requested_start.strftime('%Y-%m-%d')} is later than data max "
                f"{raw_max_date.strftime('%Y-%m-%d')}"
            )

        # --- 预热前推 (Warmup Pre-loading) ---
        # 为 _robust_normalize 的滚动窗口提供足够的前置真实数据，
        # 避免训练首日的特征全部为 0。
        warmup_days = ModelConfig.WARMUP_DAYS
        if warmup_days > 0:
            load_start = effective_start - pd.Timedelta(days=warmup_days)
            load_start = max(load_start, raw_min_date)  # 不超出数据范围
            actual_warmup = (effective_start - load_start).days
            if actual_warmup < warmup_days:
                print(
                    f"Warning: warmup clamped to {actual_warmup} natural days "
                    f"(requested {warmup_days}), data starts at {raw_min_date.strftime('%Y-%m-%d')}"
                )
        else:
            load_start = effective_start

        print(
            f"Filtering data from {load_start.strftime('%Y-%m-%d')} "
            f"(effective_start={effective_start.strftime('%Y-%m-%d')}, "
            f"warmup_days={warmup_days}, "
            f"raw_range=[{raw_min_date.strftime('%Y-%m-%d')}, {raw_max_date.strftime('%Y-%m-%d')}])..."
        )
        df = df[df['trade_date'] >= load_start]
        print(f"Filtered shape: {df.shape}")
        
        # 2. Pivot to Wide Format [Time, Assets]
        # 使用配置驱动的因子列表
        raw_tensors = {}
        assets = df['code'].unique()
        assets = sorted(assets)
        # assets_list 在资产池清理后再赋值（见下方 Asset Cleanup）
        
        print(f"Pivoting data for {len(assets)} assets...")
        
        # Pivot helper - 一次性 pivot 所有列
        pivot_df = df.pivot(index='trade_date', columns='code')
        
        # 全范围日期列表（含预热段）
        all_dates_list = pivot_df.index.strftime('%Y-%m-%d').tolist()
        
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
            
            # Convert to Tensor [Time, Assets]
            # 缺失值与上市前空值保留为 NaN，避免在主链路中伪装成有效 0
            data_np = sub_df.values.astype(np.float32)
            raw_tensors[internal_name] = torch.tensor(data_np, device=ModelConfig.DEVICE)
        
        print(f"Loaded {len(raw_tensors)} factors: {list(raw_tensors.keys())}")
        
        # 3. Construct Masks
        # listed_mask: 标的在该日是否真实存在于原始数据中（禁止由填充值“创建”）
        close_raw_df = pivot_df['close'].copy().reindex(columns=assets)
        listed_mask = torch.tensor(
            close_raw_df.notna().values,
            device=ModelConfig.DEVICE,
            dtype=torch.bool,
        )

        # data_mask: 当前阶段以 CLOSE 原始可用性作为基础数据存在性口径
        data_mask = listed_mask & torch.isfinite(raw_tensors['CLOSE'])

        # tradable_mask: 业务可交易口径
        has_price = torch.isfinite(raw_tensors['CLOSE']) & (raw_tensors['CLOSE'] > 0)
        is_trading = torch.isfinite(raw_tensors['VOL']) & (raw_tensors['VOL'] > 0)
        not_expiring = torch.isfinite(raw_tensors['LEFT_YRS']) & (raw_tensors['LEFT_YRS'] > 0.5)
        tradable_mask = listed_mask & has_price & is_trading & not_expiring
        valid_mask = tradable_mask
        
        # 4. 计算预热偏移量（在 compute_features 前计算，传入 warmup_rows）
        effective_start_str = effective_start.strftime('%Y-%m-%d')
        warmup_offset = 0
        for i, d in enumerate(all_dates_list):
            if d >= effective_start_str:
                warmup_offset = i
                break

        # 5. Compute Features (Feature Engineer)
        # 注意: 必须在裁剪前计算，让 _robust_normalize 的滚动窗口利用预热段数据
        # warmup_offset 告知标准化器有多少行真实预热数据，以减少或跳过前 window 行的零化
        print(f"Computing features (warmup_offset={warmup_offset})...")
        feat_tensor, feature_valid_tensor = FeatureEngineer.compute_features(
            raw_tensors,
            warmup_rows=warmup_offset,
            cross_sectional_mask=tradable_mask,
            return_validity=True,
        )
        
        # 5. Compute Target Return (T+1)
        # 我们预测的是 T+1 的收益率 (close_T+1 / close_T - 1)
        close = raw_tensors['CLOSE']
        ret_1d = (torch.roll(close, -1, dims=0) / (close + 1e-9)) - 1.0
        # 对无效数据的收益率置 0
        ret_1d[~valid_mask] = 0.0
        ret_1d[~torch.isfinite(ret_1d)] = 0.0
        # 最后一行也是无效的
        ret_1d[-1] = 0.0

        # --- 裁剪预热段 (Warmup Trim) ---

        if warmup_offset > 0:
            print(
                f"Warmup trim: removing {warmup_offset} pre-training rows "
                f"(dates {all_dates_list[0]} ~ {all_dates_list[warmup_offset - 1]})"
            )
            # 裁剪所有张量和日期列表
            feat_tensor = feat_tensor[warmup_offset:]
            feature_valid_tensor = feature_valid_tensor[warmup_offset:]
            ret_1d = ret_1d[warmup_offset:]
            valid_mask = valid_mask[warmup_offset:]
            listed_mask = listed_mask[warmup_offset:]
            data_mask = data_mask[warmup_offset:]
            tradable_mask = tradable_mask[warmup_offset:]
            for k in raw_tensors:
                raw_tensors[k] = raw_tensors[k][warmup_offset:]
            all_dates_list = all_dates_list[warmup_offset:]

            # 验证: 特征首行应非全零（预热生效）
            first_row_nonzero = (feat_tensor[0].abs() > 1e-9).any().item()
            if first_row_nonzero:
                print("Warmup OK: first training day features are non-zero")
            else:
                print(
                    "Warning: first training day features are still zero "
                    "(warmup may be insufficient or data too sparse)"
                )
        else:
            if warmup_days > 0:
                print("Warning: warmup_offset=0, no pre-training data was trimmed")

        # --- 资产池清理 (Asset Cleanup) ---
        # 过滤在训练期间从未出现有效数据的"幽灵标的"（可能仅存在于预热段）
        ever_valid = valid_mask.any(dim=0)  # [Assets] bool
        if not ever_valid.all():
            drop_count = (~ever_valid).sum().item()
            keep_indices = ever_valid.nonzero(as_tuple=True)[0]
            feat_tensor = feat_tensor[:, keep_indices, :]
            feature_valid_tensor = feature_valid_tensor[:, keep_indices, :]
            ret_1d = ret_1d[:, keep_indices]
            valid_mask = valid_mask[:, keep_indices]
            listed_mask = listed_mask[:, keep_indices]
            data_mask = data_mask[:, keep_indices]
            tradable_mask = tradable_mask[:, keep_indices]
            for k in raw_tensors:
                raw_tensors[k] = raw_tensors[k][:, keep_indices]
            assets = [assets[i] for i in keep_indices.tolist()]
            print(f"Asset cleanup: dropped {drop_count} assets with no valid data in training period")

        # 赋值到实例属性
        self.assets_list = assets
        self.raw_data_cache = raw_tensors
        self.feat_tensor = feat_tensor
        self.feature_valid_tensor = feature_valid_tensor
        self.target_ret = ret_1d
        self.valid_mask = valid_mask
        self.listed_mask = listed_mask
        self.data_mask = data_mask
        self.tradable_mask = tradable_mask
        self.cs_mask = tradable_mask
        self.dates_list = all_dates_list
        print(f"Valid mask: {self.valid_mask.sum().item()} / {self.valid_mask.numel()} valid samples")
        
        # 6. 计算 Train/Val 分割索引（在裁剪后的日期列表上计算）
        split_date = RobustConfig.TRAIN_TEST_SPLIT_DATE
        split_idx = 0
        used_default_split = False
        for i, d in enumerate(self.dates_list):
            if d >= split_date:
                split_idx = i
                break
        # 如果没找到 (所有日期都在切分点之前)，则用 80% 处
        if split_idx == 0:
            split_idx = int(len(self.dates_list) * 0.8)
            used_default_split = True
        self.split_idx = split_idx
        
        print(f"Data Ready. Tensor Shape: {self.feat_tensor.shape}")
        if used_default_split:
            print(f"⚠️ Warning: Split date '{split_date}' not found, using default 80% split")
        print(f"Train/Val Split: idx={self.split_idx}, date={self.dates_list[self.split_idx] if self.split_idx < len(self.dates_list) else 'N/A'}")
