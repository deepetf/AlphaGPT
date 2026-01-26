
import torch
import pandas as pd
from model_core.data_loader import CBDataLoader
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.ops_registry import OpsRegistry

# King #8 公式
FORMULA = [
    "VOLATILITY_STK", "ABS", "PCT_CHG_5_STK", "TS_MEAN5", 
    "CS_RANK", "CS_RANK", "TS_MEAN5", "SIGN", 
    "CS_RANK", "PREM", "MAX", "IF_POS"
]

def realtime_data_provider(full_loader, current_date_idx, window_size=70):
    """
    模拟实盘数据提供器：只提供 [t-window+1, t] 的数据片段
    """
    start_idx = max(0, current_date_idx - window_size + 1)
    end_idx = current_date_idx + 1 # Slice is exclusive
    
    # 构造一个模拟的 raw_data 字典
    mock_raw = {}
    for key, tensor in full_loader.raw_data_cache.items():
        # Tensor shape: [Time, Assets]
        mock_raw[key] = tensor[start_idx:end_idx, :].clone()
        
    return mock_raw

def run_realtime_simulation():
    print("🚀 Verifying King #8 Real-time Logic...")
    
    # 1. 加载全量数据作为"真值"源
    loader = CBDataLoader()
    loader.load_data()
    
    # 2. 全量计算因子 (作为 Ground Truth)
    print("\nComputing Ground Truth (Full History)...")
    vm = StackVM()
    full_factors = vm.execute(FORMULA, loader.feat_tensor)
    
    # 3. 模拟实盘：取最后一天 (T)
    last_idx = len(loader.dates_list) - 1
    target_date = loader.dates_list[last_idx]
    
    print(f"\nrunning Real-time Simulation for {target_date}...")
    
    # [关键步骤] 构造只包含过去 70 天数据的"实盘环境"
    # 这模拟了实盘时我们只能获取到历史数据的情况
    raw_data_slice = realtime_data_provider(loader, last_idx, window_size=70)
    
    # [关键步骤] 重新做特征工程
    # 这里会调用 FeatureEngineer._robust_normalize
    # 由于它是 Rolling(60) 的，即使只有 70 天数据，
    # 最后一天 (第70天) 的特征值应该与全量计算时的最后一天完全一致！
    print("  > Feature Engineering (on 70-day slice)...")
    feat_slice = FeatureEngineer.compute_features(raw_data_slice)
    
    # [关键步骤] 执行 VM
    print("  > Executing Strategy...")
    # 注意：VM 输入 [70, Assets, Feats]
    factor_slice = vm.execute(FORMULA, feat_slice)
    
    # 取最后一天的结果
    realtime_factor = factor_slice[-1] # [Assets]
    
    # 4. 对比
    ground_truth = full_factors[last_idx] # [Assets]
    
    # 处理 NaN (某些停牌标的可能是 NaN)
    mask = ~torch.isnan(ground_truth) & ~torch.isnan(realtime_factor)
    
    diff = (realtime_factor[mask] - ground_truth[mask]).abs().max().item()
    
    print(f"\n📊 Verification Result for {target_date}:")
    print(f"   Max Difference: {diff:.8f}")
    
    if diff < 1e-5:
        print("✅ SUCCESS: Real-time calculation matches backtest exactly!")
    else:
        print("❌ FAILURE: Significant mismatch found.")
        
    # 5. 输出当天的 Top 10 选股
    print(f"\n🏆 Top 10 Picks for {target_date}:")
    _, top_indices = torch.topk(realtime_factor, k=10)
    
    print(f"{'Code':<10} {'Name':<10} {'Factor':<10}")
    print("-" * 40)
    for idx in top_indices:
        code = loader.assets_list[idx]
        name = loader.names_dict.get(code, code)
        val = realtime_factor[idx].item()
        print(f"{code:<10} {name:<10} {val:.4f}")

if __name__ == "__main__":
    run_realtime_simulation()
