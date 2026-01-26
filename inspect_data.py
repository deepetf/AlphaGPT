
import torch
import numpy as np
from model_core.data_loader import CBDataLoader
from model_core.ops_registry import OpsRegistry

def inspect_pure_value():
    print("🚀 Inspecting PURE_VALUE Data...")
    
    loader = CBDataLoader()
    loader.load_data()
    
    # 获取数据
    if 'PURE_VALUE' not in loader.raw_data_cache:
        print("❌ PURE_VALUE not found in loaded data!")
        return
        
    pv = loader.raw_data_cache['PURE_VALUE']
    
    # 1. 基础统计
    print(f"\n--- Basic Stats ---")
    print(f"Shape: {pv.shape}")
    print(f"Mean: {pv.mean():.4f}")
    print(f"Std (Global): {pv.std():.4f}")
    print(f"Max: {pv.max():.4f}, Min: {pv.min():.4f}")
    
    # 2. 检查 TS_STD5
    op_std = OpsRegistry.get_op('TS_STD5')['func']
    pv_std = op_std(pv)
    
    print(f"\n--- TS_STD5(PURE_VALUE) Stats ---")
    print(f"Mean: {pv_std.mean():.6f}")
    print(f"Max:  {pv_std.max():.6f}")
    print(f"Zeros: {(pv_std == 0).sum().item()} / {pv_std.numel()}")
    
    # 3. 抽样打印
    # 找几个非零的样本看看
    mask = pv_std > 0.1
    if mask.sum() > 0:
        print("\n--- Sample High Volatility Instances ---")
        indices = torch.nonzero(mask)
        # 打印前 5 个
        for i in range(min(5, len(indices))):
            t, idx = indices[i]
            code = loader.assets_list[idx]
            date = loader.dates_list[t]
            val = pv[t, idx].item()
            std_val = pv_std[t, idx].item()
            print(f"Date: {date}, Code: {code}, PV: {val:.2f}, PV_STD5: {std_val:.4f}")
    else:
        print("\n❌ Amazing! PURE_VALUE is extremely stable (Std < 0.1 everywhere).")

if __name__ == "__main__":
    inspect_pure_value()
