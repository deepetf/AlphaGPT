
import torch
import numpy as np
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.vm import StackVM
from model_core.engine import AlphaEngine

def verify_king_logic():
    print("🚀 Verifying King #9 Logic: Signal vs Noise...")
    
    # 1. 准备环境
    loader = CBDataLoader()
    loader.load_data()
    eng = AlphaEngine() # for decode/vocab
    vm = StackVM()
    
    # King #9 Formula (Best)
    # REMAIN_SIZE TS_DELTA LOG LOG CUT_HIGH TS_STD5 LOG SIGN LOG LOG CLOSE SUB
    # 手动重构 Token 序列或直接使用 Token ID
    # 由于 Token ID 随 config 变动，我们用 ops 名称动态查找 ID 比较麻烦
    # 这里我们直接复用 StackVM 里的逻辑，手动计算各个部分
    
    # 获取数据
    remain_size = loader.raw_data_cache['REMAIN_SIZE']
    close = loader.raw_data_cache['CLOSE']
    target_ret = loader.target_ret
    valid_mask = loader.valid_mask
    
    # 2. 复现 Signal 部分: TS_STD5( CUT_HIGH( LOG( LOG( TS_DELTA(REMAIN_SIZE) ) ) ) ) ...
    # 为了简化且避免 LOG NaN 问题，我们直接抓核心逻辑：异动
    # Core Signal = TS_DELTA(REMAIN_SIZE) != 0
    
    from model_core.ops_registry import OpsRegistry
    op_delta = OpsRegistry.get_op('TS_DELTA')['func']
    
    delta = op_delta(remain_size)
    
    # 定义“异动”: 规模发生实质性变化
    # 为了排除浮点误差，阈值设为 0.01 (通常单位是万或亿，0.01 变动很小了)
    is_signal_active = torch.abs(delta) > 1e-4
    
    print(f"\nSignal Active Elements: {is_signal_active.sum().item()} / {is_signal_active.numel()}")
    
    # 3. 计算对未来的预测能力 (IC 分析)
    # 我们看 T+1 的收益率
    
    # Group A: 发生异动的样本
    ret_signal = target_ret[is_signal_active & valid_mask]
    
    # Group B: 未发生异动的样本 (有效标的中)
    ret_noise = target_ret[(~is_signal_active) & valid_mask]
    
    print("\n--- Return Analysis (T+1) ---")
    print(f"Signal Group (Active) Mean Ret: {ret_signal.mean().item() * 100:.4f}% | Std: {ret_signal.std().item():.4f} | Count: {ret_signal.numel()}")
    print(f"Noise Group (Quiet)   Mean Ret: {ret_noise.mean().item() * 100:.4f}% | Std: {ret_noise.std().item():.4f} | Count: {ret_noise.numel()}")
    
    diff = ret_signal.mean().item() - ret_noise.mean().item()
    print(f"👉 Yield Gap (Signal - Quiet): {diff * 100:.4f}% per day")
    
    # 4. 验证“低价+异动”的威力
    # 我们看看在“低价债”里，发生异动是否会有超额收益
    # 定义低价: 价格 < 110
    is_low_price = (close < 110) & valid_mask
    
    ret_low_signal = target_ret[is_low_price & is_signal_active]
    ret_low_quiet = target_ret[is_low_price & (~is_signal_active)]
    
    print("\n--- Low Price (<110) Subset Analysis ---")
    print(f"Low + Active Mean Ret: {ret_low_signal.mean().item() * 100:.4f}% | Count: {ret_low_signal.numel()}")
    print(f"Low + Quiet  Mean Ret: {ret_low_quiet.mean().item() * 100:.4f}% | Count: {ret_low_quiet.numel()}")
    
    diff_low = ret_low_signal.mean().item() - ret_low_quiet.mean().item()
    print(f"👉 Alpha in Low Price: {diff_low * 100:.4f}% per day")
    
    if diff_low > 0:
        print("\n✅ HYPOTHESIS CONFIRMED: 异动确实带来了显著的超额收益！")
    else:
        print("\n❌ HYPOTHESIS FAILED: 异动并没有卵用。")

if __name__ == "__main__":
    verify_king_logic()
