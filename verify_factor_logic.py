
import torch
import numpy as np
from model_core.config import RobustConfig
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.ops_registry import OpsRegistry

def verify_remain_size_delta():
    print("🚀 Verifying REMAIN_SIZE + TS_DELTA validity...")
    
    # 1. 加载数据
    loader = CBDataLoader()
    loader.load_data()
    
    remain_size = loader.feat_tensor[:, :, 4] # index 4 is REMAIN_SIZE based on config
    # 或者为了稳妥，重新从 cache 获取
    remain_size = loader.raw_data_cache['REMAIN_SIZE']
    
    print(f"REMAIN_SIZE shape: {remain_size.shape}")
    
    # 2. 计算 TS_DELTA
    op_delta = OpsRegistry.get_op('TS_DELTA')['func']
    delta = op_delta(remain_size)
    
    # 3. 统计非零比例
    # 考虑到浮点误差，我们统计绝对值 > 1e-4 的比例
    non_zero_mask = torch.abs(delta) > 1e-4
    non_zero_count = non_zero_mask.sum().item()
    total_count = delta.numel()
    ratio = non_zero_count / total_count
    
    print(f"Total elements: {total_count}")
    print(f"Non-zero DELTA elements: {non_zero_count}")
    print(f"Non-zero Ratio: {ratio:.4%}")
    
    # 4. 观察非零值分布
    if non_zero_count > 0:
        valid_deltas = delta[non_zero_mask]
        print(f"Delta Stats -> Mean: {valid_deltas.mean():.4f}, Max: {valid_deltas.max():.4f}, Min: {valid_deltas.min():.4f}")
    
    # 5. 回测这个因子 (单纯的 Delta)
    print("\nRunning Backtest on pure TS_DELTA(REMAIN_SIZE)...")
    bt = CBBacktest(top_k=RobustConfig.TOP_K)
    
    # 因子1: Delta本身 (看是否规模增加/减少有信号)
    # 对 NaN 填充 0
    factor_raw = torch.nan_to_num(delta, 0.0)
    score, cum_ret, sharpe = bt.evaluate(factor_raw, loader.target_ret, loader.valid_mask)
    print(f"Pure Delta Factor -> Score: {score:.2f} | Sharpe: {sharpe:.2f} | Ret: {cum_ret:.2%}")
    
    # 因子2: Delta 取绝对值 (看是否"有动静"就是好事)
    factor_abs = torch.abs(factor_raw)
    score_abs, cum_ret_abs, sharpe_abs = bt.evaluate(factor_abs, loader.target_ret, loader.valid_mask)
    print(f"Abs Delta Factor  -> Score: {score_abs:.2f} | Sharpe: {sharpe_abs:.2f} | Ret: {cum_ret_abs:.2%}")

    # 因子3: 对比基准 CLOSE NEG
    print("\nRunning Backtest on CLOSE NEG (Benchmark)...")
    close = loader.raw_data_cache['CLOSE']
    factor_close = -close
    score_c, cum_ret_c, sharpe_c = bt.evaluate(factor_close, loader.target_ret, loader.valid_mask)
    print(f"CLOSE NEG Factor  -> Score: {score_c:.2f} | Sharpe: {sharpe_c:.2f} | Ret: {cum_ret_c:.2%}")

if __name__ == "__main__":
    verify_remain_size_delta()
