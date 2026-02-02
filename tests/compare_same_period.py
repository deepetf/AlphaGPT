"""
对比相同时间段的回测 vs 模拟收益
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import json
import torch
import numpy as np
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.vm import StackVM
from model_core.config import RobustConfig

# Load data
loader = CBDataLoader()
loader.load_data()

# Load formula
with open(r"c:\Trading\Projects\AlphaGPT\model_core\best_cb_formula.json", 'r') as f:
    formula = json.load(f)['best']['formula']

# Execute formula
vm = StackVM()
factors = vm.execute(formula, loader.feat_tensor.to('cpu'))

# Run backtest
backtest = CBBacktest(top_k=RobustConfig.TOP_K, fee_rate=RobustConfig.FEE_RATE)
result = backtest.evaluate_with_details(
    factors=factors,
    target_ret=loader.target_ret,
    valid_mask=loader.valid_mask
)

# Filter to 2024 only
start_date = "2024-01-02"
end_date = "2024-12-31"

start_idx = loader.dates_list.index(start_date)
end_idx = loader.dates_list.index(end_date) + 1

# Calculate 2024 backtest performance
backtest_returns_2024 = result['daily_returns'][start_idx:end_idx]
backtest_returns_arr = np.array(backtest_returns_2024)

# Remove zeros (non-trading days)
backtest_returns_arr = backtest_returns_arr[backtest_returns_arr != 0]

cum_ret_backtest = (1 + backtest_returns_arr).prod() - 1
sharpe_backtest = (backtest_returns_arr.mean() / (backtest_returns_arr.std() + 1e-9)) * np.sqrt(252)
ann_ret_backtest = (1 + cum_ret_backtest) ** (252 / len(backtest_returns_arr)) - 1

# Load simulation results
import pandas as pd
sim_df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")
sim_returns_arr = sim_df['Sim_Return'].values[1:]  # Skip first day

cum_ret_sim = (1 + sim_returns_arr).prod() - 1
sharpe_sim = (sim_returns_arr.mean() / (sim_returns_arr.std() + 1e-9)) * np.sqrt(252)
ann_ret_sim = (1 + cum_ret_sim) ** (252 / len(sim_returns_arr)) - 1

print("="*80)
print("2024年度回测 vs 模拟对比（相同时间段）")
print("="*80)
print(f"\n时间段: {start_date} 至 {end_date}")
print(f"回测交易天数: {len(backtest_returns_arr)}")
print(f"模拟交易天数: {len(sim_returns_arr)}")

print(f"\n{'指标':<20} {'回测':<15} {'模拟':<15} {'差异':<15}")
print("-"*70)
print(f"{'累计收益率':<20} {cum_ret_backtest:>13.2%} {cum_ret_sim:>13.2%} {abs(cum_ret_backtest - cum_ret_sim):>13.2%}")
print(f"{'年化收益率':<20} {ann_ret_backtest:>13.2%} {ann_ret_sim:>13.2%} {abs(ann_ret_backtest - ann_ret_sim):>13.2%}")
print(f"{'夏普比率':<20} {sharpe_backtest:>13.4f} {sharpe_sim:>13.4f} {abs(sharpe_backtest - sharpe_sim):>13.4f}")

print(f"\n{'='*80}")
print("结论:")
if abs(cum_ret_backtest - cum_ret_sim) < 0.03:  # 3% tolerance
    print("✅ 回测与模拟在相同时间段内表现一致（差异 < 3%）")
    print("   → 差异主要来自整手交易约束，符合预期")
else:
    print("❌ 回测与模拟存在显著差异")
    print(f"   → 累计收益差异: {abs(cum_ret_backtest - cum_ret_sim):.2%}")
    print("   → 需要进一步排查原因")
print("="*80)
