"""
诊断脚本：对比模拟与回测的选股结果
"""
import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import json
import torch
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

# Check a specific date
test_date = "2024-01-03"
date_idx = loader.dates_list.index(test_date)

print(f"Date: {test_date} (idx={date_idx})")
print(f"Backtest return at idx={date_idx}: {result['daily_returns'][date_idx]:.6f}")
print(f"Backtest holdings at idx={date_idx}: {result['daily_holdings'][date_idx][:5]}...")  # First 5

# Check target_ret
print(f"\nTarget returns (first 5 assets at idx={date_idx}):")
for i in range(5):
    print(f"  Asset {i}: {loader.target_ret[date_idx, i].item():.6f}")

# Check valid mask
valid_count = loader.valid_mask[date_idx, :].sum().item()
print(f"\nValid assets at idx={date_idx}: {valid_count}")
