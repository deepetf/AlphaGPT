"""
详细对比单日持仓和收益计算
"""
import sys
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

# Analyze 2024-01-03
test_date = "2024-01-03"
date_idx = loader.dates_list.index(test_date)

print(f"="*60)
print(f"Detailed Analysis for {test_date} (idx={date_idx})")
print(f"="*60)

# Get holdings
holdings_indices = result['daily_holdings'][date_idx]
print(f"\nTop-{RobustConfig.TOP_K} Holdings (indices): {holdings_indices}")

# Get codes and returns
print(f"\n{'Code':<12} {'Target_Ret':<12} {'Weight':<10} {'Contribution':<12}")
print("-"*60)

total_gross_ret = 0.0
for idx in holdings_indices:
    code = loader.assets_list[idx]
    target_ret = loader.target_ret[date_idx, idx].item()
    weight = 1.0 / len(holdings_indices)
    contribution = weight * target_ret
    total_gross_ret += contribution
    print(f"{code:<12} {target_ret:>11.6f} {weight:>9.4f} {contribution:>11.6f}")

print("-"*60)
print(f"{'Total Gross Return':<34} {total_gross_ret:>11.6f}")

# Calculate turnover and cost
if date_idx > 0:
    prev_holdings = set(result['daily_holdings'][date_idx-1])
    curr_holdings = set(holdings_indices)
    
    # Turnover = sum of absolute weight changes
    # Assuming equal weight, turnover = (# of changes) / TOP_K
    sells = len(prev_holdings - curr_holdings)
    buys = len(curr_holdings - prev_holdings)
    turnover = (sells + buys) / RobustConfig.TOP_K
    
    print(f"Sells: {sells}, Buys: {buys}")
    print(f"Turnover: {turnover:.4f}")
    
    tx_cost = turnover * RobustConfig.FEE_RATE * 2
    print(f"Transaction Cost: {tx_cost:.6f}")
    
    net_ret = total_gross_ret - tx_cost
    print(f"Net Return: {net_ret:.6f}")
else:
    print("(First day, no previous holdings)")

print(f"\nBacktest reported return: {result['daily_returns'][date_idx]:.6f}")
