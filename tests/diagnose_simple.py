"""
深度诊断：逐日对比回测与模拟的持仓权重
"""
import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import json
import torch
import numpy as np
import pandas as pd
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.vm import StackVM
from model_core.config import RobustConfig

print("Loading data...")
loader = CBDataLoader()
loader.load_data()

with open(r"c:\Trading\Projects\AlphaGPT\model_core\best_cb_formula.json", 'r') as f:
    formula = json.load(f)['best']['formula']

print("Executing formula...")
vm = StackVM()
factors = vm.execute(formula, loader.feat_tensor.to('cpu'))

print("Running backtest...")
backtest = CBBacktest(top_k=RobustConfig.TOP_K, fee_rate=RobustConfig.FEE_RATE)
result = backtest.evaluate_with_details(
    factors=factors,
    target_ret=loader.target_ret,
    valid_mask=loader.valid_mask
)

# Load simulation holdings
with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_holdings.json", 'r', encoding='utf-8') as f:
    sim_holdings = json.load(f)

# Load simulation returns
sim_df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")

print("\n" + "="*100)
print("DETAILED DAILY COMPARISON")
print("="*100)

weight_diffs_all = []
return_diffs_all = []

for i in range(min(20, len(sim_holdings))):
    date = sim_holdings[i]['date']
    date_idx = loader.dates_list.index(date)
    
    print(f"\nDate: {date} (idx={date_idx})")
    
    # Backtest holdings
    backtest_holdings_idx = result['daily_holdings'][date_idx]
    backtest_codes = [loader.assets_list[idx] for idx in backtest_holdings_idx]
    
    # Simulation holdings
    sim_codes = list(sim_holdings[i]['holdings'].keys())
    sim_shares = sim_holdings[i]['holdings']
    
    print(f"  Backtest: {len(backtest_codes)} assets, Simulation: {len(sim_codes)} assets")
    
    # Check if codes match
    backtest_set = set(backtest_codes)
    sim_set = set(sim_codes)
    
    if backtest_set == sim_set:
        print("  Codes: MATCH")
    else:
        print(f"  Codes: DIFFER - Only BT: {backtest_set - sim_set}, Only Sim: {sim_set - backtest_set}")
        continue
    
    # Calculate weights
    bt_weight = 1.0 / len(backtest_codes) if len(backtest_codes) > 0 else 0
    total_value = sim_holdings[i]['equity'] - sim_holdings[i]['cash']
    
    weight_diffs = []
    for code in sim_codes:
        shares = sim_shares[code]
        price = loader.raw_data_cache['CLOSE'][date_idx, loader.assets_list.index(code)].item()
        amount = shares * price
        sim_w = amount / total_value if total_value > 0 else 0
        weight_diffs.append(abs(bt_weight - sim_w))
    
    avg_weight_diff = np.mean(weight_diffs)
    max_weight_diff = np.max(weight_diffs)
    weight_diffs_all.append(avg_weight_diff)
    
    print(f"  Weight Diff: Avg={avg_weight_diff:.4f}, Max={max_weight_diff:.4f}")
    
    # Returns
    backtest_ret = result['daily_returns'][date_idx]
    sim_ret_row = sim_df[sim_df['Date'] == date]
    if len(sim_ret_row) > 0:
        sim_ret = sim_ret_row['Sim_Return'].values[0]
    else:
        sim_ret = 0
    
    return_diff = abs(backtest_ret - sim_ret)
    return_diffs_all.append(return_diff)
    
    print(f"  Return: BT={backtest_ret:+.4%}, Sim={sim_ret:+.4%}, Diff={return_diff:.4%}")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"Average Weight Difference: {np.mean(weight_diffs_all):.4f}")
print(f"Max Weight Difference: {np.max(weight_diffs_all):.4f}")
print(f"Average Return Difference: {np.mean(return_diffs_all):.4%}")
print(f"Max Return Difference: {np.max(return_diffs_all):.4%}")
print("="*100)
