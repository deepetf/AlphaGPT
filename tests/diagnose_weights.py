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

# Analyze first 10 days
print("\n" + "="*100)
print("DETAILED DAILY COMPARISON (First 10 days)")
print("="*100)

for i in range(min(10, len(sim_holdings))):
    date = sim_holdings[i]['date']
    date_idx = loader.dates_list.index(date)
    
    print(f"\n{'='*100}")
    print(f"Date: {date} (idx={date_idx})")
    print(f"{'='*100}")
    
    # Backtest holdings
    backtest_holdings_idx = result['daily_holdings'][date_idx]
    backtest_codes = [loader.assets_list[idx] for idx in backtest_holdings_idx]
    
    # Simulation holdings
    sim_codes = list(sim_holdings[i]['holdings'].keys())
    sim_shares = sim_holdings[i]['holdings']
    
    print(f"\nBacktest Holdings: {len(backtest_codes)} assets")
    print(f"Simulation Holdings: {len(sim_codes)} assets")
    
    # Check if codes match
    backtest_set = set(backtest_codes)
    sim_set = set(sim_codes)
    
    if backtest_set == sim_set:
        print("✅ Holdings codes MATCH")
    else:
        print("❌ Holdings codes DIFFER")
        print(f"   Only in Backtest: {backtest_set - sim_set}")
        print(f"   Only in Simulation: {sim_set - backtest_set}")
    
    # Calculate weights
    print(f"\n{'Code':<15} {'BT Weight':<12} {'Sim Shares':<12} {'Sim Amount':<15} {'Sim Weight':<12}")
    print("-"*80)
    
    # Backtest: equal weight
    bt_weight = 1.0 / len(backtest_codes) if len(backtest_codes) > 0 else 0
    
    # Simulation: calculate actual weights
    total_value = sim_holdings[i]['equity'] - sim_holdings[i]['cash']
    
    for code in sorted(backtest_set | sim_set):
        bt_w = bt_weight if code in backtest_set else 0
        
        if code in sim_set:
            shares = sim_shares[code]
            price = loader.raw_data_cache['CLOSE'][date_idx, loader.assets_list.index(code)].item()
            amount = shares * price
            sim_w = amount / total_value if total_value > 0 else 0
            print(f"{code:<15} {bt_w:>10.4f} {shares:>11} {amount:>14,.2f} {sim_w:>11.4f}")
        else:
            print(f"{code:<15} {bt_w:>10.4f} {'N/A':>11} {'N/A':>14} {'N/A':>11}")
    
    # Calculate weight difference
    if backtest_set == sim_set and len(sim_codes) > 0:
        weight_diffs = []
        for code in sim_codes:
            shares = sim_shares[code]
            price = loader.raw_data_cache['CLOSE'][date_idx, loader.assets_list.index(code)].item()
            amount = shares * price
            sim_w = amount / total_value if total_value > 0 else 0
            weight_diffs.append(abs(bt_weight - sim_w))
        
        avg_weight_diff = np.mean(weight_diffs)
        max_weight_diff = np.max(weight_diffs)
        
        print(f"\nWeight Difference: Avg={avg_weight_diff:.4f}, Max={max_weight_diff:.4f}")
    
    # Calculate expected return
    backtest_ret = result['daily_returns'][date_idx]
    
    # Simulation return (from next day's equity change)
    if i < len(sim_holdings) - 1:
        sim_ret = (sim_holdings[i+1]['equity'] - sim_holdings[i]['equity']) / sim_holdings[i]['equity']
    else:
        sim_ret = 0
    
    print(f"\nBacktest Return: {backtest_ret:+.4%}")
    print(f"Simulation Return: {sim_ret:+.4%}")
    print(f"Difference: {abs(backtest_ret - sim_ret):.4%}")

print("\n" + "="*100)
print("DIAGNOSIS COMPLETE")
print("="*100)
