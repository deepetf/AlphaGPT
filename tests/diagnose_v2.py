"""
Precise diagnosis of simulation vs backtest differences
"""
import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import json
import numpy as np
import pandas as pd
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.vm import StackVM
from model_core.config import RobustConfig

print("="*80)
print("PRECISE DIAGNOSIS: Simulation vs Backtest Differences")
print("="*80)

# Load data
print("\n[1] Loading data...")
loader = CBDataLoader()
loader.load_data()

with open(r"c:\Trading\Projects\AlphaGPT\model_core\best_cb_formula.json", 'r') as f:
    formula = json.load(f)['best']['formula']

vm = StackVM()
factors = vm.execute(formula, loader.feat_tensor.to('cpu'))

backtest = CBBacktest(top_k=RobustConfig.TOP_K, fee_rate=RobustConfig.FEE_RATE)
result = backtest.evaluate_with_details(
    factors=factors,
    target_ret=loader.target_ret,
    valid_mask=loader.valid_mask
)

sim_df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")
with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_holdings.json", 'r', encoding='utf-8') as f:
    sim_holdings = json.load(f)

print("\n" + "="*80)
print("DIAGNOSIS 4: Daily Return Decomposition (CRITICAL)")
print("="*80)

# For each day, calculate expected return from holdings
discrepancies = []

for i in range(min(10, len(sim_holdings) - 1)):
    date = sim_holdings[i]['date']
    next_date = sim_holdings[i+1]['date']
    date_idx = loader.dates_list.index(date)
    next_date_idx = loader.dates_list.index(next_date)
    
    holdings = sim_holdings[i]['holdings']
    total_value_t = 0
    total_value_t1 = 0
    
    for code, shares in holdings.items():
        asset_idx = loader.assets_list.index(code)
        price_t = loader.raw_data_cache['CLOSE'][date_idx, asset_idx].item()
        price_t1 = loader.raw_data_cache['CLOSE'][next_date_idx, asset_idx].item()
        
        total_value_t += shares * price_t
        total_value_t1 += shares * price_t1
    
    # Expected return from holdings (T holdings -> T+1 value change)
    expected_ret = (total_value_t1 / total_value_t) - 1 if total_value_t > 0 else 0
    
    # Simulation return (for next_date in CSV)
    sim_row = sim_df[sim_df['Date'] == next_date]
    sim_ret = sim_row['Sim_Return'].values[0] if len(sim_row) > 0 else 0
    
    # Backtest return (for date)
    bt_ret = result['daily_returns'][date_idx]
    
    discrepancies.append({
        'date': date,
        'next_date': next_date,
        'expected': expected_ret,
        'sim': sim_ret,
        'backtest': bt_ret,
        'sim_diff': sim_ret - expected_ret,
        'bt_diff': bt_ret - expected_ret
    })
    
    print(f"\n{date} -> {next_date}:")
    print(f"  Expected (from holdings): {expected_ret:+.4%}")
    print(f"  Simulation ({next_date}):  {sim_ret:+.4%}  Diff: {sim_ret - expected_ret:+.4%}")
    print(f"  Backtest ({date}):         {bt_ret:+.4%}  Diff: {bt_ret - expected_ret:+.4%}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

avg_sim_diff = np.mean([abs(d['sim_diff']) for d in discrepancies])
avg_bt_diff = np.mean([abs(d['bt_diff']) for d in discrepancies])

print(f"\nAverage |Sim - Expected|: {avg_sim_diff:.4%}")
print(f"Average |BT - Expected|:  {avg_bt_diff:.4%}")

if avg_sim_diff > avg_bt_diff:
    print("\n=> Simulation has LARGER deviation from expected returns")
    print("   Possible cause: timing mismatch or cost model")
else:
    print("\n=> Backtest has LARGER deviation from expected returns")
    print("   Possible cause: backtest uses weights, not shares")

# Check correlation between expected and actual
sim_rets = [d['sim'] for d in discrepancies]
bt_rets = [d['backtest'] for d in discrepancies]
expected_rets = [d['expected'] for d in discrepancies]

corr_sim_exp = np.corrcoef(sim_rets, expected_rets)[0, 1]
corr_bt_exp = np.corrcoef(bt_rets, expected_rets)[0, 1]
corr_sim_bt = np.corrcoef(sim_rets, bt_rets)[0, 1]

print(f"\nCorrelations:")
print(f"  Simulation vs Expected: {corr_sim_exp:.4f}")
print(f"  Backtest vs Expected:   {corr_bt_exp:.4f}")
print(f"  Simulation vs Backtest: {corr_sim_bt:.4f}")

print("\n" + "="*80)
