"""
精确诊断差异来源
按优先级排查：
1. 收益时间对齐
2. 成本模型一致性
3. 持仓系统同步性
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

print("="*80)
print("精确诊断：模拟vs回测差异根本原因")
print("="*80)

# Load data
print("\n[1] 加载数据...")
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

# Load simulation data
sim_df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")
with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_holdings.json", 'r', encoding='utf-8') as f:
    sim_holdings = json.load(f)

print("\n" + "="*80)
print("诊断1: 收益时间对齐检查")
print("="*80)

# Check if simulation returns align with backtest T+1 returns
print("\n回测逻辑: target_ret[t] = (close[t+1] / close[t]) - 1")
print("含义: T日选股 → T日收盘价买入 → T+1日收盘价卖出的收益")

# Get first few days for detailed comparison
test_dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]

print(f"\n{'日期':<12} {'回测Return':<14} {'模拟Return':<14} {'差异':<12} {'说明'}")
print("-"*70)

for date in test_dates:
    date_idx = loader.dates_list.index(date)
    bt_ret = result['daily_returns'][date_idx]
    
    # Find simulation return for this date
    sim_row = sim_df[sim_df['Date'] == date]
    if len(sim_row) > 0:
        sim_ret = sim_row['Sim_Return'].values[0]
        diff = sim_ret - bt_ret
        
        # Get next day for verification
        if date_idx + 1 < len(loader.dates_list):
            next_date = loader.dates_list[date_idx + 1]
        else:
            next_date = "N/A"
        
        print(f"{date:<12} {bt_ret:>12.4%} {sim_ret:>12.4%} {diff:>10.4%}   BT: {date}→{next_date}")
    else:
        print(f"{date:<12} {bt_ret:>12.4%} {'N/A':>12} {'N/A':>10}")

print("\n" + "="*80)
print("诊断2: 成本模型一致性检查")
print("="*80)

# Compare cost models for first 10 days
print("\n回测成本模型: turnover * fee_rate * 2 (权重换手)")
print("模拟成本模型: amount * fee_rate (成交金额)")

with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_trades.json", 'r', encoding='utf-8') as f:
    sim_trades = json.load(f)

total_sim_cost = 0
total_bt_cost = 0

print(f"\n{'日期':<12} {'模拟成本':<12} {'回测成本(估)':<14} {'差异':<12} {'换手率':<10}")
print("-"*70)

for i in range(min(20, len(sim_trades))):
    date = sim_trades[i]['date']
    date_idx = loader.dates_list.index(date)
    
    # Simulation cost (actual)
    buys = sim_trades[i]['buys']
    sells = sim_trades[i]['sells']
    sim_cost = sum(b['amount'] * RobustConfig.FEE_RATE for b in buys)
    sim_cost += sum(s['amount'] * RobustConfig.FEE_RATE for s in sells)
    
    # Backtest cost (estimated from turnover)
    # turnover = sum of absolute weight changes
    equity = sim_holdings[i]['equity']
    num_trades = len(buys) + len(sells)
    # 近似换手率 = (买+卖的只数) / 持仓数 * 平均权重
    num_holdings = len(sim_holdings[i]['holdings'])
    if num_holdings > 0:
        avg_weight = 1.0 / num_holdings
        turnover = num_trades * avg_weight
        bt_cost_est = equity * turnover * RobustConfig.FEE_RATE
    else:
        turnover = 0
        bt_cost_est = 0
    
    diff = sim_cost - bt_cost_est
    total_sim_cost += sim_cost
    total_bt_cost += bt_cost_est
    
    print(f"{date:<12} {sim_cost:>10.2f} {bt_cost_est:>12.2f} {diff:>10.2f} {turnover:>8.2%}")

print(f"\n累计成本 - 模拟: {total_sim_cost:.2f}, 回测(估): {total_bt_cost:.2f}, 差异: {total_sim_cost - total_bt_cost:.2f}")

print("\n" + "="*80)
print("诊断3: 持仓系统同步性检查")
print("="*80)

# Check if simulation holdings match backtest top-k selections
mismatch_count = 0
total_checks = 0

print(f"\n{'日期':<12} {'回测TopK':<12} {'模拟持仓':<12} {'Jaccard':<10} {'状态'}")
print("-"*60)

for i in range(min(20, len(sim_holdings))):
    date = sim_holdings[i]['date']
    date_idx = loader.dates_list.index(date)
    
    # Backtest holdings
    bt_holdings_idx = result['daily_holdings'][date_idx]
    bt_holdings = set([loader.assets_list[idx] for idx in bt_holdings_idx])
    
    # Simulation holdings
    sim_holdings_set = set(sim_holdings[i]['holdings'].keys())
    
    # Jaccard
    intersection = len(bt_holdings & sim_holdings_set)
    union = len(bt_holdings | sim_holdings_set)
    jaccard = intersection / union if union > 0 else 0
    
    status = "✓" if jaccard == 1.0 else "✗"
    if jaccard < 1.0:
        mismatch_count += 1
    total_checks += 1
    
    print(f"{date:<12} {len(bt_holdings):<12} {len(sim_holdings_set):<12} {jaccard:<10.2%} {status}")

print(f"\n持仓一致性: {total_checks - mismatch_count}/{total_checks} ({(total_checks - mismatch_count)/total_checks:.1%})")

print("\n" + "="*80)
print("诊断4: 逐日收益分解")
print("="*80)

# For each day, calculate expected return based on holdings and actual prices
print("\n验证：用持仓和价格直接计算预期收益，对比模拟和回测")

for i in range(min(5, len(sim_holdings) - 1)):
    date = sim_holdings[i]['date']
    next_date = sim_holdings[i+1]['date']
    date_idx = loader.dates_list.index(date)
    next_date_idx = loader.dates_list.index(next_date)
    
    print(f"\n--- {date} → {next_date} ---")
    
    holdings = sim_holdings[i]['holdings']
    total_value_t = 0
    total_value_t1 = 0
    
    print(f"{'Code':<12} {'Shares':<8} {'Price_T':<12} {'Price_T+1':<12} {'Ret':<10}")
    
    for code, shares in holdings.items():
        asset_idx = loader.assets_list.index(code)
        price_t = loader.raw_data_cache['CLOSE'][date_idx, asset_idx].item()
        price_t1 = loader.raw_data_cache['CLOSE'][next_date_idx, asset_idx].item()
        
        value_t = shares * price_t
        value_t1 = shares * price_t1
        ret = (price_t1 / price_t - 1)
        
        total_value_t += value_t
        total_value_t1 += value_t1
        
        print(f"{code:<12} {shares:<8} {price_t:>10.2f} {price_t1:>10.2f} {ret:>8.2%}")
    
    # Expected return from holdings
    expected_ret = (total_value_t1 / total_value_t) - 1 if total_value_t > 0 else 0
    
    # Simulation return (from CSV, for next day)
    sim_row = sim_df[sim_df['Date'] == next_date]
    sim_ret = sim_row['Sim_Return'].values[0] if len(sim_row) > 0 else 0
    
    # Backtest return
    bt_ret = result['daily_returns'][date_idx]
    
    print(f"\n从持仓计算的预期收益: {expected_ret:.4%}")
    print(f"模拟记录的收益 ({next_date}): {sim_ret:.4%}")
    print(f"回测记录的收益 ({date}): {bt_ret:.4%}")
    print(f"差异 (模拟-预期): {sim_ret - expected_ret:.4%}")
    print(f"差异 (回测-预期): {bt_ret - expected_ret:.4%}")

print("\n" + "="*80)
print("诊断完成")
print("="*80)
