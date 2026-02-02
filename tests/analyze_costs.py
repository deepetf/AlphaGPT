"""
验证交易成本模型差异
"""
import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import json
import numpy as np

# Load simulation trades
with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_trades.json", 'r', encoding='utf-8') as f:
    trades = json.load(f)

# Load holdings
with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_holdings.json", 'r', encoding='utf-8') as f:
    holdings = json.load(f)

print("Analyzing transaction costs...")
print("="*80)

fee_rate = 0.001
total_sim_cost = 0
total_turnover_cost = 0

for i in range(min(10, len(trades))):
    date = trades[i]['date']
    buys = trades[i]['buys']
    sells = trades[i]['sells']
    
    # Simulation cost (actual implementation)
    buy_cost = sum(b['amount'] * fee_rate for b in buys)
    sell_cost = sum(s['amount'] * fee_rate for s in sells)
    sim_cost = buy_cost + sell_cost
    
    # Backtest-style cost (turnover-based)
    # Turnover = sum of absolute weight changes
    # For equal-weight: turnover ≈ (# of trades) / (# of holdings) * 2
    equity = holdings[i]['equity']
    num_holdings = len(holdings[i]['holdings'])
    
    if num_holdings > 0:
        target_weight = 1.0 / num_holdings
        # Approximate turnover: each buy/sell represents a weight change
        turnover = (len(buys) + len(sells)) * target_weight
        turnover_cost = equity * turnover * fee_rate
    else:
        turnover_cost = 0
    
    total_sim_cost += sim_cost
    total_turnover_cost += turnover_cost
    
    print(f"{date}: Sim={sim_cost:.2f}, Turnover={turnover_cost:.2f}, Diff={abs(sim_cost - turnover_cost):.2f}")

print("="*80)
print(f"Total Simulation Cost: {total_sim_cost:.2f}")
print(f"Total Turnover Cost: {total_turnover_cost:.2f}")
print(f"Difference: {abs(total_sim_cost - total_turnover_cost):.2f}")
print(f"Relative Difference: {abs(total_sim_cost - total_turnover_cost) / total_sim_cost * 100:.2f}%")
