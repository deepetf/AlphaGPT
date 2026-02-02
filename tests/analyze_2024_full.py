"""
2024全年验证结果深度分析
"""
import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import pandas as pd
import numpy as np
import json

# Load data
df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
df['Quarter'] = df['Date'].dt.to_period('Q')

print("="*80)
print("2024 FULL YEAR VERIFICATION ANALYSIS")
print("="*80)

# Overall metrics
overall_corr = np.corrcoef(df['Sim_Return'], df['Backtest_Return'])[0, 1]
overall_mae = np.abs(df['Sim_Return'] - df['Backtest_Return']).mean()

print(f"\nOverall Metrics:")
print(f"  Correlation: {overall_corr:.4f}")
print(f"  MAE: {overall_mae:.6f}")
print(f"  Total Days: {len(df)}")

# Quarterly breakdown
print(f"\n{'Quarter':<12} {'Days':<8} {'Correlation':<15} {'MAE':<12}")
print("-"*50)
for quarter, group in df.groupby('Quarter'):
    if len(group) > 1:
        corr = np.corrcoef(group['Sim_Return'], group['Backtest_Return'])[0, 1]
        mae = np.abs(group['Sim_Return'] - group['Backtest_Return']).mean()
        print(f"{str(quarter):<12} {len(group):<8} {corr:>13.4f} {mae:>11.6f}")

# Monthly breakdown
print(f"\n{'Month':<12} {'Days':<8} {'Correlation':<15} {'MAE':<12} {'Sim Ret':<12} {'BT Ret':<12}")
print("-"*80)
for month, group in df.groupby('Month'):
    if len(group) > 1:
        corr = np.corrcoef(group['Sim_Return'], group['Backtest_Return'])[0, 1]
        mae = np.abs(group['Sim_Return'] - group['Backtest_Return']).mean()
        sim_ret = group['Sim_Return'].sum()
        bt_ret = group['Backtest_Return'].sum()
        print(f"{str(month):<12} {len(group):<8} {corr:>13.4f} {mae:>11.6f} {sim_ret:>11.2%} {bt_ret:>11.2%}")

# Identify problematic periods
print(f"\nProblematic Periods (Correlation < 0.7):")
for month, group in df.groupby('Month'):
    if len(group) > 1:
        corr = np.corrcoef(group['Sim_Return'], group['Backtest_Return'])[0, 1]
        if corr < 0.7:
            print(f"  {month}: Corr={corr:.4f}, Days={len(group)}")

# Cumulative return comparison
df['Sim_Cumulative'] = (1 + df['Sim_Return']).cumprod() - 1
df['BT_Cumulative'] = (1 + df['Backtest_Return']).cumprod() - 1

print(f"\nCumulative Returns:")
print(f"  Simulation: {df['Sim_Cumulative'].iloc[-1]:.2%}")
print(f"  Backtest: {df['BT_Cumulative'].iloc[-1]:.2%}")
print(f"  Difference: {abs(df['Sim_Cumulative'].iloc[-1] - df['BT_Cumulative'].iloc[-1]):.2%}")

# Load holdings to check consistency
with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_holdings.json", 'r', encoding='utf-8') as f:
    holdings = json.load(f)

avg_holdings = np.mean([len(day['holdings']) for day in holdings])
avg_cash_pct = np.mean([day['cash'] / day['equity'] for day in holdings])

print(f"\nHoldings Analysis:")
print(f"  Average Holdings: {avg_holdings:.1f} / 10")
print(f"  Average Cash %: {avg_cash_pct:.2%}")
print(f"  Capital Utilization: {(1 - avg_cash_pct):.2%}")

print("\n" + "="*80)
