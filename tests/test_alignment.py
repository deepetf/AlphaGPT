"""
验证收益率时间错位假设
"""
import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import pandas as pd
import numpy as np

# Load simulation returns
sim_df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")

print("Current alignment (from CSV):")
print("="*80)
print(sim_df.head(10).to_string())

print("\n\nHypothesis: Simulation returns are 1 day ahead")
print("="*80)
print("If we shift simulation returns by 1 day forward:")
print()

# Shift simulation returns forward by 1
sim_df['Sim_Return_Shifted'] = sim_df['Sim_Return'].shift(-1)
sim_df['Diff_Shifted'] = sim_df['Sim_Return_Shifted'] - sim_df['Backtest_Return']

print(sim_df[['Date', 'Sim_Return', 'Sim_Return_Shifted', 'Backtest_Return', 'Diff_Shifted']].head(10).to_string())

# Calculate correlation with shifted data
sim_shifted = sim_df['Sim_Return_Shifted'].dropna().values
backtest = sim_df['Backtest_Return'].iloc[:-1].values

correlation_shifted = np.corrcoef(sim_shifted, backtest)[0, 1]
mae_shifted = np.abs(sim_shifted - backtest).mean()

print(f"\n\nMetrics with 1-day shift:")
print(f"Correlation: {correlation_shifted:.6f} (original: 0.898)")
print(f"MAE: {mae_shifted:.6f} (original: 0.0026)")

if correlation_shifted > 0.95:
    print("\n✅ HYPOTHESIS CONFIRMED: Returns are 1-day misaligned!")
else:
    print("\n❌ HYPOTHESIS REJECTED: Problem is elsewhere")
