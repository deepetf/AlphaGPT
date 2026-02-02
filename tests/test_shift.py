import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import pandas as pd
import numpy as np

sim_df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")

print("Testing 1-day shift hypothesis...")

# Original correlation
sim_orig = sim_df['Sim_Return'].values
backtest_orig = sim_df['Backtest_Return'].values
corr_orig = np.corrcoef(sim_orig, backtest_orig)[0, 1]
mae_orig = np.abs(sim_orig - backtest_orig).mean()

print(f"Original: Corr={corr_orig:.4f}, MAE={mae_orig:.6f}")

# Shift simulation forward by 1 day
sim_shifted = sim_df['Sim_Return'].shift(-1).dropna().values
backtest_for_shifted = sim_df['Backtest_Return'].iloc[:-1].values

corr_shifted = np.corrcoef(sim_shifted, backtest_for_shifted)[0, 1]
mae_shifted = np.abs(sim_shifted - backtest_for_shifted).mean()

print(f"Shifted:  Corr={corr_shifted:.4f}, MAE={mae_shifted:.6f}")

if corr_shifted > corr_orig + 0.05:
    print("\nCONFIRMED: 1-day misalignment detected!")
else:
    print("\nREJECTED: Not a 1-day shift issue")
