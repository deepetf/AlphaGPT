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

start_idx = loader.dates_list.index("2024-01-02")
end_idx = loader.dates_list.index("2024-12-31") + 1

backtest_returns_2024 = result['daily_returns'][start_idx:end_idx]
backtest_returns_arr = np.array(backtest_returns_2024)
backtest_returns_arr = backtest_returns_arr[backtest_returns_arr != 0]

cum_ret_backtest = (1 + backtest_returns_arr).prod() - 1
ann_ret_backtest = (1 + cum_ret_backtest) ** (252 / len(backtest_returns_arr)) - 1

sim_df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")
sim_returns_arr = sim_df['Sim_Return'].values[1:]

cum_ret_sim = (1 + sim_returns_arr).prod() - 1
ann_ret_sim = (1 + cum_ret_sim) ** (252 / len(sim_returns_arr)) - 1

print(f"Backtest 2024: Cum={cum_ret_backtest:.4f}, Ann={ann_ret_backtest:.4f}")
print(f"Simulation 2024: Cum={cum_ret_sim:.4f}, Ann={ann_ret_sim:.4f}")
print(f"Difference: Cum={abs(cum_ret_backtest - cum_ret_sim):.4f}, Ann={abs(ann_ret_backtest - ann_ret_sim):.4f}")
