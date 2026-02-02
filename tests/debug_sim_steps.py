
import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import logging
import os
import json
import torch
from model_core.data_loader import CBDataLoader
from model_core.config import RobustConfig
from model_core.vm import StackVM
from model_core.backtest import CBBacktest
from strategy_manager.cb_runner import CBStrategyRunner
from strategy_manager.cb_portfolio import CBPortfolioManager
from tests.verify_strategy import SimAccount, MockTrader, DailyRecord

# Configure logging to see details
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("DebugSimulation")

def debug_sim():
    loader = CBDataLoader()
    loader.load_data()
    
    with open(r"c:\Trading\Projects\AlphaGPT\model_core\best_cb_formula.json", 'r') as f:
        formula = json.load(f)['best']['formula']
    
    dates = loader.dates_list[346:350] # 2024-01-02 to ...
    
    account = SimAccount(initial_cash=100000.0)
    portfolio = CBPortfolioManager(state_path="temp_debug.json")
    portfolio.clear_all()
    mock_trader = MockTrader()
    runner = CBStrategyRunner(loader=loader, portfolio=portfolio, trader=mock_trader)
    runner.load_strategy()
    
    prev_equity = account.initial_cash
    
    for i, date in enumerate(dates):
        print(f"\n--- Day {i}: {date} ---")
        date_idx = loader.dates_list.index(date)
        
        prices = {}
        for asset_idx, code in enumerate(loader.assets_list):
            prices[code] = loader.raw_data_cache['CLOSE'][date_idx, asset_idx].item()
            
        equity_before = account.get_equity(prices)
        daily_ret = (equity_before - prev_equity) / prev_equity
        
        print(f"Equity Before trading: {equity_before:.2f}")
        print(f"Daily Return: {daily_ret:+.6%}")
        
        # Trading
        mock_trader.orders = []
        runner.run(date=date, simulate=False)
        orders = mock_trader.orders
        print(f"Generated {len(orders)} orders")
        
        for order in orders:
            account.execute_order(order, fee_rate=RobustConfig.FEE_RATE)
            
        equity_after = account.get_equity(prices)
        print(f"Equity After trading: {equity_after:.2f}")
        print(f"Fees paid: {equity_before - equity_after:.2f}")
        
        prev_equity = equity_after

if __name__ == "__main__":
    debug_sim()
