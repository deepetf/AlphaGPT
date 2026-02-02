import json
import os
import pandas as pd
import numpy as np

def analyze_turnover():
    artifacts_dir = r"c:\Trading\Projects\AlphaGPT\tests\artifacts"
    trades_file = os.path.join(artifacts_dir, "daily_trades.json")
    holdings_file = os.path.join(artifacts_dir, "daily_holdings.json")
    
    with open(trades_file, 'r', encoding='utf-8') as f:
        trades = json.load(f)
        
    with open(holdings_file, 'r', encoding='utf-8') as f:
        holdings = json.load(f)
        
    # Map date to equity
    equity_map = {h['date']: h['equity'] for h in holdings}
    
    daily_turnovers = []
    
    # Skip first day (building position)
    if not trades:
        print("No trades found")
        return

    start_date = trades[0]['date']
    
    for t_record in trades:
        date = t_record['date']
        if date == start_date:
            continue
            
        buy_amt = sum(b['amount'] for b in t_record['buys'])
        sell_amt = sum(s['amount'] for s in t_record['sells'])
        
        equity = equity_map.get(date, 0)
        if equity > 0:
            # Standard definition: min(buy, sell) / equity for pure rebalancing turnover
            # turnover = min(buy_amt, sell_amt) / equity
            
            # Using (Buy+Sell)/2
            turnover = (buy_amt + sell_amt) / 2 / equity
            daily_turnovers.append(turnover)
            
    if not daily_turnovers:
        print("No turnover data")
        return

    avg_daily_to = np.mean(daily_turnovers)
    annual_to = avg_daily_to * 242
    
    print(f"Analysis Period: {len(daily_turnovers)} days")
    print(f"Average Daily Turnover: {avg_daily_to*100:.2f}%")
    print(f"Annualized Turnover: {annual_to*100:.2f}%")
    print(f"Max Daily Turnover: {max(daily_turnovers)*100:.2f}%")
    
if __name__ == "__main__":
    analyze_turnover()
