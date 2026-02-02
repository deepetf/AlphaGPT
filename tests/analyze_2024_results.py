"""
生成2024全年策略运行详细分析报告
"""
import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

import json
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
returns_df = pd.read_csv(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_returns.csv")
with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_trades.json", 'r', encoding='utf-8') as f:
    trades = json.load(f)
with open(r"c:\Trading\Projects\AlphaGPT\tests\artifacts\daily_holdings.json", 'r', encoding='utf-8') as f:
    holdings = json.load(f)

print("="*80)
print("2024年度策略运行详细分析报告")
print("="*80)

# 1. 月度收益统计
print("\n## 月度收益统计\n")
returns_df['Date'] = pd.to_datetime(returns_df['Date'])
returns_df['Month'] = returns_df['Date'].dt.to_period('M')

monthly_stats = returns_df.groupby('Month').agg({
    'Sim_Return': lambda x: (1 + x).prod() - 1,
    'Date': 'count'
}).rename(columns={'Sim_Return': 'Monthly_Return', 'Date': 'Trading_Days'})

print(f"{'月份':<12} {'月度收益':<12} {'交易天数':<10}")
print("-" * 40)
for month, row in monthly_stats.iterrows():
    print(f"{str(month):<12} {row['Monthly_Return']:>10.2%} {row['Trading_Days']:>10.0f}")

# 2. 交易统计
print("\n## 交易统计\n")
total_buys = sum(len(day['buys']) for day in trades)
total_sells = sum(len(day['sells']) for day in trades)
total_buy_amount = sum(sum(order['amount'] for order in day['buys']) for day in trades)
total_sell_amount = sum(sum(order['amount'] for order in day['sells']) for day in trades)

print(f"总买入次数: {total_buys}")
print(f"总卖出次数: {total_sells}")
print(f"总买入金额: {total_buy_amount:,.2f} 元")
print(f"总卖出金额: {total_sell_amount:,.2f} 元")
print(f"平均每日交易: {(total_buys + total_sells) / len(trades):.1f} 笔")

# 3. 持仓分析
print("\n## 持仓分析\n")
avg_holdings = np.mean([len(day['holdings']) for day in holdings])
avg_cash = np.mean([day['cash'] for day in holdings])
avg_equity = np.mean([day['equity'] for day in holdings])
cash_utilization = (avg_equity - avg_cash) / avg_equity

print(f"平均持仓数量: {avg_holdings:.1f} 只")
print(f"平均现金余额: {avg_cash:,.2f} 元")
print(f"平均账户权益: {avg_equity:,.2f} 元")
print(f"资金利用率: {cash_utilization:.2%}")

# 4. 风险指标
print("\n## 风险指标\n")
returns_arr = returns_df['Sim_Return'].values
daily_vol = returns_arr.std()
ann_vol = daily_vol * np.sqrt(252)
downside_returns = returns_arr[returns_arr < 0]
downside_vol = downside_returns.std() if len(downside_returns) > 0 else 0
sortino = (returns_arr.mean() / (downside_vol + 1e-9)) * np.sqrt(252)

print(f"日波动率: {daily_vol:.4f}")
print(f"年化波动率: {ann_vol:.2%}")
print(f"索提诺比率: {sortino:.4f}")
print(f"负收益天数: {len(downside_returns)} / {len(returns_arr)} ({len(downside_returns)/len(returns_arr):.1%})")

# 5. 最佳/最差交易日
print("\n## 最佳/最差交易日\n")
best_day = returns_df.loc[returns_df['Sim_Return'].idxmax()]
worst_day = returns_df.loc[returns_df['Sim_Return'].idxmin()]

print(f"最佳交易日: {best_day['Date'].strftime('%Y-%m-%d')}, 收益率: {best_day['Sim_Return']:+.2%}")
print(f"最差交易日: {worst_day['Date'].strftime('%Y-%m-%d')}, 收益率: {worst_day['Sim_Return']:+.2%}")

# 6. 持仓周转率
print("\n## 持仓周转分析\n")
turnover_days = sum(1 for day in trades if len(day['buys']) > 0 or len(day['sells']) > 0)
avg_turnover_per_rebalance = (total_buys + total_sells) / turnover_days if turnover_days > 0 else 0

print(f"发生调仓天数: {turnover_days} / {len(trades)} ({turnover_days/len(trades):.1%})")
print(f"平均每次调仓交易数: {avg_turnover_per_rebalance:.1f} 笔")

# 7. 收益分布
print("\n## 收益分布\n")
positive_days = len(returns_arr[returns_arr > 0])
negative_days = len(returns_arr[returns_arr < 0])
zero_days = len(returns_arr[returns_arr == 0])

print(f"正收益天数: {positive_days} ({positive_days/len(returns_arr):.1%})")
print(f"负收益天数: {negative_days} ({negative_days/len(returns_arr):.1%})")
print(f"零收益天数: {zero_days} ({zero_days/len(returns_arr):.1%})")

print("\n" + "="*80)
print("报告生成完成")
print("="*80)
