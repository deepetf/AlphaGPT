"""
简化版验证脚本：仅验证选股一致性 (Jaccard Index)

目标：验证 Event-Driven 策略与 Vector Backtest 的 Top-K 选股是否完全一致
"""

import os
import sys
import json
import torch
from typing import List, Set

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model_core.config import RobustConfig
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.vm import StackVM
from strategy_manager.cb_runner import CBStrategyRunner
from strategy_manager.cb_portfolio import CBPortfolioManager
from execution.cb_trader import Order


class MockTrader:
    """Mock Trader"""
    def __init__(self):
        self.orders = []
    
    def submit_orders(self, orders, date):
        self.orders.extend(orders)
        return type('Result', (), {'success': True, 'message': f'OK'})()


def get_topk_from_simulation(runner, loader, date: str, top_k: int) -> Set[str]:
    """从模拟中获取 Top-K 选股"""
    mock_trader = MockTrader()
    runner.trader = mock_trader
    
    # Run strategy
    runner.run(date=date, simulate=False)
    
    # Extract buy orders (these are the selected stocks)
    selected = set()
    for order in mock_trader.orders:
        if str(order.side).upper() == "BUY" or "BUY" in str(order.side).upper():
            selected.add(order.code)
    
    return selected


def get_topk_from_backtest(factors, valid_mask, loader, date: str, top_k: int) -> Set[str]:
    """从回测中获取 Top-K 选股"""
    date_idx = loader.dates_list.index(date)
    
    # Get factors for this date
    date_factors = factors[date_idx, :].clone()
    
    # Apply valid mask
    date_mask = valid_mask[date_idx, :]
    date_factors[~date_mask] = -1e9
    
    # Get top-k
    _, top_indices = torch.topk(date_factors, k=top_k)
    
    selected = set()
    for idx in top_indices:
        code = loader.assets_list[idx.item()]
        selected.add(code)
    
    return selected


def calculate_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """计算 Jaccard Index"""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def main():
    print("="*60)
    print("简化版验证：选股一致性检查")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    loader = CBDataLoader()
    loader.load_data()
    
    # Load formula
    print("[2/5] Loading strategy...")
    strategy_path = os.path.join(project_root, "model_core", "best_cb_formula.json")
    with open(strategy_path, 'r', encoding='utf-8') as f:
        formula = json.load(f)['best']['formula']
    
    print(f"Formula: {' '.join(formula)}")
    
    # Execute formula
    print("[3/5] Executing formula...")
    vm = StackVM()
    factors = vm.execute(formula, loader.feat_tensor.to('cpu'), cs_mask=loader.cs_mask.to('cpu'))
    
    # Create runner
    print("[4/5] Initializing runner...")
    temp_portfolio_path = os.path.join(project_root, "tests", "artifacts", "temp_portfolio_simple.json")
    portfolio = CBPortfolioManager(state_path=temp_portfolio_path)
    portfolio.clear_all()
    
    runner = CBStrategyRunner(
        loader=loader,
        portfolio=portfolio,
        trader=MockTrader()
    )
    runner.load_strategy()
    
    # Test dates
    test_dates = [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-02-01",
        "2024-03-01"
    ]
    
    print(f"\n[5/5] Comparing Top-{RobustConfig.TOP_K} selections...")
    print("="*60)
    
    results = []
    for date in test_dates:
        if date not in loader.dates_list:
            print(f"⚠️  {date}: Not in dataset, skipping")
            continue
        
        # Get selections
        sim_topk = get_topk_from_simulation(runner, loader, date, RobustConfig.TOP_K)
        backtest_topk = get_topk_from_backtest(factors, loader.valid_mask, loader, date, RobustConfig.TOP_K)
        
        # Calculate Jaccard
        jaccard = calculate_jaccard(sim_topk, backtest_topk)
        
        # Find differences
        only_sim = sim_topk - backtest_topk
        only_backtest = backtest_topk - sim_topk
        
        results.append({
            'date': date,
            'jaccard': jaccard,
            'sim_count': len(sim_topk),
            'backtest_count': len(backtest_topk),
            'only_sim': only_sim,
            'only_backtest': only_backtest
        })
        
        status = "✅" if jaccard == 1.0 else "❌"
        print(f"{status} {date}: Jaccard={jaccard:.4f} | Sim={len(sim_topk)} | Backtest={len(backtest_topk)}")
        
        if jaccard < 1.0:
            print(f"   Only in Sim: {list(only_sim)[:3]}...")
            print(f"   Only in Backtest: {list(only_backtest)[:3]}...")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_jaccard = sum(r['jaccard'] for r in results) / len(results) if results else 0
    perfect_matches = sum(1 for r in results if r['jaccard'] == 1.0)
    
    print(f"Average Jaccard Index: {avg_jaccard:.4f}")
    print(f"Perfect Matches: {perfect_matches}/{len(results)}")
    
    if avg_jaccard == 1.0:
        print("\n✅ 验证通过：选股完全一致！")
        print("   → 如果收益率仍有差异，问题在于资金管理或成交逻辑")
    else:
        print(f"\n❌ 验证失败：选股存在差异 (Jaccard={avg_jaccard:.4f})")
        print("   → 需要检查因子计算或时间切片逻辑")
    
    # Save detailed results
    output_path = os.path.join(project_root, "tests", "artifacts", "selection_comparison.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'avg_jaccard': avg_jaccard,
                'perfect_matches': perfect_matches,
                'total_tests': len(results)
            },
            'details': [{
                'date': r['date'],
                'jaccard': r['jaccard'],
                'sim_count': r['sim_count'],
                'backtest_count': r['backtest_count'],
                'only_sim': list(r['only_sim']),
                'only_backtest': list(r['only_backtest'])
            } for r in results]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
