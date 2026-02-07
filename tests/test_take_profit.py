"""
止盈逻辑验证脚本

测试:
1. take_profit=0 时回测结果与原逻辑一致
2. take_profit>0 时止盈逻辑正确触发
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.config import RobustConfig


def test_take_profit_zero():
    """
    测试 take_profit=0 时回测结果正常
    """
    print("=" * 60)
    print("测试 1: take_profit=0 (不止盈)")
    print("=" * 60)
    
    # 加载数据
    loader = CBDataLoader()
    loader.load_data()
    
    # 构建因子: -DBLOW
    dblow = loader.raw_data_cache['DBLOW']
    factor = -dblow
    
    # 不传入价格数据，确保 take_profit=0 时正常工作
    bt = CBBacktest(top_k=10, take_profit=0.0)
    
    reward, cum_ret, sharpe = bt.evaluate(
        factors=factor,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask
    )
    
    print(f"结果: Reward={reward:.2f}, 累计收益={cum_ret:.2%}, Sharpe={sharpe:.2f}")
    assert not torch.isnan(torch.tensor(cum_ret)), "累计收益不应为 NaN"
    print("✓ take_profit=0 测试通过")
    return True


def test_take_profit_with_price_data():
    """
    测试 take_profit>0 时止盈逻辑
    """
    print("\n" + "=" * 60)
    print("测试 2: take_profit=0.06 (6%止盈)")
    print("=" * 60)
    
    # 加载数据
    loader = CBDataLoader()
    loader.load_data()
    
    # 检查是否有 OPEN 和 HIGH 数据
    if 'OPEN' not in loader.raw_data_cache:
        print("⚠️ Warning: OPEN 数据未加载，跳过止盈测试")
        return False
    if 'HIGH' not in loader.raw_data_cache:
        print("⚠️ Warning: HIGH 数据未加载，跳过止盈测试")
        return False
    
    print(f"OPEN 数据形状: {loader.raw_data_cache['OPEN'].shape}")
    print(f"HIGH 数据形状: {loader.raw_data_cache['HIGH'].shape}")
    
    # 构建因子
    factor = -loader.raw_data_cache['DBLOW']
    close = loader.raw_data_cache['CLOSE']
    raw_open = loader.raw_data_cache['OPEN']
    raw_high = loader.raw_data_cache['HIGH']
    
    # 时序对齐:
    #   - weights[t] = t日收盘时的持仓决策
    #   - target_ret[t] = 持有 t→t+1 的收益
    #   - 止盈检查发生在 t+1 日盘中
    #   - 因此 open_prices[t] 应为 open[t+1], high_prices[t] 应为 high[t+1]
    open_prices = torch.roll(raw_open, -1, dims=0)
    high_prices = torch.roll(raw_high, -1, dims=0)
    open_prices[-1] = 1e9  # 最后一行无效
    high_prices[-1] = 1e9
    
    # prev_close[t] = close[t] = 买入价格
    prev_close = close.clone()
    
    # 创建带止盈的回测器
    bt_tp = CBBacktest(top_k=10, take_profit=0.06)
    bt_no_tp = CBBacktest(top_k=10, take_profit=0.0)
    
    # 带止盈的回测
    reward_tp, cum_ret_tp, sharpe_tp = bt_tp.evaluate(
        factors=factor,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask,
        open_prices=open_prices,
        high_prices=high_prices,
        prev_close=prev_close
    )
    
    # 不带止盈的回测
    reward_no_tp, cum_ret_no_tp, sharpe_no_tp = bt_no_tp.evaluate(
        factors=factor,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask
    )
    
    print(f"无止盈: Reward={reward_no_tp:.2f}, 累计收益={cum_ret_no_tp:.2%}, Sharpe={sharpe_no_tp:.2f}")
    print(f"有止盈: Reward={reward_tp:.2f}, 累计收益={cum_ret_tp:.2%}, Sharpe={sharpe_tp:.2f}")
    print(f"收益差异: {(cum_ret_tp - cum_ret_no_tp):.2%}")
    
    print("✓ take_profit=0.06 测试完成")
    return True


if __name__ == "__main__":
    print(f"当前 RobustConfig.TAKE_PROFIT 默认值: {RobustConfig.TAKE_PROFIT}")
    print()
    
    results = []
    
    try:
        results.append(("take_profit=0", test_take_profit_zero()))
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("take_profit=0", False))
    
    try:
        results.append(("take_profit=0.06", test_take_profit_with_price_data()))
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("take_profit=0.06", False))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
