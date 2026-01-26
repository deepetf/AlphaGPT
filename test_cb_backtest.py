"""
测试 CBBacktest 回测引擎

验证:
1. 回测逻辑是否正确
2. 使用一个简单因子 (如双低) 测试回测效果
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.data_loader import CryptoDataLoader
from model_core.backtest import CBBacktest


def test_backtest_with_dblow():
    """
    使用双低因子测试回测
    双低越低越好，所以我们用 -DBLOW 作为因子 (值越高越好)
    """
    print("=" * 60)
    print("测试 CBBacktest: 双低因子回测")
    print("=" * 60)
    
    # 1. 加载数据
    loader = CryptoDataLoader()
    loader.load_data()
    
    # 2. 构建因子: -DBLOW (双低越低得分越高)
    dblow = loader.raw_data_cache['DBLOW']
    factor = -dblow  # 取负，这样双低越低，因子值越高，越容易被选入 Top-K
    
    print(f"Factor shape: {factor.shape}")
    print(f"Factor sample (last day): min={factor[-1].min():.2f}, max={factor[-1].max():.2f}")
    
    # 3. 运行回测
    bt = CBBacktest(top_k=20, fee_rate=0.0001)
    
    reward, cum_ret = bt.evaluate(
        factors=factor,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask
    )
    
    print(f"\n回测结果:")
    print(f"  Reward (Sharpe * 10): {reward:.2f}")
    print(f"  累计收益率: {cum_ret:.2%}")
    
    return True


def test_backtest_with_random():
    """
    使用随机因子测试回测 (应该接近0收益)
    """
    print("\n" + "=" * 60)
    print("测试 CBBacktest: 随机因子回测")
    print("=" * 60)
    
    # 1. 加载数据
    loader = CryptoDataLoader()
    loader.load_data()
    
    # 2. 构建因子: 随机
    T, N = loader.target_ret.shape
    factor = torch.randn(T, N, device=loader.target_ret.device)
    
    # 3. 运行回测
    bt = CBBacktest(top_k=20, fee_rate=0.0001)
    
    reward, cum_ret = bt.evaluate(
        factors=factor,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask
    )
    
    print(f"\n回测结果:")
    print(f"  Reward (Sharpe * 10): {reward:.2f}")
    print(f"  累计收益率: {cum_ret:.2%}")
    
    return True


def test_backtest_with_vm():
    """
    测试从 VM 执行公式到回测的完整流程
    """
    print("\n" + "=" * 60)
    print("测试完整流程: VM -> Backtest")
    print("=" * 60)
    
    from model_core.vm import StackVM
    from model_core.config import ModelConfig
    
    # 1. 加载数据
    loader = CryptoDataLoader()
    loader.load_data()
    
    # 2. 初始化 VM
    vm = StackVM()
    
    # 3. 构造一个简单公式: 取 DBLOW 特征然后取负
    # DBLOW 在 INPUT_FEATURES 中是第 3 个 (index=3)
    # NEG 在 OpsRegistry 中是第 4 个 (index = feat_offset + 4)
    feat_idx = ModelConfig.INPUT_FEATURES.index('DBLOW')
    neg_idx = vm.feat_offset + 4  # NEG 算子
    
    formula = [feat_idx, neg_idx]  # DBLOW, NEG -> -DBLOW
    print(f"Formula tokens: {formula}")
    
    # 4. 执行公式
    result = vm.execute(formula, loader.feat_tensor)
    
    if result is None:
        print("VM 执行失败!")
        return False
    
    print(f"VM result shape: {result.shape}")
    
    # 5. 回测
    bt = CBBacktest(top_k=20, fee_rate=0.0001)
    
    reward, cum_ret = bt.evaluate(
        factors=result,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask
    )
    
    print(f"\n回测结果:")
    print(f"  Reward (Sharpe * 10): {reward:.2f}")
    print(f"  累计收益率: {cum_ret:.2%}")
    
    return True


if __name__ == "__main__":
    results = []
    
    try:
        results.append(("双低因子回测", test_backtest_with_dblow()))
    except Exception as e:
        print(f"双低因子回测失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("双低因子回测", False))
    
    try:
        results.append(("随机因子回测", test_backtest_with_random()))
    except Exception as e:
        print(f"随机因子回测失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("随机因子回测", False))
    
    try:
        results.append(("VM完整流程", test_backtest_with_vm()))
    except Exception as e:
        print(f"VM完整流程失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("VM完整流程", False))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
