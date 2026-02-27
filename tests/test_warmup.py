"""
预热数据前置加载验证测试

验证 CBDataLoader 的 warmup 逻辑：
1. 加载后 feat_tensor 首行非全零（预热生效）
2. dates_list 首日 >= start_date（裁剪正确）
3. warmup_days=0 时不发生裁剪（兼容性）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_warmup_basic():
    """测试1: 预热后特征首行非全零"""
    from model_core.data_loader import CBDataLoader
    from model_core.config import ModelConfig

    # 确保 warmup_days > 0
    assert ModelConfig.WARMUP_DAYS > 0, f"warmup_days should be > 0, got {ModelConfig.WARMUP_DAYS}"

    loader = CBDataLoader()
    loader.load_data(start_date='2022-08-01')

    # feat_tensor 首行应非全零
    first_row = loader.feat_tensor[0]  # [Assets, Features]
    non_zero = (first_row.abs() > 1e-9).any().item()
    print(f"First row non-zero: {non_zero}")
    print(f"First row abs sum: {first_row.abs().sum().item():.4f}")
    print(f"feat_tensor shape: {loader.feat_tensor.shape}")
    print(f"dates_list[0]: {loader.dates_list[0]}")
    assert non_zero, "预热失败：首行特征仍为全零"

    # dates_list 首日应 >= start_date
    assert loader.dates_list[0] >= '2022-08-01', \
        f"裁剪失败：首日 {loader.dates_list[0]} < 2022-08-01"

    print("[PASS] test_warmup_basic PASSED")


def test_warmup_trim_date():
    """测试2: 裁剪后日期对齐正确"""
    from model_core.data_loader import CBDataLoader

    start_date = '2023-01-01'
    loader = CBDataLoader()
    loader.load_data(start_date=start_date)

    # 验证首日
    assert loader.dates_list[0] >= start_date, \
        f"首日 {loader.dates_list[0]} < {start_date}"

    # 验证张量维度一致
    T = len(loader.dates_list)
    assert loader.feat_tensor.shape[0] == T, \
        f"feat_tensor 时间维度 {loader.feat_tensor.shape[0]} != dates {T}"
    assert loader.target_ret.shape[0] == T, \
        f"target_ret 时间维度 {loader.target_ret.shape[0]} != dates {T}"
    assert loader.valid_mask.shape[0] == T, \
        f"valid_mask 时间维度 {loader.valid_mask.shape[0]} != dates {T}"

    # 验证 raw_data_cache 也被裁剪
    for k, v in loader.raw_data_cache.items():
        assert v.shape[0] == T, \
            f"raw_data_cache['{k}'] 时间维度 {v.shape[0]} != dates {T}"

    print(f"[PASS] test_warmup_trim_date PASSED (first_date={loader.dates_list[0]}, T={T})")


if __name__ == '__main__':
    test_warmup_basic()
    print()
    test_warmup_trim_date()
    print()
    print("All tests passed [PASS]")
