"""
测试 model_core 改造后的数据加载和特征计算

运行: python test_cb_model_core.py
"""
import sys
import os

# 确保能找到 model_core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """测试配置"""
    print("=" * 60)
    print("1. 测试配置")
    print("=" * 60)
    
    from model_core.config import ModelConfig
    
    print(f"CB_PARQUET_PATH: {ModelConfig.CB_PARQUET_PATH}")
    print(f"BASIC_FACTORS: {len(ModelConfig.BASIC_FACTORS)} 个")
    for name, col, method in ModelConfig.BASIC_FACTORS[:5]:
        print(f"  - {name} <- {col} ({method})")
    print(f"  ... 共 {len(ModelConfig.BASIC_FACTORS)} 个")
    print(f"INPUT_FEATURES: {ModelConfig.INPUT_FEATURES}")
    print(f"INPUT_DIM: {ModelConfig.INPUT_DIM}")
    print()
    return True

def test_ops_registry():
    """测试算子注册"""
    print("=" * 60)
    print("2. 测试算子注册")
    print("=" * 60)
    
    from model_core.ops_registry import OpsRegistry
    
    ops = OpsRegistry.list_ops()
    print(f"已注册算子: {len(ops)} 个")
    for op in ops[:10]:
        info = OpsRegistry.get_op(op)
        print(f"  - {op} (arity={info['arity']}): {info['description']}")
    if len(ops) > 10:
        print(f"  ... 共 {len(ops)} 个")
    print()
    return True

def test_data_loader():
    """测试数据加载"""
    print("=" * 60)
    print("3. 测试数据加载")
    print("=" * 60)
    
    from model_core.data_loader import CryptoDataLoader
    
    loader = CryptoDataLoader()
    loader.load_data()
    
    print(f"raw_data_cache keys: {list(loader.raw_data_cache.keys())}")
    print(f"feat_tensor shape: {loader.feat_tensor.shape}")
    print(f"target_ret shape: {loader.target_ret.shape}")
    print(f"valid_mask shape: {loader.valid_mask.shape}")
    print(f"assets count: {len(loader.assets_list)}")
    print()
    return True

def test_alphagpt():
    """测试 AlphaGPT 模型"""
    print("=" * 60)
    print("4. 测试 AlphaGPT 模型")
    print("=" * 60)
    
    from model_core.alphagpt import AlphaGPT
    
    model = AlphaGPT()
    print(f"Vocab size: {model.vocab_size}")
    print(f"Features: {model.features_list}")
    print(f"Ops (前10): {model.ops_list[:10]}...")
    print(f"Total vocab: {model.vocab[:15]}...")
    print()
    return True

def test_vm():
    """测试 StackVM"""
    print("=" * 60)
    print("5. 测试 StackVM")
    print("=" * 60)
    
    import torch
    from model_core.vm import StackVM
    from model_core.config import ModelConfig
    
    vm = StackVM()
    print(f"feat_offset: {vm.feat_offset}")
    print(f"op_map keys (前10): {list(vm.op_map.keys())[:10]}...")
    
    # 构造一个简单的测试张量 [Time=10, Assets=5, Features=6]
    feat_tensor = torch.randn(10, 5, ModelConfig.INPUT_DIM, device=ModelConfig.DEVICE)
    
    # 测试公式: 取第0个特征 (相当于直接输出 CLOSE)
    formula = [0]
    result = vm.execute(formula, feat_tensor)
    print(f"Formula [0] result shape: {result.shape if result is not None else 'None'}")
    
    # 测试公式: ADD(特征0, 特征1)
    # ADD 在 ops_registry 中是第一个注册的算子，token = feat_offset + 0
    formula = [0, 1, vm.feat_offset + 0]  # CLOSE, VOL, ADD
    result = vm.execute(formula, feat_tensor)
    print(f"Formula [0, 1, ADD] result shape: {result.shape if result is not None else 'None'}")
    
    print()
    return True

def main():
    results = []
    
    try:
        results.append(("Config", test_config()))
    except Exception as e:
        print(f"Config 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Config", False))
    
    try:
        results.append(("OpsRegistry", test_ops_registry()))
    except Exception as e:
        print(f"OpsRegistry 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("OpsRegistry", False))
    
    try:
        results.append(("DataLoader", test_data_loader()))
    except Exception as e:
        print(f"DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DataLoader", False))
    
    try:
        results.append(("AlphaGPT", test_alphagpt()))
    except Exception as e:
        print(f"AlphaGPT 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("AlphaGPT", False))
    
    try:
        results.append(("StackVM", test_vm()))
    except Exception as e:
        print(f"StackVM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("StackVM", False))
    
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    if all(r[1] for r in results):
        print("\n🎉 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")

if __name__ == "__main__":
    main()
