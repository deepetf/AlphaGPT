import sys
import os
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.data_loader import CryptoDataLoader

def check_valid_count():
    loader = CryptoDataLoader()
    loader.load_data()
    
    # 获取有效数量序列
    valid_counts = loader.valid_mask.sum(dim=1).cpu().numpy()
    
    print("\n" + "="*50)
    print("有效转债数量统计")
    print("="*50)
    print(f"最大单日数量: {valid_counts.max()}")
    print(f"平均单日数量: {valid_counts.mean():.1f}")
    print(f"最近单日数量: {valid_counts[-1]}")
    
    # 打印最后几天的具体数量
    print("\n最近5个交易日的有效数量:")
    for i in range(1, 6):
        print(f"T-{i}: {valid_counts[-i]}")

if __name__ == "__main__":
    check_valid_count()
