
import torch
import json
import os
import pandas as pd
from model_core.config import RobustConfig
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.vm import StackVM
from model_core.ops_registry import OpsRegistry

def reproduce_king9():
    print("🚀 Reproducing King #9 (Sharpe 2.81)...")
    
    # 1. 准备数据
    loader = CBDataLoader()
    loader.load_data()
    
    # 2. 准备公式
    # "REMAIN_SIZE TS_DELTA LOG LOG CUT_HIGH TS_STD5 LOG SIGN LOG LOG CLOSE SUB"
    # Token IDs 会变，但我们可以用人类可读字符串重新映射
    readable_formula = "REMAIN_SIZE TS_DELTA LOG LOG CUT_HIGH TS_STD5 LOG SIGN LOG LOG CLOSE SUB"
    ops_list = readable_formula.split()
    
    # 将字符串转换为 Token ID (模拟 AlphaGPT 的输出)
    # 其实 StackVM 可以稍作修改直接支持字符串列表，或者我们手动把字符串列表传进去
    # 这里我们不用 Token ID，直接用 StackVM 内部调用的逻辑 (StackVM.execute 也支持 token list 如果我们魔改一下，
    # 但原版 execute 接收的是 tensor/long list。
    # 最简单的办法：手动构造一个临时的 OpsMap)
    
    # 我们直接手动执行这一串算子，绕过 StackVM 的 Token 解析，确保逻辑绝对一致
    print(f"Executing Formula: {readable_formula}")
    
    # Stack 模拟
    stack = []
    
    # 辅助函数：获取 Tensor
    def get_feature(name):
        # 尝试从输入特征索引映射
        # 这里直接从 raw_data_cache 取最稳
        return loader.raw_data_cache[name].clone()

    for token in ops_list:
        if token in loader.raw_data_cache:
            # 是特征
            stack.append(get_feature(token))
        else:
            # 是算子
            op_info = OpsRegistry.get_op(token)
            if not op_info:
                raise ValueError(f"Unknown op: {token}")
            
            func = op_info['func']
            arity = op_info['arity']
            
            if len(stack) < arity:
                print(f"Error: Stack underflow for {token}")
                return
            
            args = stack[-arity:]
            stack = stack[:-arity]
            
            res = func(*args)
            stack.append(res)
            
    final_factor = stack[0]
    
    # 3. 回测
    print("\nRunning Backtest (Top-K=10)...")
    bt = CBBacktest(top_k=RobustConfig.TOP_K)
    
    # 获取详细记录
    details = bt.evaluate_with_details(
        factors=final_factor,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask
    )
    
    print(f"\n✅ Reproduction Result:")
    print(f"Score:  {details['reward']:.2f}")
    print(f"Sharpe: {details['sharpe']:.2f}")
    print(f"Return: {details['cum_ret']:.2%}")
    
    # 4. 保存交易记录
    output_path = 'model_core/king9_reproduction.json'
    trades = []
    
    for t, (indices, daily_ret) in enumerate(zip(details['daily_holdings'], details['daily_returns'])):
        if t >= len(loader.dates_list): break
        
        date = loader.dates_list[t]
        
        # 将资产索引转换为名称
        holdings = []
        for idx in indices:
            if idx < len(loader.assets_list):
                code = loader.assets_list[idx]
                name = loader.names_dict.get(code, code)
                holdings.append(name)
        
        trades.append({
            'date': date,
            'holdings': holdings,
            'daily_ret': round(daily_ret, 6)
        })
        
    result = {
        'formula': readable_formula,
        'metrics': {
            'score': details['reward'],
            'sharpe': details['sharpe'],
            'return': details['cum_ret']
        },
        'trades': trades
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    print(f"\nDetailed trades saved to: {output_path}")

if __name__ == "__main__":
    reproduce_king9()
