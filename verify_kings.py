"""
King 因子验证器 (King Factor Validator)

用法:
    python verify_kings.py                 # 验证所有 Kings
    python verify_kings.py --king 3        # 只验证 King #3
    python verify_kings.py --formula "CLOSE NEG"  # 验证自定义公式
"""
import torch
import json
import os
import argparse
from model_core.config import RobustConfig
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.ops_registry import OpsRegistry

from model_core.vm import StackVM

# 全局加载数据 (避免重复加载)
_loader = None
_vm = None

def get_loader():
    global _loader, _vm
    if _loader is None:
        print("Loading data...")
        _loader = CBDataLoader()
        _loader.load_data()
        _vm = StackVM()
    return _loader

def execute_formula(readable_formula: str, loader) -> torch.Tensor:
    """
    使用 StackVM 执行 RPN 格式的因子公式 (与训练时完全一致)
    
    Args:
        readable_formula: 空格分隔的 token 字符串，如 "CLOSE NEG"
    
    Returns:
        因子值 Tensor [Time, Assets]
    """
    global _vm
    if _vm is None:
        _vm = StackVM()
    
    formula_list = readable_formula.split()
    
    try:
        result = _vm.execute(formula_list, loader.feat_tensor)
        return result
    except Exception as e:
        print(f"❌ Execution error: {e}")
        return None

def verify_formula(readable_formula: str, label: str = "", save_trades: bool = True):
    """验证单个因子公式"""
    loader = get_loader()
    
    print(f"\n{'='*60}")
    print(f"🔍 Verifying: {label}")
    print(f"   Formula: {readable_formula}")
    print('='*60)
    
    # 执行公式
    factor = execute_formula(readable_formula, loader)
    if factor is None:
        print("❌ Execution failed")
        return None
    
    # 统计因子信息 (使用 nanmean 避免 NaN 影响)
    valid_factor = factor[loader.valid_mask]
    valid_factor_clean = valid_factor[~torch.isnan(valid_factor)]
    if len(valid_factor_clean) > 0:
        mean_val = valid_factor_clean.mean().item()
        std_val = valid_factor_clean.std().item()
    else:
        mean_val, std_val = float('nan'), float('nan')
    print(f"   Factor Stats: Mean={mean_val:.4f}, Std={std_val:.4f}")
    print(f"   NaN Ratio: {torch.isnan(factor).sum().item() / factor.numel():.2%}")
    
    # 回测 (使用 RobustConfig 统一配置参数)
    bt = CBBacktest(top_k=RobustConfig.TOP_K)
    
    # 传统回测 (用于保存交易记录)
    details = bt.evaluate_with_details(
        factors=factor,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask
    )
    
    # 稳健性评估 (用于显示分段指标)
    robust_metrics = bt.evaluate_robust(
        factors=factor,
        target_ret=loader.target_ret,
        valid_mask=loader.valid_mask,
        split_idx=loader.split_idx
    )
    
    # 手动计算 Composite Score (复用 engine.py 逻辑)
    # 若被 Hard Filter 淘汰，分数可能不准确，这里主要展示 Soft Score

    base_score = (RobustConfig.TRAIN_WEIGHT * robust_metrics['sharpe_train'] + 
                  RobustConfig.VAL_WEIGHT * robust_metrics['sharpe_val'])
    stability_bonus = robust_metrics['stability_metric'] * RobustConfig.STABILITY_W
    mdd_penalty = robust_metrics['max_drawdown'] * RobustConfig.MDD_W
    # 假设公式长度为 readable_formula 分词后的长度
    formula_len = len(readable_formula.split())
    len_penalty = formula_len * RobustConfig.LEN_W
    
    composite_score = (base_score + stability_bonus) * RobustConfig.SCALE - mdd_penalty - len_penalty

    print(f"\n✅ Backtest Result:")
    print(f"   Composite Score: {composite_score:.2f} (New Reward)")
    print(f"   Score (Legacy):  {details['reward']:.2f}")
    print(f"   Sharpe (All): {details['sharpe']:.2f}")
    print(f"   Sharpe (Train/Val): {robust_metrics['sharpe_train']:.2f} / {robust_metrics['sharpe_val']:.2f}")
    print(f"   Max Drawdown: {robust_metrics['max_drawdown']:.1%}")
    print(f"   Stability: {robust_metrics['stability_metric']:.2f} (Sharpe Std: {robust_metrics['sharpe_std']:.2f})")
    print(f"   Active Ratio: {robust_metrics['active_ratio']:.1%}")
    print(f"   Return: {details['cum_ret']:.2%}")
    
    # 保存交易记录
    if save_trades:
        trades_dir = 'model_core/verified_trades'
        os.makedirs(trades_dir, exist_ok=True)
        
        # 日期对齐检查
        holdings_len = len(details['daily_holdings'])
        dates_len = len(loader.dates_list)
        if holdings_len != dates_len:
            print(f"   ⚠️ Warning: Length mismatch - holdings={holdings_len}, dates={dates_len}")
        
        trades = []
        for t, (indices, daily_ret) in enumerate(zip(details['daily_holdings'], details['daily_returns'])):
            if t >= len(loader.dates_list): break
            
            date = loader.dates_list[t]
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
            'label': label,
            'formula': readable_formula,
            'metrics': {
                'score': details['reward'],
                'sharpe': details['sharpe'],
                'return': details['cum_ret']
            },
            'trades': trades
        }
        
        safe_label = label.replace('#', '').replace(' ', '_').lower()
        file_path = os.path.join(trades_dir, f'{safe_label}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"   Trades saved to: {file_path}")
    
    return details

def verify_all_kings():
    """验证 best_cb_formula.json 中的所有 Kings"""
    json_path = 'model_core/best_cb_formula.json'
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    history = data.get('history', [])
    if not history:
        print("❌ No kings found in history")
        return
    
    print(f"Found {len(history)} Kings to verify\n")
    
    results = []
    for i, king in enumerate(history):
        # 支持新格式 (formula 字段) 和旧格式 (readable 字段)
        formula = king.get('formula') or king.get('readable', '')
        if isinstance(formula, list):
            readable = ' '.join(formula)
        else:
            readable = formula
        
        original_score = king.get('score', 0)
        original_sharpe = king.get('sharpe', 0)
        
        label = f"King #{i+1}"
        details = verify_formula(readable, label, save_trades=True)
        
        if details:
            results.append({
                'king': i+1,
                'original_score': original_score,
                'original_sharpe': original_sharpe,
                'verified_score': details['reward'],
                'verified_sharpe': details['sharpe'],
                'match': abs(original_sharpe - details['sharpe']) < 0.1
            })
    
    # 汇总报告
    print("\n" + "="*80)
    print("📊 VERIFICATION SUMMARY")
    print("="*80)
    print(f"{'King':<10} {'Orig Score':<12} {'Orig Sharpe':<12} {'Verified Score':<15} {'Verified Sharpe':<15} {'Match':<10}")
    print("-"*80)
    
    for r in results:
        match_str = "✅" if r['match'] else "❌"
        print(f"#{r['king']:<9} {r['original_score']:<12.2f} {r['original_sharpe']:<12.2f} {r['verified_score']:<15.2f} {r['verified_sharpe']:<15.2f} {match_str}")

def main():
    parser = argparse.ArgumentParser(description='验证 AlphaGPT 挖掘出的因子')
    parser.add_argument('--king', type=int, help='只验证指定的 King 号 (1-based)')
    parser.add_argument('--formula', type=str, help='验证自定义公式 (如 "CLOSE NEG")')
    parser.add_argument('--all', action='store_true', help='验证所有 Kings')
    
    args = parser.parse_args()
    
    if args.formula:
        verify_formula(args.formula, label=f"Custom: {args.formula[:30]}")
    elif args.king:
        json_path = 'model_core/best_cb_formula.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        history = data.get('history', [])
        if args.king <= len(history):
            king = history[args.king - 1]
            # 兼容新格式 (formula 字段) 和旧格式 (readable 字段)
            formula = king.get('formula') or king.get('readable', '')
            if isinstance(formula, list):
                readable = ' '.join(formula)
            else:
                readable = formula
            verify_formula(readable, label=f"King #{args.king}")
        else:
            print(f"❌ King #{args.king} not found (only {len(history)} kings)")
    else:
        verify_all_kings()

if __name__ == "__main__":
    main()
