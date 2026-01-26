"""
AlphaGPT 训练引擎 (可转债版)

使用 Policy Gradient 训练 Transformer 模型生成 Alpha 因子公式。
V2: 集成稳健性评估 (分段验证、滚动稳定性、最大回撤、可交易性约束)
"""
import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .config import ModelConfig, RobustConfig
from .data_loader import CBDataLoader
from .alphagpt import AlphaGPT
from .vm import StackVM
from .backtest import CBBacktest
from .ops_registry import OpsRegistry


# 全局变量，用于子进程共享只读数据 (避免 Pickling 开销)
_global_vm = None
_global_bt = None
_global_feat = None
_global_ret = None
_global_mask = None
_global_split_idx = None  # 训练/验证切分索引

def _init_worker(feat_tensor, target_ret, valid_mask, split_idx):
    """子进程初始化函数"""
    global _global_vm, _global_bt, _global_feat, _global_ret, _global_mask, _global_split_idx
    _global_vm = StackVM()
    _global_bt = CBBacktest(top_k=10, fee_rate=0.0001)
    
    # 将 Tensor 移动到 CPU 以避免多进程 CUDA/XPU 冲突
    _global_feat = feat_tensor.to('cpu')
    _global_ret = target_ret.to('cpu')
    _global_mask = valid_mask.to('cpu')
    _global_split_idx = split_idx

def _worker_eval(formula):
    """
    子进程执行函数 (V2: 稳健性评估)
    
    使用 evaluate_robust 获取多维指标，并计算综合奖励。
    """
    global _global_vm, _global_bt, _global_feat, _global_ret, _global_mask, _global_split_idx
    
    try:
        # 1. 执行公式
        res = _global_vm.execute(formula, _global_feat)
        
        if res is None:
            return -5.0, None
        
        # 2. 检查因子方差
        if res.std() < 1e-4:
            return -2.0, None

        # 3. 稳健性评估
        metrics = _global_bt.evaluate_robust(
            factors=res,
            target_ret=_global_ret,
            valid_mask=_global_mask,
            split_idx=_global_split_idx
        )
        
        # 4. 硬淘汰条件 (Hard Filters)
        # 4.1 验证集 Sharpe 太低
        if metrics['sharpe_val'] < RobustConfig.MIN_SHARPE_VAL:
            return -5.0, None
        
        # 4.2 活跃率太低 (选不到足够标的)
        if metrics['active_ratio'] < RobustConfig.MIN_ACTIVE_RATIO:
            return -4.0, None
        
        # 4.3 训练/验证方向翻转 (过拟合信号)
        if metrics['sharpe_train'] * metrics['sharpe_val'] < 0:
            return -3.0, None
        
        # 4.4 有效交易日太少 (统计不可靠)
        if metrics['valid_days_train'] < RobustConfig.MIN_VALID_DAYS or metrics['valid_days_val'] < RobustConfig.MIN_VALID_DAYS:
            return -2.5, None
        
        # 5. 综合评分 (Soft Scoring)
        # 5.1 基础分: 加权 Sharpe
        base_score = (RobustConfig.TRAIN_WEIGHT * metrics['sharpe_train'] + 
                      RobustConfig.VAL_WEIGHT * metrics['sharpe_val'])
        
        # 5.2 稳定性加成 (Mean - K*Std 越高越好)
        stability_bonus = metrics['stability_metric'] * RobustConfig.STABILITY_W
        
        # 5.3 回撤惩罚
        mdd_penalty = metrics['max_drawdown'] * RobustConfig.MDD_W
        
        # 5.4 长度惩罚
        len_penalty = len(formula) * RobustConfig.LEN_W
        
        # 5.5 最终分数
        final_score = (base_score + stability_bonus) * RobustConfig.SCALE - mdd_penalty - len_penalty
        
        # 返回分数和详细信息
        return final_score, (final_score, metrics['cum_ret'], metrics['sharpe_all'], formula, metrics)
    
    except Exception:
        return -5.0, None


class AlphaEngine:
    def __init__(self):
        print("Initializing AlphaEngine...")
        print(f"Using Device: {ModelConfig.DEVICE}")
        
        # 1. 初始化并加载数据
        # 确保 engine 拥有唯一的数据加载器实例
        self.loader = CBDataLoader()
        self.loader.load_data()
        
        # 2. 初始化模型
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # 3. 追踪最佳结果
        self.best_score = -float('inf')
        self.best_formula = None
        self.best_formula_readable = None
        
        # 4. 记录所有 New King 历史
        self.king_history = []
        
        print(f"Model vocab size: {self.model.vocab_size}")

    def _tokens_to_strings(self, tokens: list) -> list:
        """[内部方法] 将 token ID 列表转换为字符串列表 (仅用于训练时的即时转换)"""
        vocab = self.model.vocab
        return [vocab[t] if t < len(vocab) else f'?{t}' for t in tokens]
    
    def decode_formula(self, formula: list) -> str:
        """将公式字符串列表转换为可读字符串"""
        if not formula:
            return ''
        if not isinstance(formula[0], str):
            raise TypeError("decode_formula only accepts string lists, not token IDs")
        return ' '.join(formula)

    def train(self):
        print("🚀 Starting CB Alpha Mining (Multi-Process CPU)...")
        print("=" * 60)
        print("🔧 TRAINING CONFIGURATION")
        print(f"   • Steps:       {ModelConfig.TRAIN_STEPS}")
        print(f"   • Batch Size:  {ModelConfig.BATCH_SIZE}")
        print(f"   • Device:      {ModelConfig.DEVICE}")
        print(f"   • Workers:     {os.cpu_count() or 4}")
        print(f"   • Split Date:  {RobustConfig.TRAIN_TEST_SPLIT_DATE}")
        print("-" * 60)
        print("🧬 GENOME (Vocabulary)")
        print(f"   • Factors ({len(ModelConfig.INPUT_FEATURES)}): {ModelConfig.INPUT_FEATURES}")
        print(f"   • Operators ({len(OpsRegistry.list_ops())}): {OpsRegistry.list_ops()}")
        print("=" * 60)
        
        start_time = time.time()
        
        # workers 数量设置为 CPU 核心数 (逻辑核心)
        num_workers = os.cpu_count() or 4
        print(f"Using {num_workers} worker processes")
        
        # 准备共享数据 (转为 CPU Tensor)
        cpu_feat = self.loader.feat_tensor.to('cpu')
        cpu_ret = self.loader.target_ret.to('cpu')
        cpu_mask = self.loader.valid_mask.to('cpu')
        split_idx = self.loader.split_idx
        
        # 启动进程池
        # 注意: Windows 下每次都需要在这里从头启动 executor 比较安全，或者长期持有
        # 这里我们选择长期持有 executor 上下文
        with ProcessPoolExecutor(
            max_workers=num_workers, 
            initializer=_init_worker,
            initargs=(cpu_feat, cpu_ret, cpu_mask, split_idx)
        ) as executor:
            
            pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
            
            for step in pbar:
                bs = ModelConfig.BATCH_SIZE
                inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
                
                log_probs = []
                tokens_list = []
                
                # 自回归生成公式 (在 Main Process GPU 上进行)
                for _ in range(ModelConfig.MAX_FORMULA_LEN):
                    logits, _ = self.model(inp)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    
                    log_probs.append(dist.log_prob(action))
                    tokens_list.append(action)
                    inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                
                seqs = torch.stack(tokens_list, dim=1)
                
                # 准备任务: 立即将 token ID 转为字符串，消除解码不一致风险
                formula_list = [self._tokens_to_strings(seq.tolist()) for seq in seqs]
                rewards_list = []
                
                # 并行回测 (CPU)
                results = list(executor.map(_worker_eval, formula_list))
                
                # 聚合结果
                for i, (rew, best_info) in enumerate(results):
                    rewards_list.append(rew)
                    
                    if best_info:
                        # V2: best_info 现包含 (score, cum_ret, sharpe_all, formula, metrics)
                        score_val, ret_val, sharpe_val, formula_str, metrics = best_info
                        if score_val > self.best_score:
                            self.best_score = score_val
                            self.best_formula = formula_str  # 现在是字符串列表
                            self.best_formula_readable = self.decode_formula(formula_str)
                            
                            # 记录到历史 (V2: 包含稳健性指标)
                            king_num = len(self.king_history) + 1
                            self.king_history.append({
                                'step': step,
                                'score': score_val,
                                'sharpe': sharpe_val,
                                'sharpe_train': metrics.get('sharpe_train', 0),
                                'sharpe_val': metrics.get('sharpe_val', 0),
                                'max_drawdown': metrics.get('max_drawdown', 0),
                                'stability': metrics.get('stability_metric', 0),
                                'return': ret_val,
                                'formula': formula_str,
                                'readable': self.best_formula_readable
                            })
                            
                            # 保存交易细节到独立文件
                            self._save_king_trades(king_num, formula_str, score_val, sharpe_val, ret_val)
                            
                            tqdm.write(f"[!] New King #{king_num}: Score {score_val:.2f} | Sharpe T/V {metrics.get('sharpe_train', 0):.2f}/{metrics.get('sharpe_val', 0):.2f} | MDD {metrics.get('max_drawdown', 0):.1%} | {self.best_formula_readable}")

                rewards = torch.tensor(rewards_list, device=ModelConfig.DEVICE)
                
                # 优势函数归一化
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                
                # Loss & Update
                loss = 0
                for t in range(len(log_probs)):
                    loss += -log_probs[t] * adv
                
                loss = loss.mean()
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                pbar.set_postfix({
                    'AvgRew': f'{rewards.mean().item():.2f}',
                    'Best': f'{self.best_score:.2f}'
                })

            end_time = time.time()
            duration = end_time - start_time
            print(f"\n✅ Training Completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
            
            self._save_results()
    
    def _save_king_trades(self, king_num: int, formula: list, score: float, sharpe: float, ret: float):
        """保存 New King 的交易细节"""
        from .vm import StackVM
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        trades_dir = os.path.join(output_dir, 'king_trades')
        os.makedirs(trades_dir, exist_ok=True)
        
        # 重新执行公式获取因子值
        vm = StackVM()
        factors = vm.execute(formula, self.loader.feat_tensor)
        
        if factors is None:
            return
        
        # 使用详细回测获取交易记录
        bt = CBBacktest(top_k=10, fee_rate=0.0001)
        details = bt.evaluate_with_details(
            factors=factors,
            target_ret=self.loader.target_ret,
            valid_mask=self.loader.valid_mask
        )
        
        # 构建交易记录
        trades = []
        for t, (indices, daily_ret) in enumerate(zip(details['daily_holdings'], details['daily_returns'])):
            date = self.loader.dates_list[t] if t < len(self.loader.dates_list) else f"Day_{t}"
            
            # 将资产索引转换为名称
            holdings = []
            for idx in indices:
                if idx < len(self.loader.assets_list):
                    code = self.loader.assets_list[idx]
                    name = self.loader.names_dict.get(code, code)
                    holdings.append(name)
            
            trades.append({
                'date': date,
                'holdings': holdings,
                'daily_ret': round(daily_ret, 6)
            })
        
        # 保存到文件
        result = {
            'king_num': king_num,
            'formula': self.decode_formula(formula),
            'score': score,
            'sharpe': sharpe,
            'return': ret,
            'trades': trades
        }
        
        file_path = os.path.join(trades_dir, f'king_{king_num}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def _save_results(self):
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建完整结果，包含最佳因子和进化历史
        result = {
            'best': {
                'formula': self.best_formula,  # 现在是字符串列表
                'readable': self.best_formula_readable,
                'score': self.best_score
            },
            'history': self.king_history,
            'total_kings': len(self.king_history)
        }
        
        result_path = os.path.join(output_dir, 'best_cb_formula.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Saved to: {result_path}")
        print(f"   Total New Kings discovered: {len(self.king_history)}")
        print(f"   Best Score: {self.best_score:.2f}")
        print(f"   Best Formula: {self.best_formula_readable}")
        
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'alphagpt_cb.pt'))

if __name__ == "__main__":
    # Windows 为了支持 ProcessPool，必须要有这个 protect
    eng = AlphaEngine()
    eng.train()