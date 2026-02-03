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
import argparse
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
    
    # 关键优化: 强制单线程运行，防止多进程 CPU 竞争 (Oversubscription)
    torch.set_num_threads(1)
    
    _global_vm = StackVM()
    _global_bt = CBBacktest(top_k=RobustConfig.TOP_K)
    
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
        # 0. 公式结构验证 (在昂贵的回测之前进行)
        from .formula_validator import validate_formula
        is_valid, structural_penalty, reason = validate_formula(formula)
        
        if not is_valid:
            # 硬过滤: 直接拒绝
            return -5.0, None
        
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
        
        # 5.3 年化收益率奖励 (鼓励高回报策略)
        ret_bonus = metrics['annualized_ret'] * RobustConfig.RET_W
        
        # 5.4 回撤惩罚
        mdd_penalty = metrics['max_drawdown'] * RobustConfig.MDD_W
        
        # 5.5 长度惩罚
        len_penalty = len(formula) * RobustConfig.LEN_W
        
        # 5.6 公式结构惩罚 (来自 validate_formula)
        # structural_penalty 是负数或 0，直接加到分数上
        
        # 5.7 最终分数
        final_score = (base_score + stability_bonus) * RobustConfig.SCALE + ret_bonus - mdd_penalty - len_penalty + structural_penalty
        
        # 返回分数和详细信息
        return final_score, (final_score, metrics['annualized_ret'], metrics['sharpe_all'], formula, metrics)
    
    except (ImportError, NameError, AttributeError, SyntaxError) as e:
        # 系统级错误：直接抛出，中断训练，方便 Debug (如刚才的 ImportError)
        import traceback
        traceback.print_exc()
        raise e  # 让主进程感知到 Worker 挂了

    except Exception:
        # 运行时错误 (如除零、NaN、矩阵尺寸不匹配)：视为公式无效，给最低分
        # 如果需要调试，可以打开下面的注释
        # import traceback
        # traceback.print_exc()
        return -5.0, None


class AlphaEngine:
    def __init__(self):
        print("Initializing AlphaEngine...")
        # 打印配置来源
        config_source = getattr(RobustConfig, '_config_path', 'default_config.yaml')
        print(f"📄 Config Source: {config_source}")
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
        self.best_sharpe = 0.0
        self.best_return = 0.0
        
        # 4. 记录所有 New King 历史
        self.king_history = []
        
        # 5. 多样性池 (formula_readable -> metrics_dict)
        self.diverse_pool = {}
        
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
    
    def _calculate_similarity(self, formula_a: list, formula_b: list) -> float:
        """
        计算两个公式的 Jaccard 相似度 (基于 Token 集合)
        """
        set_a = set(formula_a)
        set_b = set(formula_b)
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def train(self):
        print("🚀 Starting CB Alpha Mining (Multi-Process CPU)...")
        print("=" * 60)
        print("🔧 TRAINING CONFIGURATION")
        print(f"   • Steps:       {ModelConfig.TRAIN_STEPS}")
        print(f"   • Batch Size:  {ModelConfig.BATCH_SIZE}")
        print(f"   • Device:      {ModelConfig.DEVICE}")
        print(f"   • Workers:     {os.cpu_count() or 4}")
        print(f"   • Split Date:  {RobustConfig.TRAIN_TEST_SPLIT_DATE}")
        print(f"   • Top-K:       {RobustConfig.TOP_K}")
        print(f"   • Fee Rate:    {RobustConfig.FEE_RATE:.4f} ({RobustConfig.FEE_RATE*100:.2f}% 单边)")
        print(f"   • Entropy β:   {RobustConfig.ENTROPY_BETA_START} → {RobustConfig.ENTROPY_BETA_END} (linear decay)")
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
                entropies = []
                tokens_list = []
                
                # 自回归生成公式 (在 Main Process GPU 上进行)
                for _ in range(ModelConfig.MAX_FORMULA_LEN):
                    logits, _ = self.model(inp)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    
                    log_probs.append(dist.log_prob(action))
                    entropies.append(dist.entropy())
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
                        # V2.2: best_info 现包含 (score, annualized_ret, sharpe_all, formula, metrics)
                        score_val, ret_val, sharpe_val, formula_str, metrics = best_info
                        
                        # 优化: 只有提升超过阈值才视为 New King，减少 I/O 阻塞
                        if score_val > self.best_score + ModelConfig.MIN_SCORE_IMPROVEMENT:
                            self.best_score = score_val
                            self.best_formula = formula_str  # 现在是字符串列表
                            self.best_formula_readable = self.decode_formula(formula_str)
                            self.best_sharpe = sharpe_val
                            self.best_return = ret_val
                            
                            # 记录到历史 (V2.2: 包含稳健性指标 + 年化收益)
                            king_num = len(self.king_history) + 1
                            self.king_history.append({
                                'step': step,
                                'score': score_val,
                                'sharpe': sharpe_val,
                                'sharpe_train': metrics.get('sharpe_train', 0),
                                'sharpe_val': metrics.get('sharpe_val', 0),
                                'max_drawdown': metrics.get('max_drawdown', 0),
                                'stability': metrics.get('stability_metric', 0),
                                'annualized_ret': ret_val,  # 年化收益率
                                'formula': formula_str,
                                'readable': self.best_formula_readable
                            })
                            
                            # 保存交易细节到独立文件
                            self._save_king_trades(king_num, formula_str, score_val, sharpe_val, ret_val)
                            
                            tqdm.write(f"[!] New King #{king_num}: Score {score_val:.2f} | Sharpe T/V {metrics.get('sharpe_train', 0):.2f}/{metrics.get('sharpe_val', 0):.2f} | MDD {metrics.get('max_drawdown', 0):.1%} | {self.best_formula_readable}")

                        # V2.3: 收集多样性公式 (Diversity Pool)
                        # 仅当分数足够高且公式独特时入池
                        if score_val > 0:
                            readable = self.decode_formula(formula_str)
                            new_result = {
                                'score': score_val,
                                'sharpe': sharpe_val,
                                'annualized_ret': ret_val,
                                'formula': formula_str,
                                'readable': readable
                            }
                            
                            # V2.3: Jaccard 相似度过滤
                            # 检查与池中现有公式的相似度
                            similar_key = None
                            for pool_key, pool_data in self.diverse_pool.items():
                                similarity = self._calculate_similarity(formula_str, pool_data['formula'])
                                if similarity > RobustConfig.JACCARD_THRESHOLD:  # 相似度阈值
                                    similar_key = pool_key
                                    break
                            
                            # 入池逻辑
                            if similar_key is None:
                                # Case 1: 与池中无相似公式 -> 正常入池
                                if readable not in self.diverse_pool:
                                    if len(self.diverse_pool) < RobustConfig.DIVERSITY_POOL_SIZE:
                                        self.diverse_pool[readable] = new_result
                                    else:
                                        # 池满: 替换最低分公式 (如果新公式更好)
                                        min_key = min(self.diverse_pool, key=lambda k: self.diverse_pool[k]['score'])
                                        if score_val > self.diverse_pool[min_key]['score']:
                                            del self.diverse_pool[min_key]
                                            self.diverse_pool[readable] = new_result
                            else:
                                # Case 2: 与池中某公式高度相似 -> 仅当分数显著更高 (+10%) 时替换
                                similar_score = self.diverse_pool[similar_key]['score']
                                if score_val > similar_score * 1.1:  # 需要高出 10%
                                    del self.diverse_pool[similar_key]
                                    self.diverse_pool[readable] = new_result

                rewards = torch.tensor(rewards_list, device=ModelConfig.DEVICE)
                
                # 优势函数归一化
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                
                # Loss & Update
                loss = 0
                for t in range(len(log_probs)):
                    loss += -log_probs[t] * adv
                
                loss = loss.mean()
                
                # V2.3: 熵正则化 (带线性衰减)
                # current_beta = START - (START - END) * (step / TOTAL_STEPS)
                total_steps = ModelConfig.TRAIN_STEPS
                current_beta = RobustConfig.ENTROPY_BETA_START - (
                    RobustConfig.ENTROPY_BETA_START - RobustConfig.ENTROPY_BETA_END
                ) * (step / total_steps)
                avg_entropy = torch.stack(entropies).mean()
                loss = loss - current_beta * avg_entropy
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                pbar.set_postfix({
                    'AvgRew': f'{rewards.mean().item():.2f}',
                    'Best': f'{self.best_score:.2f}',
                    'Ent': f'{avg_entropy.item():.2f}',
                    'β': f'{current_beta:.3f}'
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
        bt = CBBacktest(top_k=RobustConfig.TOP_K)
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
                'score': self.best_score,
                'sharpe': self.best_sharpe,
                'annualized_ret': self.best_return  # 年化收益率
            },
            'history': self.king_history,
            'total_kings': len(self.king_history),
            'diverse_top_50': sorted(
                list(self.diverse_pool.values()),
                key=lambda x: x['score'],
                reverse=True
            )[:RobustConfig.DIVERSITY_POOL_SIZE]
        }
        
        result_path = os.path.join(output_dir, 'best_cb_formula.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Saved to: {result_path}")
        print(f"   Total New Kings discovered: {len(self.king_history)}")
        print(f"   Best Score: {self.best_score:.2f}")
        print(f"   Best Sharpe: {self.best_sharpe:.2f}")
        print(f"   Best Annualized Return: {self.best_return:.2%}")
        print(f"   Best Formula: {self.best_formula_readable}")
        
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'alphagpt_cb.pt'))

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description='AlphaGPT 训练引擎 - 可转债因子挖掘',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m model_core.engine                          # 使用默认配置
  python -m model_core.engine --config my_config.yaml  # 使用自定义配置
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径 (YAML 格式)，不指定则使用 default_config.yaml'
    )
    args = parser.parse_args()
    
    # 加载配置 (必须在创建 AlphaEngine 之前)
    from .config_loader import load_config
    config = load_config(args.config)
    
    # 记录配置文件路径以便在 init 中打印
    RobustConfig._config_path = args.config if args.config else "default_config.yaml"
    
    if args.config:
        print(f"📁 已加载自定义配置: {args.config}")
    else:
        print("📁 使用默认配置: default_config.yaml")
    
    # Windows 为了支持 ProcessPool，必须要有这个 protect
    eng = AlphaEngine()
    eng.train()