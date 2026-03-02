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
from collections import Counter, OrderedDict, deque

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
# 止盈所需价格数据
_global_open = None
_global_high = None
_global_prev_close = None

def _init_worker(feat_tensor, target_ret, valid_mask, split_idx, 
                 open_prices=None, high_prices=None, prev_close=None, config_path=None):
    """子进程初始化函数"""
    # 关键修复: 子进程需要重新加载动态配置，否则 INPUT_FEATURES 为空导致校验失败
    if config_path:
        try:
            from .config_loader import load_config
            load_config(config_path)
        except Exception as e:
            print(f"[Worker] Failed to load config from {config_path}: {e}")

    global _global_vm, _global_bt, _global_feat, _global_ret, _global_mask, _global_split_idx
    global _global_open, _global_high, _global_prev_close
    
    # 关键优化: 强制单线程运行，防止多进程 CPU 竞争 (Oversubscription)
    torch.set_num_threads(1)
    
    _global_vm = StackVM()
    _global_bt = CBBacktest(top_k=RobustConfig.TOP_K, take_profit=RobustConfig.TAKE_PROFIT)
    
    # 将 Tensor 移动到 CPU 以避免多进程 CUDA/XPU 冲突
    _global_feat = feat_tensor.to('cpu')
    _global_ret = target_ret.to('cpu')
    _global_mask = valid_mask.to('cpu')
    _global_split_idx = split_idx
    
    # 止盈价格数据
    _global_open = open_prices.to('cpu') if open_prices is not None else None
    _global_high = high_prices.to('cpu') if high_prices is not None else None
    _global_prev_close = prev_close.to('cpu') if prev_close is not None else None

def _worker_eval(formula):
    """
    子进程执行函数 (V2: 稳健性评估)
    
    使用 evaluate_robust 获取多维指标，并计算综合奖励。
    """
    global _global_vm, _global_bt, _global_feat, _global_ret, _global_mask, _global_split_idx
    global _global_open, _global_high, _global_prev_close
    
    try:
        # 0. 公式结构验证 (在昂贵的回测之前进行)
        from .formula_validator import validate_formula
        is_valid, structural_penalty, reason = validate_formula(formula)
        
        if not is_valid:
            return RobustConfig.PENALTY_STRUCT, None, "STRUCT_INVALID", reason
        
        # 1. 执行公式
        try:
            res = _global_vm.execute(formula, _global_feat)
            if res is None:
                return RobustConfig.PENALTY_EXEC, None, "EXEC_NONE", "Returned None"
        except Exception as e:
            return RobustConfig.PENALTY_EXEC, None, "EXEC_ERR", type(e).__name__
        
        # 2. 检查因子方差
        var_threshold = 1e-4
        if res.std() < var_threshold:
            return RobustConfig.PENALTY_LOWVAR, None, "LOW_VARIANCE", f"std={res.std():.2e}, thr={var_threshold}"


        # 3. 稳健性评估 (含止盈逻辑)
        metrics = _global_bt.evaluate_robust(
            factors=res,
            target_ret=_global_ret,
            valid_mask=_global_mask,
            split_idx=_global_split_idx,
            open_prices=_global_open,
            high_prices=_global_high,
            prev_close=_global_prev_close
        )
        
        # 4. 硬淘汰条件 (Hard Filters) - [V3.5] 增加线性梯度与 Clamp
        gaps = []
        fail_status = "PASS"
        fail_reason = "OK"
        
        # 4.1 验证集 Sharpe 太低
        if metrics['sharpe_val'] < RobustConfig.MIN_SHARPE_VAL:
            # 归一化 Gap: (阈值 - 实际值) / max(1, abs(阈值))
            sharpe_gap = (RobustConfig.MIN_SHARPE_VAL - metrics['sharpe_val']) / max(1.0, abs(RobustConfig.MIN_SHARPE_VAL))
            gaps.append(max(0.0, float(sharpe_gap)))
            if fail_status == "PASS":
                fail_status, fail_reason = "METRIC_SHARPE", f"val={metrics['sharpe_val']:.2f}"
        
        # 4.2 活跃率太低 (选不到足够标的)
        if metrics['active_ratio'] < RobustConfig.MIN_ACTIVE_RATIO:
            active_gap = (RobustConfig.MIN_ACTIVE_RATIO - metrics['active_ratio']) / RobustConfig.MIN_ACTIVE_RATIO
            gaps.append(max(0.0, float(active_gap)))
            if fail_status == "PASS":
                fail_status, fail_reason = "METRIC_ACTIVE", f"ratio={metrics['active_ratio']:.2f}"
        
        # 4.4 有效交易日太少 (统计不可靠)
        min_days = RobustConfig.MIN_VALID_DAYS
        if metrics['valid_days_train'] < min_days or metrics['valid_days_val'] < min_days:
            worst_days = min(metrics['valid_days_train'], metrics['valid_days_val'])
            days_gap = (min_days - worst_days) / min_days
            gaps.append(max(0.0, float(days_gap)))
            if fail_status == "PASS":
                fail_status, fail_reason = "METRIC_DAYS", f"tr={metrics['valid_days_train']},val={metrics['valid_days_val']}"

        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            # Clamped Metric Penalty: clamp(base - scale * gap, min=-6.0, max=-4.0)
            p_max = RobustConfig.PENALTY_METRIC_MAX
            p_min = RobustConfig.PENALTY_METRIC_MIN
            penalty = p_max - (p_max - p_min) * avg_gap
            penalty = max(float(p_min), min(float(p_max), penalty))
            return float(penalty), None, fail_status, f"{fail_reason} | gap={avg_gap:.2f}"

        
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
        
        # 5.7 [New] 翻转软惩罚 (Dynamic Soft Penalty)
        flip_penalty = 0.0
        is_flip = False
        detail_msg = "OK"
        if metrics['sharpe_train'] * metrics['sharpe_val'] < 0:
            is_flip = True
            # 动态惩罚: 罚分与 Val 的亏损程度成正比: Penalty = -1 * COEF * abs(Val)
            flip_penalty = -1.0 * RobustConfig.PENALTY_FLIP_COEF * abs(metrics['sharpe_val'])
            detail_msg = f"Soft Flip ({flip_penalty:.2f}, val={metrics['sharpe_val']:.2f})"

        # 5.8 最终分数
        final_score = (base_score + stability_bonus) * RobustConfig.SCALE + ret_bonus - mdd_penalty - len_penalty + structural_penalty + flip_penalty
        
        # 返回分数和详细信息
        if is_flip:
            # [Candidate Isolation] 有分数(RL可学习)，但info=None(不作为King候选)
            return final_score, None, "METRIC_FLIP", detail_msg
        else:
            return final_score, (final_score, metrics['annualized_ret'], metrics['sharpe_all'], formula, metrics), "PASS", "OK"
    
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
        return RobustConfig.PENALTY_EXEC, None, "EXEC_ERR", "RuntimeError"



class AlphaEngine:
    def __init__(self, data_start_date=None):
        print("Initializing AlphaEngine...")
        # 打印配置来源
        config_source = getattr(RobustConfig, '_config_path', 'default_config.yaml')
        print(f"Config source: {config_source}")
        print(f"Using Device: {ModelConfig.DEVICE}")
        print(f"Take Profit: {RobustConfig.TAKE_PROFIT}")
        
        # 1. 初始化并加载数据
        # 确保 engine 拥有唯一的数据加载器实例
        self.loader = CBDataLoader()
        effective_data_start_date = data_start_date or "2022-08-01"
        print(f"Data start date: {effective_data_start_date}")
        self.loader.load_data(start_date=effective_data_start_date)
        
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
        
        # 6. Session 级回测缓存 (V3.5: formula_tuple -> result_tuple)
        self.eval_cache = OrderedDict()
        
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
        print("Starting CB Alpha Mining (Multi-Process CPU)...")
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print(f"   - Steps:       {ModelConfig.TRAIN_STEPS}")
        print(f"   - Batch Size:  {ModelConfig.BATCH_SIZE}")
        print(f"   - Device:      {ModelConfig.DEVICE}")
        print(f"   - Workers:     {os.cpu_count() or 4}")
        print(f"   - Split Date:  {RobustConfig.TRAIN_TEST_SPLIT_DATE}")
        print(f"   - Top-K:       {RobustConfig.TOP_K}")
        print(f"   - Fee Rate:    {RobustConfig.FEE_RATE:.4f} ({RobustConfig.FEE_RATE*100:.2f}% single-side)")
        print(f"   - Entropy beta:{RobustConfig.ENTROPY_BETA_START} -> {RobustConfig.ENTROPY_BETA_END} (linear decay)")
        print(f"   - Factors ({len(ModelConfig.INPUT_FEATURES)}): {ModelConfig.INPUT_FEATURES}")
        print(f"   - Operators ({len(OpsRegistry.list_ops())}): {OpsRegistry.list_ops()}")
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
        
        # 止盈价格数据准备
        # 时序对齐说明:
        #   - weights[t] = t日收盘时的持仓决策
        #   - target_ret[t] = close[t+1]/close[t] - 1 = 持有 t→t+1 的收益
        #   - 止盈检查发生在持仓期间(t+1日盘中)
        #   - 因此: open_prices[t] 应为 open[t+1], high_prices[t] 应为 high[t+1]
        #   - prev_close[t] = close[t] = 买入价格
        cpu_open = None
        cpu_high = None
        cpu_prev_close = None
        if RobustConfig.TAKE_PROFIT > 0:
            # 只在启用止盈时加载价格数据
            if 'OPEN' in self.loader.raw_data_cache and 'HIGH' in self.loader.raw_data_cache:
                raw_open = self.loader.raw_data_cache['OPEN'].to('cpu')
                raw_high = self.loader.raw_data_cache['HIGH'].to('cpu')
                close = self.loader.raw_data_cache['CLOSE'].to('cpu')
                
                # 时序对齐: roll(-1) 使得 [t] 位置存储的是 t+1 日的价格
                cpu_open = torch.roll(raw_open, -1, dims=0)
                cpu_high = torch.roll(raw_high, -1, dims=0)
                # 最后一行无有效数据，置为极大值使其不触发止盈
                cpu_open[-1] = 1e9
                cpu_high[-1] = 1e9
                
                # prev_close[t] = close[t] = 买入价（t日收盘价）
                cpu_prev_close = close.clone()
                
                print(f"   - Take Profit: {RobustConfig.TAKE_PROFIT:.1%} enabled, price data loaded (time-aligned)")
            else:
                print("   - Warning: TAKE_PROFIT enabled but OPEN/HIGH data not available")
        
        # 获取当前 Config 路径 (传递给子进程)
        from .config_loader import get_loaded_config_path
        config_path = get_loaded_config_path()
        
        # 启动进程池
        # 注意: Windows 下每次都需要在这里从头启动 executor 比较安全，或者长期持有
        # 这里我们选择长期持有 executor 上下文
        with ProcessPoolExecutor(
            max_workers=num_workers, 
            initializer=_init_worker,
            initargs=(cpu_feat, cpu_ret, cpu_mask, split_idx, cpu_open, cpu_high, cpu_prev_close, config_path)
        ) as executor:
            
            pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
            global_stats = Counter()
            global_struct_reasons = Counter()
            stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_stats.jsonl')
            
            # [Grammar] 1. 准备元数据 (移至循环外以优化性能)
            is_feat, is_unary, is_binary, net_change = self.model.get_grammar_masks(ModelConfig.DEVICE)
            
            # [V4.1] 状态监控变量 (工程加固版)
            # 1. 成功率窗口 (用于平滑控制)
            # [V4.1.1] 初始化下调为 0.5，防止前期"历史太好"导致一级熵触发反应迟钝
            window_size = 10
            hard_pass_rate_history = deque([0.5]*window_size, maxlen=window_size)
            hard_pass_abs_history = deque([int(ModelConfig.BATCH_SIZE * 0.5)]*window_size, maxlen=window_size)
            struct_rate_history = deque([0.0]*window_size, maxlen=window_size)
            
            # 2. 持续故障触发器 (熔断逻辑)
            struct_failure_strike = 0  # 连续高 Struct 计数
            lowvar_failure_strike = 0  # 连续高 LowVar 计数
            lowvar_recovery_strike = 0
            saturation_strike = 0
            lowvar_penalty_multiplier = 1.0
            low_reward_std_strike = 0
            steps_since_new_king = 0
            steps_since_pool_update = 0
            
            # 3. 控制器状态
            cool_down_timer = 0
            total_steps = ModelConfig.TRAIN_STEPS
            
            for step in pbar:
                step_stats = Counter()
                # [V4.1] 定义三级成功计数
                counts = {"HardPass": 0, "MetricPass": 0, "SimPass": 0}
                bs = ModelConfig.BATCH_SIZE
                
                # [V4.1] 计算当前平滑指标
                rolling_hpr = sum(hard_pass_rate_history) / window_size
                rolling_hpa = sum(hard_pass_abs_history) / window_size
                rolling_str = sum(struct_rate_history) / window_size
                
                # [V4.1] 自适应熵控制回路 (Rolling 版)
                base_beta = RobustConfig.ENTROPY_BETA_START - (
                    RobustConfig.ENTROPY_BETA_START - RobustConfig.ENTROPY_BETA_END
                ) * (step / total_steps)
                
                if cool_down_timer > 0:
                    cool_down_timer -= 1
                    current_beta = getattr(self, '_current_beta_locked', base_beta)
                else:
                    # [V4.1.1] 强化一级触发 (PR < 1%) -> 救火档锁定 0.06
                    if rolling_hpr < 0.01 or rolling_hpa < 1:
                        current_beta = 0.06
                        self._current_beta_locked = current_beta
                        cool_down_timer = 10
                    elif rolling_hpr < 0.05:
                        current_beta = base_beta + 0.005
                    elif rolling_hpr > 0.02 and rolling_str < 0.9:
                        current_beta = base_beta
                    else:
                        current_beta = base_beta

                entropy_boost = 0.0
                if steps_since_new_king >= RobustConfig.STAGNATION_PATIENCE:
                    entropy_boost = max(entropy_boost, RobustConfig.STAGNATION_ENTROPY_BOOST)
                if (
                    steps_since_pool_update >= RobustConfig.REWARD_STD_PATIENCE
                    and low_reward_std_strike >= RobustConfig.REWARD_STD_PATIENCE
                ):
                    entropy_boost = max(entropy_boost, RobustConfig.COLLAPSE_ENTROPY_BOOST)
                if entropy_boost > 0:
                    current_beta = min(current_beta + entropy_boost, 0.08)

                inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

                current_depths = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)

                
                log_probs = []
                entropies = []
                tokens_list = []

                
                # 自回归生成公式 (在 Main Process GPU 上进行)
                for gen_step in range(ModelConfig.MAX_FORMULA_LEN):
                    logits, _ = self.model(inp)

                    # [Grammar] 2. Action Masking
                    mask = torch.ones(bs, self.model.vocab_size, dtype=torch.bool, device=ModelConfig.DEVICE)
                    
                    # Rule 1: D < 1 -> Ban All Ops (Underflow)
                    mask &= ~((current_depths < 1).unsqueeze(1) & (is_unary | is_binary))
                    
                    # Rule 2: D < 2 -> Ban Binary (Underflow)
                    mask &= ~((current_depths < 2).unsqueeze(1) & is_binary)
                    
                    # Rule 3: R <= D - 1 -> Ban Feature (Overrun, 无法归约)
                    # 剩余步数 R = (Total - 1) - current_step
                    # e.g. Total=12. Step=0, R=11. Step=11, R=0.
                    R = ModelConfig.MAX_FORMULA_LEN - 1 - gen_step
                    mask &= ~((R <= current_depths - 1).unsqueeze(1) & is_feat)
                    
                    # Rule 4: R < D - 1 -> Ban Unary (Force Binary Reduce)
                    mask &= ~((R < current_depths - 1).unsqueeze(1) & is_unary)
                    
                    # Rule 5: D >= MAX -> Ban Feature (Stack Limit)
                    mask &= ~((current_depths >= RobustConfig.MAX_STACK_DEPTH).unsqueeze(1) & is_feat)
                    
                    # Apply Mask
                    logits = logits.masked_fill(~mask, -1e9)

                    dist = Categorical(logits=logits)

                    action = dist.sample()
                    
                    # [Grammar] 3. 更新深度
                    current_depths += net_change[action]
                    
                    log_probs.append(dist.log_prob(action))
                    entropies.append(dist.entropy())
                    tokens_list.append(action)
                    inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                
                seqs = torch.stack(tokens_list, dim=1)
                
                # 准备任务: 立即将 token ID 转为字符串，消除解码不一致风险
                formula_list = [self._tokens_to_strings(seq.tolist()) for seq in seqs]
                
                # [V3.5] 批次去重与 LRU 缓存 (核心提速逻辑)
                unique_formula_to_indices = {}
                for idx, f in enumerate(formula_list):
                    # Canonical Key: 对 RPN 进行简单规范化 (目前仅作为 tuple)
                    f_tuple = tuple(f)
                    if f_tuple not in unique_formula_to_indices:
                        unique_formula_to_indices[f_tuple] = []
                    unique_formula_to_indices[f_tuple].append(idx)
                
                num_unique_gen = len(unique_formula_to_indices)
                
                # 识别不在缓存中的唯一公式
                to_eval_formulas = []
                f_idx_to_eval = {}
                for f_tuple in unique_formula_to_indices:
                    if f_tuple not in self.eval_cache:
                        f_idx_to_eval[f_tuple] = len(to_eval_formulas)
                        to_eval_formulas.append(list(f_tuple))
                
                num_to_eval = len(to_eval_formulas)
                
                # 仅回测从未见过且当前批次唯一的公式
                if to_eval_formulas:
                    new_results = list(executor.map(_worker_eval, to_eval_formulas))
                    for f_tuple, eval_idx in f_idx_to_eval.items():
                        # LRU Update
                        if f_tuple in self.eval_cache:
                            self.eval_cache.move_to_end(f_tuple)
                        self.eval_cache[f_tuple] = new_results[eval_idx]
                    
                    # [V3.5] Cache 容量管理 (LRU 淘汰)
                    max_cache = RobustConfig.CACHE_MAX_SIZE
                    if len(self.eval_cache) > max_cache:
                        # 淘汰最早的 20%
                        for _ in range(int(max_cache * 0.2)):
                            self.eval_cache.popitem(last=False)
                
                # 组装 512 个结果
                results = []
                for f in formula_list:
                    res_tuple = self.eval_cache[tuple(f)]
                    results.append(res_tuple)
                    # 每次命中都移到末尾 (LRU)
                    self.eval_cache.move_to_end(tuple(f))
                
                # [V3.5] 提速指标计算
                batch_hit_rate = (bs - num_unique_gen) / bs
                cache_hit_rate = (num_unique_gen - num_to_eval) / bs
                uniq_rate_gen = num_unique_gen / bs
                
                rewards_list = []
                step_gaps = []
                step_struct_reasons = Counter()
                step_new_king = 0
                step_pool_updates = 0

                
                # 聚合结果
                for i, (rew, best_info, status, detail) in enumerate(results):
                    final_status = status
                    
                    # [V4.1] 三级成功定义
                    is_hard_pass = status not in ["STRUCT_INVALID", "EXEC_ERR", "EXEC_NONE", "LOW_VARIANCE"]
                    if is_hard_pass:
                        counts["HardPass"] += 1
                        if status == "PASS":
                            counts["MetricPass"] += 1
                    
                    # [V4.1] 动态惩罚倍率应用 (针对顽固 LowVar)
                    if status == "LOW_VARIANCE":
                        rew = rew * lowvar_penalty_multiplier
                    
                    rewards_list.append(rew)
                    step_stats[status] += 1
                    
                    if status == "STRUCT_INVALID":
                        global_struct_reasons[detail] += 1
                        step_struct_reasons[detail] += 1
                    
                    # 提取 Gap 指标
                    if "gap=" in detail:
                        try:
                            gap_val = float(detail.split("gap=")[-1])
                            step_gaps.append(gap_val)
                        except: pass
                    
                    score_val = None
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
                            step_new_king += 1
                            
                            # 记录到历史 (V2.2: 包含稳健性指标 + 年化收益 + IC/IR)
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
                                # IC/IR 指标
                                'ic_mean': metrics.get('ic_mean', 0),
                                'ic_std': metrics.get('ic_std', 0),
                                'ic_ir': metrics.get('ic_ir'),
                                'ic_ir_annual': metrics.get('ic_ir_annual'),
                                'valid_ic_days': metrics.get('valid_ic_days', 0),
                                'skipped_ic_days': metrics.get('skipped_ic_days', 0),
                                'formula': formula_str,
                                'readable': self.best_formula_readable
                            })
                            
                            # 保存交易细节到独立文件
                            self._save_king_trades(king_num, formula_str, score_val, sharpe_val, ret_val)
                            
                            # IC/IR 安全格式化
                            ic_val = metrics.get('ic_mean', 0)
                            ir_val = metrics.get('ic_ir')
                            ir_str = f"{ir_val:.2f}" if ir_val is not None else "None"
                            
                            tqdm.write(f"[!] New King #{king_num}: Score {score_val:.2f} | Sharpe T/V {metrics.get('sharpe_train', 0):.2f}/{metrics.get('sharpe_val', 0):.2f} | IC {ic_val:.3f} (IR {ir_str}) | MDD {metrics.get('max_drawdown', 0):.1%} | {self.best_formula_readable}")

                        # V2.3: 收集多样性公式 (Diversity Pool)
                        # 仅当分数足够高且公式独特时入池
                        if status == "PASS" and score_val > 0:
                            readable = self.decode_formula(formula_str)
                            new_result = {
                                'step': step,  # 记录产生步数
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
                                        step_pool_updates += 1
                                    else:
                                        # 池满: 替换最低分公式 (如果新公式更好)
                                        min_key = min(self.diverse_pool, key=lambda k: self.diverse_pool[k]['score'])
                                        if score_val > self.diverse_pool[min_key]['score']:
                                            del self.diverse_pool[min_key]
                                            self.diverse_pool[readable] = new_result
                                            step_pool_updates += 1
                            else:
                                # Case 2: 与池中某公式高度相似 -> 仅当分数显著更高 (+10%) 时替换
                                similar_score = self.diverse_pool[similar_key]['score']
                                if score_val > similar_score * 1.1:  # 需要高出 10%
                                    del self.diverse_pool[similar_key]
                                    self.diverse_pool[readable] = new_result
                                    step_pool_updates += 1
                                    final_status = "SIM_REPLACE"
                                else:
                                    final_status = "SIM_REJECT"
                                    # [V3.5] 应用相似度拒绝惩罚，避免躲进冗余区
                                    rewards_list[i] = RobustConfig.PENALTY_SIM
                    
                    # [V4.1.2] SimPass 定义修复: 通用的 MetricPass 且未被 SIM_REJECT (包含 SIM_REPLACE)
                    if final_status != "SIM_REJECT" and is_hard_pass and status == "PASS":
                        counts["SimPass"] += 1

                    
                    step_stats[final_status] += 1
                    global_stats[final_status] += 1

                if step_new_king > 0:
                    steps_since_new_king = 0
                else:
                    steps_since_new_king += 1
                if step_pool_updates > 0:
                    steps_since_pool_update = 0
                else:
                    steps_since_pool_update += 1

                # [V4.1.1] 奖励饱和监控 (Standard Deviation)
                rewards_tensor = torch.tensor(rewards_list, dtype=torch.float)
                reward_std = rewards_tensor.std().item()
                if reward_std < RobustConfig.REWARD_STD_FLOOR:
                    low_reward_std_strike += 1
                else:
                    low_reward_std_strike = 0
                # 修正: 监控 MetricFail 的 Std 而不是 MetricPass (PASS)
                metric_fail_rewards = [r for i, r in enumerate(rewards_list) if "METRIC" in results[i][2]]
                metric_fail_std = torch.tensor(metric_fail_rewards).std().item() if len(metric_fail_rewards) > 1 else 0.0

                log_entry = {
                    "step": step,
                    "stats": dict(step_stats),
                    "reward_std": reward_std,
                    "metric_fail_std": metric_fail_std, # 新增 MetricFail Std
                    "entropy_boost": entropy_boost,
                    "steps_since_new_king": steps_since_new_king,
                    "steps_since_pool_update": steps_since_pool_update,
                    "low_reward_std_strike": low_reward_std_strike,
                    "timestamp": time.time()
                }
                with open(stats_path, 'a', encoding='utf-8') as stats_file:
                    stats_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                if step % 20 == 0:
                    total = bs
                    if step_gaps:
                        avg_gap = np.mean(step_gaps)
                        p50_gap = np.percentile(step_gaps, 50)
                        p90_gap = np.percentile(step_gaps, 90)
                    else:
                        avg_gap = p50_gap = p90_gap = 0.0
                    
                    # [V4.1] 动力学观测 2.0++: 包含 FailAbs 与分级 Pass
                    struct_abs = step_stats['STRUCT_INVALID']
                    lowvar_abs = step_stats['LOW_VARIANCE']
                    metric_abs = sum(step_stats[s] for s in step_stats if 'METRIC' in s)
                    sim_abs = step_stats['SIM_REJECT']
                    
                    # [V4.1.1] 计算 SimFailShare 与 TopFail
                    # [V4.1.2] 修正: 分母为 MetricPass (在这些 Good Ones 里有多少是重复的)
                    sim_fs_denom = counts['MetricPass']
                    sim_fail_share = sim_abs / sim_fs_denom if sim_fs_denom > 0 else 0.0
                    top_fails = step_struct_reasons.most_common(3)
                    top_fail_str = " | ".join([f"{k}:{v}" for k, v in top_fails])
                    
                    msg = (
                        f"[Step {step}] "
                        f"H/M/S_Pass: {counts['HardPass']}/{counts['MetricPass']}/{counts['SimPass']} | "
                        f"RollPR {rolling_hpr:.1%} | "
                        f"FailAbs[S:{struct_abs}, L:{lowvar_abs}, M:{metric_abs}, R:{sim_abs}] | "
                        f"Gap(avg/p50/p90) {avg_gap:.2f}/{p50_gap:.2f}/{p90_gap:.2f} | "
                        f"RStd {reward_std:.2f} | MStd {metric_fail_std:.2f} | "
                        f"SimFS {sim_fail_share:.1%} | B:{current_beta:.4f} "
                        f"(+{entropy_boost:.3f}) | KWait:{steps_since_new_king}"
                    )
                    tqdm.write(msg)
                    if top_fail_str:
                        tqdm.write(f"   TopFail: {top_fail_str}")
                
                # [V4.1] 更新滚动窗口与持续故障触发器
                hpr = counts['HardPass'] / bs
                hard_pass_rate_history.append(hpr)
                hard_pass_abs_history.append(counts['HardPass'])
                struct_rate = step_stats['STRUCT_INVALID'] / bs
                struct_rate_history.append(struct_rate)
                
                # 熔断规则 1: 结构坍缩 (30步 > 90%)
                if struct_rate > 0.9:
                    struct_failure_strike += 1
                else:
                    struct_failure_strike = 0
                
                if struct_failure_strike >= 30:
                    tqdm.write(">>> [CRITICAL] Struct Collapse detected! Entering Reinforcement Mode.")
                    current_beta = 0.06
                    self._current_beta_locked = current_beta
                    cool_down_timer = 20
                    struct_failure_strike = 0
                
                # 熔断规则 2: 低方差坍缩 (50步 > 70%)
                if (step_stats['LOW_VARIANCE'] / bs) > 0.7:
                    lowvar_failure_strike += 1
                else:
                    lowvar_failure_strike = max(0, lowvar_failure_strike - 1)
                
                if lowvar_failure_strike >= 50:
                    tqdm.write(">>> [WARNING] Persistent LowVar detected. Hardening penalty.")
                    lowvar_penalty_multiplier = 1.15
                    lowvar_failure_strike = 0
                
                # [V4.1.1] 熔断规则 3: 低方差恢复
                if (step_stats['LOW_VARIANCE'] / bs) < 0.1:
                    lowvar_recovery_strike += 1
                else:
                    lowvar_recovery_strike = 0
                if lowvar_recovery_strike >= 20 and lowvar_penalty_multiplier > 1.0:
                    tqdm.write(">>> [INFO] LowVar recovered. Normalizing penalty.")
                    lowvar_penalty_multiplier = 1.0
                    lowvar_recovery_strike = 0

                # [V4.1.1] 熔断规则 4: 奖励饱和告警
                if reward_std < RobustConfig.REWARD_STD_FLOOR:
                    saturation_strike += 1
                else:
                    saturation_strike = 0
                if saturation_strike >= RobustConfig.REWARD_STD_PATIENCE:
                    tqdm.write(
                        f">>> [LOG] Reward Saturation Detected "
                        f"(RStd {reward_std:.3f} < {RobustConfig.REWARD_STD_FLOOR:.3f})."
                    )
                    saturation_strike = 0

                rewards = torch.tensor(rewards_list, device=ModelConfig.DEVICE)

                
                # 优势函数归一化
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                if reward_std < RobustConfig.REWARD_STD_FLOOR and RobustConfig.ADV_NOISE_STD > 0:
                    adv = adv + torch.randn_like(adv) * RobustConfig.ADV_NOISE_STD
                
                # Loss & Update
                loss = 0
                for t in range(len(log_probs)):
                    loss += -log_probs[t] * adv
                
                loss = loss.mean()
                
                # V3.5: 使用带精炼控制回路的熵正则化
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
            print(f"\nTraining completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
            
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
        
        print(f"\nSaved to: {result_path}")
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
    parser.add_argument(
        '--data-start-date',
        type=str,
        default=None,
        help='data start date (YYYY-MM-DD), default=2022-08-01'
    )
    args = parser.parse_args()
    
    # 加载配置 (必须在创建 AlphaEngine 之前)
    from .config_loader import load_config
    config = load_config(args.config)
    
    # 记录配置文件路径以便在 init 中打印
    # 记录配置路径 (仅用于打印/调试)
    RobustConfig._config_path = args.config if args.config else "default_config.yaml"  # type: ignore[attr-defined]
    
    if args.config:
        print(f"Loaded custom config: {args.config}")
    else:
        print("Using default config: default_config.yaml")
    
    # Windows 为了支持 ProcessPool，必须要有这个 protect
    eng = AlphaEngine(data_start_date=args.data_start_date)
    eng.train()
