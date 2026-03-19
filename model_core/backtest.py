"""
可转债回测引擎 (Convertible Bond Backtest Engine)

支持 Top-K 轮动策略的向量化回测，用于 AlphaGPT 的 RL 训练。
V2: 增加分段验证、滚动稳定性、最大回撤、可交易性约束。
"""
import torch
from .config import ModelConfig, RobustConfig
from .signal_utils import build_topk_weights, default_min_valid_count


class CBBacktest:
    """
    可转债 Top-K 轮动回测器
    
    核心逻辑:
    1. 每天根据因子值对所有有效转债排序
    2. 选出因子值最高的 Top-K 个标的
    3. 等权持仓 (1/K)
    4. 计算组合收益和夏普比率作为 Reward
    """
    
    def __init__(self, top_k: int = 20, fee_rate: float = None, take_profit: float = None):
        """
        Args:
            top_k: 每天持有的转债数量
            fee_rate: 单边交易费率，默认使用 RobustConfig.FEE_RATE
            take_profit: 止盈涨幅阈值，0 表示不止盈，默认使用 RobustConfig.TAKE_PROFIT
        """
        self.top_k = top_k
        self.fee_rate = fee_rate if fee_rate is not None else RobustConfig.FEE_RATE
        self.take_profit = take_profit if take_profit is not None else RobustConfig.TAKE_PROFIT
        # 要求有效标的至少是 top_k 的 2 倍 (最少 30 个)，保证选择性
        self.min_valid_count = default_min_valid_count(
            top_k=top_k,
            override=RobustConfig.SIGNAL_MIN_VALID_COUNT,
            floor=RobustConfig.MIN_VALID_COUNT,
        )

    @staticmethod
    def _carry_forward_weights(weights: torch.Tensor, valid_trading_day: torch.Tensor) -> torch.Tensor:
        """
        连续持仓口径:
        - 有新信号日: 使用当日新权重
        - 无新信号日: 延续上一日持仓
        - 初始无仓: 权重为 0
        """
        if weights.dim() != 2 or valid_trading_day.dim() != 1:
            raise ValueError("weights must be 2D and valid_trading_day must be 1D")
        if weights.shape[0] != valid_trading_day.shape[0]:
            raise ValueError("weights and valid_trading_day length mismatch")

        continuous = torch.zeros_like(weights)
        for t in range(weights.shape[0]):
            if valid_trading_day[t]:
                continuous[t] = weights[t]
            elif t > 0:
                continuous[t] = continuous[t - 1]
        return continuous

    def _compute_net_returns(
        self,
        weights: torch.Tensor,
        target_ret: torch.Tensor,
        open_prices: torch.Tensor = None,
        high_prices: torch.Tensor = None,
        prev_close: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        给定权重路径，计算全样本净收益与换手。
        """
        device = weights.device
        t_count = weights.shape[0]

        prev_weights = torch.roll(weights, 1, dims=0)
        prev_weights[0] = 0
        turnover = torch.abs(weights - prev_weights).sum(dim=1)
        tx_cost = turnover * self.fee_rate * 2

        effective_ret = target_ret.clone()
        tp_extra_cost = torch.zeros(t_count, device=device)

        if self.take_profit > 0 and open_prices is not None and high_prices is not None and prev_close is not None:
            valid_price_mask = (
                (prev_close > 0) & (prev_close < 10000) &
                (open_prices > 0) & (open_prices < 10000) &
                (high_prices > 0) & (high_prices < 10000)
            )

            tp_trigger_price = prev_close * (1 + self.take_profit)
            tp_holding_mask = prev_weights > 0

            open_gap_up = (open_prices >= tp_trigger_price) & valid_price_mask
            gap_up_mask = open_gap_up & tp_holding_mask

            intraday_tp = (high_prices >= tp_trigger_price) & (~open_gap_up) & valid_price_mask
            intra_tp_mask = intraday_tp & tp_holding_mask

            open_ret = (open_prices / prev_close) - 1.0
            effective_ret[gap_up_mask] = open_ret[gap_up_mask]
            effective_ret[intra_tp_mask] = self.take_profit

            daily_k = tp_holding_mask.sum(dim=1).float()
            gap_up_count = gap_up_mask.sum(dim=1).float()
            intra_tp_count = intra_tp_mask.sum(dim=1).float()
            safe_k = torch.where(daily_k > 0, daily_k, torch.ones_like(daily_k))
            tp_extra_cost += (gap_up_count + intra_tp_count) * 2 * self.fee_rate / safe_k

        gross_ret = (weights * effective_ret).sum(dim=1)
        net_ret = gross_ret - tx_cost - tp_extra_cost
        return net_ret, turnover
    
    def evaluate(self, factors: torch.Tensor, target_ret: torch.Tensor, 
                 valid_mask: torch.Tensor,
                 open_prices: torch.Tensor = None,
                 high_prices: torch.Tensor = None,
                 prev_close: torch.Tensor = None) -> tuple:
        """
        评估因子的回测表现
        
        Args:
            factors: [Time, Assets] 因子值张量
            target_ret: [Time, Assets] 资产收益率张量 (T+1 收益)
            valid_mask: [Time, Assets] 有效标的掩码 (True=可交易)
            open_prices: [Time, Assets] 开盘价，止盈时需要
            high_prices: [Time, Assets] 最高价，止盈时需要
            prev_close: [Time, Assets] 前收盘价，止盈时需要
        
        Returns:
            reward: 标量，用于 RL 训练的奖励信号
            cum_ret: 累计收益率 (用于日志输出)
            sharpe: 夏普比率
        """
        device = factors.device
        T, N = factors.shape
        
        weights, valid_trading_day, daily_valid_count, _ = build_topk_weights(
            factors=factors,
            valid_mask=valid_mask,
            top_k=self.top_k,
            min_valid_count=self.min_valid_count,
            clean_enabled=RobustConfig.SIGNAL_CLEAN_ENABLED,
            winsor_q=RobustConfig.SIGNAL_WINSOR_Q,
            clip_value=RobustConfig.SIGNAL_CLIP,
            rank_output=RobustConfig.SIGNAL_RANK_OUTPUT,
        )
        
        net_ret, turnover = self._compute_net_returns(
            weights=weights,
            target_ret=target_ret,
            open_prices=open_prices,
            high_prices=high_prices,
            prev_close=prev_close,
        )
        
        # 8. 仅对有效交易日计算 Sharpe
        valid_net_ret = net_ret[valid_trading_day]
        
        if len(valid_net_ret) < 10:
            # 有效交易日太少，无法计算有意义的 Sharpe
            return torch.tensor(-10.0, device=device), 0.0, 0.0
        
        # 累计收益 (只计算有效交易日)
        cum_ret = (1 + valid_net_ret).prod() - 1
        
        # 夏普比率 (年化)
        mean_ret = valid_net_ret.mean()
        std_ret = valid_net_ret.std() + 1e-9
        sharpe = mean_ret / std_ret * (252 ** 0.5)
        
        # 9. 换手惩罚 (只对有效交易日)
        valid_turnover = turnover[valid_trading_day]
        avg_turnover = valid_turnover.mean()
        turnover_penalty = torch.clamp(avg_turnover - 0.3, min=0) * 2
        
        # 10. 活跃度检查
        avg_holding = (weights > 0).sum(dim=1).float()[valid_trading_day].mean()  # 平均每天持仓数量
        if avg_holding < self.top_k * 0.5:
            activity_penalty = 5.0
        else:
            activity_penalty = 0.0
        
        # 11. 计算最终奖励
        reward = sharpe * 10 - turnover_penalty - activity_penalty
        
        # 12. 稳健性检查
        if torch.isnan(reward) or torch.isinf(reward):
            reward = torch.tensor(-10.0, device=device)
        
        return reward, cum_ret.item(), sharpe.item()
    
    def evaluate_with_details(self, factors: torch.Tensor, target_ret: torch.Tensor, 
                               valid_mask: torch.Tensor) -> dict:
        """
        评估因子并返回详细交易记录
        
        Returns:
            dict: {
                'reward': float,
                'cum_ret': float,
                'sharpe': float,
                'daily_holdings': List[List[int]],  # 每天持仓的资产索引
                'daily_returns': List[float]        # 每天的组合收益率
            }
        """
        device = factors.device
        T, N = factors.shape
        
        weights, valid_trading_day, daily_valid_count, daily_holdings = build_topk_weights(
            factors=factors,
            valid_mask=valid_mask,
            top_k=self.top_k,
            min_valid_count=self.min_valid_count,
            clean_enabled=RobustConfig.SIGNAL_CLEAN_ENABLED,
            winsor_q=RobustConfig.SIGNAL_WINSOR_Q,
            clip_value=RobustConfig.SIGNAL_CLIP,
            rank_output=RobustConfig.SIGNAL_RANK_OUTPUT,
        )
        
        net_ret, turnover = self._compute_net_returns(weights=weights, target_ret=target_ret)
        
        # 仅对有效交易日计算指标
        valid_net_ret = net_ret[valid_trading_day]
        
        if len(valid_net_ret) < 10:
            return {
                'reward': -10.0,
                'cum_ret': 0.0,
                'sharpe': 0.0,
                'daily_holdings': daily_holdings,
                'daily_returns': net_ret.tolist()
            }
        
        cum_ret = (1 + valid_net_ret).prod() - 1
        mean_ret = valid_net_ret.mean()
        std_ret = valid_net_ret.std() + 1e-9
        sharpe = mean_ret / std_ret * (252 ** 0.5)
        
        valid_turnover = turnover[valid_trading_day]
        avg_turnover = valid_turnover.mean()
        turnover_penalty = torch.clamp(avg_turnover - 0.3, min=0) * 2
        
        avg_holding = (weights > 0).sum(dim=1).float()[valid_trading_day].mean()
        activity_penalty = 5.0 if avg_holding < self.top_k * 0.5 else 0.0
        
        reward = sharpe * 10 - turnover_penalty - activity_penalty
        if torch.isnan(reward) or torch.isinf(reward):
            reward = torch.tensor(-10.0, device=device)
        
        return {
            'reward': reward.item() if hasattr(reward, 'item') else reward,
            'cum_ret': cum_ret.item(),
            'sharpe': sharpe.item(),
            'daily_holdings': daily_holdings,
            'daily_returns': net_ret.tolist()
        }
    
    def evaluate_robust(self, factors: torch.Tensor, target_ret: torch.Tensor,
                        valid_mask: torch.Tensor, split_idx: int,
                        open_prices: torch.Tensor = None,
                        high_prices: torch.Tensor = None,
                        prev_close: torch.Tensor = None) -> dict:
        """
        稳健性评估方法 (Robust Evaluation)
        
        返回分段 Sharpe、滚动稳定性、最大回撤、活跃率等多维指标。
        
        Args:
            factors: [Time, Assets] 因子值张量
            target_ret: [Time, Assets] 资产收益率张量 (T+1 收益)
            valid_mask: [Time, Assets] 有效标的掩码
            split_idx: 训练/验证切分索引
            open_prices: [Time, Assets] 开盘价，止盈时需要
            high_prices: [Time, Assets] 最高价，止盈时需要
            prev_close: [Time, Assets] 前收盘价，止盈时需要
        
        Returns:
            dict: 包含多维评估指标
        """
        device = factors.device
        T, N = factors.shape
        
        # 1. 基础计算 (与 evaluate 相同)
        weights, valid_trading_day, daily_valid_count, _ = build_topk_weights(
            factors=factors,
            valid_mask=valid_mask,
            top_k=self.top_k,
            min_valid_count=self.min_valid_count,
            clean_enabled=RobustConfig.SIGNAL_CLEAN_ENABLED,
            winsor_q=RobustConfig.SIGNAL_WINSOR_Q,
            clip_value=RobustConfig.SIGNAL_CLIP,
            rank_output=RobustConfig.SIGNAL_RANK_OUTPUT,
        )
        
        net_ret, turnover = self._compute_net_returns(
            weights=weights,
            target_ret=target_ret,
            open_prices=open_prices,
            high_prices=high_prices,
            prev_close=prev_close,
        )
        continuous_weights = self._carry_forward_weights(weights, valid_trading_day)
        net_ret_full, _ = self._compute_net_returns(
            weights=continuous_weights,
            target_ret=target_ret,
            open_prices=open_prices,
            high_prices=high_prices,
            prev_close=prev_close,
        )
        
        # 2. 分段计算 Sharpe
        # Train: [0, split_idx), Val: [split_idx, T)
        train_mask = valid_trading_day.clone()
        train_mask[split_idx:] = False
        val_mask = valid_trading_day.clone()
        val_mask[:split_idx] = False
        
        train_ret = net_ret[train_mask]
        val_ret = net_ret[val_mask]
        train_ret_full = net_ret_full[:split_idx]
        val_ret_full = net_ret_full[split_idx:]

        sharpe_train_valid_days = self._calc_sharpe(train_ret)
        sharpe_val_valid_days = self._calc_sharpe(val_ret)
        sharpe_train = self._calc_sharpe(train_ret_full)
        sharpe_val = self._calc_sharpe(val_ret_full)
        
        # 3. 滚动稳定性
        valid_net_ret = net_ret[valid_trading_day]
        stability_metric, sharpe_std = self._calc_rolling_stability(valid_net_ret)
        
        # 4. 最大回撤
        max_drawdown = self._calc_max_drawdown(valid_net_ret)
        
        # 5. 活跃率 (实际持仓数 / top_k 的比例)
        # 修正: 应该统计持仓只数 (weights > 0 的数量)，而不是权重和 (总是1)
        holding_counts = (weights > 0).sum(dim=1).float()  # 每天持仓只数
        valid_holding_counts = holding_counts[valid_trading_day]
        if len(valid_holding_counts) > 0:
            active_ratio = (valid_holding_counts / self.top_k).mean().item()
        else:
            active_ratio = 0.0
        
        # 6. 年化收益率 (Annualized Return)
        if len(valid_net_ret) > 0:
            cum_ret_valid = (1 + valid_net_ret).prod() - 1
            n_days_valid = len(valid_net_ret)
            annualized_ret_valid_days = ((1 + cum_ret_valid) ** (252.0 / n_days_valid) - 1).item() if hasattr(cum_ret_valid, 'item') else ((1 + cum_ret_valid) ** (252.0 / n_days_valid) - 1)
        else:
            annualized_ret_valid_days = 0.0

        if len(net_ret_full) > 0:
            cum_ret_full = (1 + net_ret_full).prod() - 1
            annualized_ret = ((1 + cum_ret_full) ** (252.0 / len(net_ret_full)) - 1).item() if hasattr(cum_ret_full, 'item') else ((1 + cum_ret_full) ** (252.0 / len(net_ret_full)) - 1)
        else:
            annualized_ret = 0.0
        
        # 7. 全局 Sharpe
        sharpe_all_valid_days = self._calc_sharpe(valid_net_ret)
        sharpe_all = self._calc_sharpe(net_ret_full)
        
        # 8. IC/IR 指标计算
        ic_metrics = self._compute_ic_metrics(factors, target_ret, valid_mask)
        
        return {
            'sharpe_train': sharpe_train,
            'sharpe_val': sharpe_val,
            'sharpe_all': sharpe_all,
            'sharpe_train_valid_days': sharpe_train_valid_days,
            'sharpe_val_valid_days': sharpe_val_valid_days,
            'sharpe_all_valid_days': sharpe_all_valid_days,
            'stability_metric': stability_metric,
            'sharpe_std': sharpe_std,
            'max_drawdown': max_drawdown,
            'active_ratio': active_ratio,
            'annualized_ret': annualized_ret,  # 年化收益率
            'annualized_ret_valid_days': annualized_ret_valid_days,
            'valid_days_train': int(train_mask.sum().item()),
            'valid_days_val': int(val_mask.sum().item()),
            'valid_signal_days': int(valid_trading_day.sum().item()),
            'valid_day_ratio': float(valid_trading_day.float().mean().item()) if T > 0 else 0.0,
            # IC/IR 指标
            'ic_mean': ic_metrics['ic_mean'],
            'ic_std': ic_metrics['ic_std'],
            'ic_ir': ic_metrics['ic_ir'],
            'ic_ir_annual': ic_metrics['ic_ir_annual'],
            'valid_ic_days': ic_metrics['valid_ic_days'],
            'total_ic_days': ic_metrics['total_ic_days'],
            'skipped_ic_days': ic_metrics['skipped_ic_days'],
        }
    
    def _calc_sharpe(self, returns: torch.Tensor) -> float:
        """计算年化夏普比率"""
        if len(returns) < 5:
            return 0.0
        mean_ret = returns.mean()
        std_ret = returns.std() + 1e-9
        sharpe = mean_ret / std_ret * (252 ** 0.5)
        return sharpe.item() if hasattr(sharpe, 'item') else sharpe
    
    def _calc_rolling_stability(self, returns: torch.Tensor) -> tuple:
        """
        计算滚动 Sharpe 的稳定性指标
        
        Returns:
            stability_metric: Mean(rolling_sharpe) - K * Std(rolling_sharpe)
            sharpe_std: 滚动 Sharpe 的标准差
        """
        window = RobustConfig.ROLLING_WINDOW
        k = RobustConfig.STABILITY_K
        
        if len(returns) < window + 10:
            return 0.0, 0.0
        
        # 滚动 Sharpe (简单实现)
        rolling_sharpes = []
        for i in range(window, len(returns)):
            window_ret = returns[i-window:i]
            rs = self._calc_sharpe(window_ret)
            rolling_sharpes.append(rs)
        
        if len(rolling_sharpes) == 0:
            return 0.0, 0.0
        
        rs_tensor = torch.tensor(rolling_sharpes, dtype=torch.float32)
        rs_mean = rs_tensor.mean().item()
        rs_std = rs_tensor.std().item()
        
        stability = rs_mean - k * rs_std
        return stability, rs_std
    
    def _calc_max_drawdown(self, returns: torch.Tensor) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 0.0
        
        # 累计净值
        cum_returns = (1 + returns).cumprod(dim=0)
        
        # 滚动最大值
        running_max = torch.cummax(cum_returns, dim=0)[0]
        
        # 回撤
        drawdown = (running_max - cum_returns) / (running_max + 1e-9)
        max_dd = drawdown.max().item()
        
        return max_dd
    
    def _rank_with_ties(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        计算 1-based 平均排名 (处理并列)
        
        Args:
            tensor: 1D tensor [N]
        
        Returns:
            ranks: 1D tensor [N]，1-based 平均排名
        """
        n = tensor.shape[0]
        if n == 0:
            return tensor.clone()
        
        # 向量化 Ranking 实现 (Performance Optimized)
        # 1. 排序
        # 必须在排序后的张量上使用 unique_consecutive
        sorted_indices = torch.argsort(tensor)
        sorted_tensor = tensor[sorted_indices]
        
        # 2. 找到唯一值和计数
        # unique_consecutive 仅合并相邻重复，因此依赖于步骤 1 的排序
        unique_vals, inverse_indices, counts = torch.unique_consecutive(
            sorted_tensor, return_counts=True, return_inverse=True
        )
        
        # 3. 计算每个唯一值的平均排名 (1-based)
        # 累积计数给出了该组的结束位置 (End Rank, 1-based)
        cumsum_counts = counts.cumsum(dim=0, dtype=torch.float64)
        
        # Start Rank (1-based) = End - count + 1
        # Avg Rank = (Start + End) / 2 = (2 * End - count + 1) / 2
        # 数学推导: ((cumsum - count + 1) + cumsum) / 2
        avg_ranks_unique = (2 * cumsum_counts - counts + 1) / 2.0
        
        # 4. 映射回排序后的数组
        # inverse_indices 将唯一值的平均排名广播回每个元素
        ranks_sorted = avg_ranks_unique[inverse_indices]
        
        # 5. 映射回原始顺序 (Scatter)
        ranks = torch.empty_like(tensor, dtype=torch.float64)
        ranks[sorted_indices] = ranks_sorted
        
        return ranks
    
    def _compute_daily_ic(self, factor_day: torch.Tensor, ret_day: torch.Tensor, 
                          mask_day: torch.Tensor) -> float:
        """
        计算单日 Spearman IC (信息系数)
        
        严格遵循 Implementation Plan 的 5 步公式：
        1. 掩码与样本过滤
        2. 中心化排名
        3. 标准差 (ddof=0)
        4. 分母检查
        5. 无偏 IC
        
        Args:
            factor_day: [N] 单日因子值
            ret_day: [N] 单日收益率
            mask_day: [N] 有效掩码
        
        Returns:
            float: IC 值，或 None 如果需要跳过
        """
        # 确定 eps 值
        dtype = factor_day.dtype
        eps = 1e-6 if dtype == torch.float32 else 1e-12
        
        # Step 1: Mask & Sample Filter
        # 组合有效掩码：valid_mask & isfinite(factor) & isfinite(ret)
        combined_mask = mask_day.clone()
        combined_mask &= torch.isfinite(factor_day)
        combined_mask &= torch.isfinite(ret_day)
        
        valid_count = combined_mask.sum().item()
        
        # Hard skip: 样本数 < 2
        if valid_count < 2:
            return None
        
        # 提取有效样本
        valid_factor = factor_day[combined_mask].to(torch.float64)
        valid_ret = ret_day[combined_mask].to(torch.float64)
        
        # Step 2: Centered Ranks
        rank_f = self._rank_with_ties(valid_factor)
        rank_r = self._rank_with_ties(valid_ret)
        
        xr = rank_f - rank_f.mean()
        yr = rank_r - rank_r.mean()
        
        # Step 3: Standard Deviation (ddof=0)
        # PyTorch: std(unbiased=False) 使用 ddof=0
        sx = xr.std(unbiased=False)
        sy = yr.std(unbiased=False)
        
        # Step 4: Denominator Check
        den = sx * sy
        if den <= eps:
            return None
        
        # Step 5: Unbiased IC
        ic = (xr * yr).mean() / den
        
        # Step 6: Clamping
        ic = torch.clamp(ic, -1.0, 1.0)
        
        return ic.item()
    
    def _compute_ic_metrics(self, factors: torch.Tensor, target_ret: torch.Tensor,
                            valid_mask: torch.Tensor) -> dict:
        """
        计算 IC/IR 相关指标
        
        Args:
            factors: [T, N] 因子值张量
            target_ret: [T, N] 收益率张量
            valid_mask: [T, N] 有效掩码
        
        Returns:
            dict: {
                'ic_mean': float,
                'ic_std': float,
                'ic_ir': float or None,
                'ic_ir_annual': float or None,
                'valid_ic_days': int,
                'total_ic_days': int,
                'skipped_ic_days': int
            }
        """
        T, N = factors.shape
        dtype = factors.dtype
        eps = 1e-6 if dtype == torch.float32 else 1e-12
        
        daily_ics = []
        valid_ic_days = 0
        skipped_ic_days = 0
        
        # 遍历每天计算 IC (注意: target_ret[t] 已经是 t->t+1 收益，直接对齐)
        for t in range(T):
            # 虽然 target_ret[-1] 为 0，但我们仍应尝试计算依赖 valid_mask 过滤
            mask_day = valid_mask[t]
            factor_day = factors[t]
            ret_day = target_ret[t]
            
            ic = self._compute_daily_ic(factor_day, ret_day, mask_day)
            
            if ic is not None:
                daily_ics.append(ic)
                valid_ic_days += 1
            else:
                skipped_ic_days += 1
        
        total_days = T   
        
        # Null Strategy: 有效天数 < 2
        if valid_ic_days < 2:
            return {
                'ic_mean': None,
                'ic_std': None,
                'ic_ir': None,
                'ic_ir_annual': None,
                'valid_ic_days': valid_ic_days,
                'total_ic_days': total_days,
                'skipped_ic_days': skipped_ic_days
            }
        
        # 转为 float64 进行聚合
        ic_tensor = torch.tensor(daily_ics, dtype=torch.float64)
        
        ic_mean = ic_tensor.mean().item()
        # Global IC Std: 使用 ddof=1 (unbiased=True)
        ic_std = ic_tensor.std(unbiased=True).item()
        
        # Null Strategy: IC Std <= eps
        if ic_std <= eps:
            return {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': None,
                'ic_ir_annual': None,
                'valid_ic_days': valid_ic_days,
                'total_ic_days': total_days,
                'skipped_ic_days': skipped_ic_days
            }
        
        ic_ir = ic_mean / ic_std
        ic_ir_annual = ic_ir * (252 ** 0.5)
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_ir_annual': ic_ir_annual,
            'valid_ic_days': valid_ic_days,
            'total_ic_days': total_days,
            'skipped_ic_days': skipped_ic_days
        }


# 保留旧类名以兼容现有代码
MemeBacktest = CBBacktest
