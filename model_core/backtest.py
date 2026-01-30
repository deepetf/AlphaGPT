"""
可转债回测引擎 (Convertible Bond Backtest Engine)

支持 Top-K 轮动策略的向量化回测，用于 AlphaGPT 的 RL 训练。
V2: 增加分段验证、滚动稳定性、最大回撤、可交易性约束。
"""
import torch
from .config import ModelConfig, RobustConfig


class CBBacktest:
    """
    可转债 Top-K 轮动回测器
    
    核心逻辑:
    1. 每天根据因子值对所有有效转债排序
    2. 选出因子值最高的 Top-K 个标的
    3. 等权持仓 (1/K)
    4. 计算组合收益和夏普比率作为 Reward
    """
    
    def __init__(self, top_k: int = 20, fee_rate: float = None):
        """
        Args:
            top_k: 每天持有的转债数量
            fee_rate: 单边交易费率，默认使用 RobustConfig.FEE_RATE
        """
        self.top_k = top_k
        self.fee_rate = fee_rate if fee_rate is not None else RobustConfig.FEE_RATE
        # 要求有效标的至少是 top_k 的 2 倍 (最少 30 个)，保证选择性
        self.min_valid_count = max(30, top_k * 2)
    
    def evaluate(self, factors: torch.Tensor, target_ret: torch.Tensor, 
                 valid_mask: torch.Tensor) -> tuple:
        """
        评估因子的回测表现
        
        Args:
            factors: [Time, Assets] 因子值张量
            target_ret: [Time, Assets] 资产收益率张量 (T+1 收益)
            valid_mask: [Time, Assets] 有效标的掩码 (True=可交易)
        
        Returns:
            reward: 标量，用于 RL 训练的奖励信号
            cum_ret: 累计收益率 (用于日志输出)
            sharpe: 夏普比率
        """
        device = factors.device
        T, N = factors.shape
        
        # 1. 应用有效性掩码
        masked_factors = factors.clone()
        masked_factors[~valid_mask] = -1e9
        
        # 2. 计算每日有效标的数量
        daily_valid_count = valid_mask.sum(dim=1)  # [T]
        
        # 3. 标记有效交易日 (有效标的 >= min_valid_count)
        valid_trading_day = daily_valid_count >= self.min_valid_count  # [T]
        
        # 4. 精确选择 Top-K (使用 topk 索引，避免超额持仓)
        # 确定每天实际可选的数量 (min(top_k, 有效标的数))
        actual_k = torch.clamp(daily_valid_count, max=self.top_k)  # [T]
        
        # 初始化权重矩阵
        weights = torch.zeros(T, N, device=device)
        
        for t in range(T):
            if not valid_trading_day[t]:
                continue  # 有效标的不足，跳过
            
            k = int(actual_k[t].item())
            if k == 0:
                continue
            
            # 使用 topk 获取精确的 K 个索引
            _, top_indices = torch.topk(masked_factors[t], k=k, largest=True)
            
            # 等权分配
            weights[t, top_indices] = 1.0 / k
        
        # 5. 计算换手率
        prev_weights = torch.roll(weights, 1, dims=0)
        prev_weights[0] = 0
        turnover = torch.abs(weights - prev_weights).sum(dim=1)  # [T]
        
        # 6. 计算交易成本
        tx_cost = turnover * self.fee_rate * 2
        
        # 7. 计算组合收益
        gross_ret = (weights * target_ret).sum(dim=1)  # [T]
        net_ret = gross_ret - tx_cost  # [T]
        
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
        avg_holding = weights.sum(dim=1)[valid_trading_day].mean()  # 平均每天持仓数量
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
        
        # 应用有效性掩码
        masked_factors = factors.clone()
        masked_factors[~valid_mask] = -1e9
        
        # 计算每日有效标的数量
        daily_valid_count = valid_mask.sum(dim=1)
        valid_trading_day = daily_valid_count >= self.min_valid_count
        actual_k = torch.clamp(daily_valid_count, max=self.top_k)
        
        # 精确选择 Top-K
        weights = torch.zeros(T, N, device=device)
        daily_holdings = []
        
        for t in range(T):
            if not valid_trading_day[t]:
                daily_holdings.append([])
                continue
            
            k = int(actual_k[t].item())
            if k == 0:
                daily_holdings.append([])
                continue
            
            _, top_indices = torch.topk(masked_factors[t], k=k, largest=True)
            weights[t, top_indices] = 1.0 / k
            daily_holdings.append(top_indices.tolist())
        
        # 换手率和交易成本
        prev_weights = torch.roll(weights, 1, dims=0)
        prev_weights[0] = 0
        turnover = torch.abs(weights - prev_weights).sum(dim=1)
        tx_cost = turnover * self.fee_rate * 2
        
        # 组合收益
        gross_ret = (weights * target_ret).sum(dim=1)
        net_ret = gross_ret - tx_cost
        
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
        
        avg_holding = weights.sum(dim=1)[valid_trading_day].mean()
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
                        valid_mask: torch.Tensor, split_idx: int) -> dict:
        """
        稳健性评估方法 (Robust Evaluation)
        
        返回分段 Sharpe、滚动稳定性、最大回撤、活跃率等多维指标。
        
        Args:
            factors: [Time, Assets] 因子值张量
            target_ret: [Time, Assets] 资产收益率张量 (T+1 收益)
            valid_mask: [Time, Assets] 有效标的掩码
            split_idx: 训练/验证切分索引
        
        Returns:
            dict: 包含多维评估指标
        """
        device = factors.device
        T, N = factors.shape
        
        # 1. 基础计算 (与 evaluate 相同)
        masked_factors = factors.clone()
        masked_factors[~valid_mask] = -1e9
        
        daily_valid_count = valid_mask.sum(dim=1)
        valid_trading_day = daily_valid_count >= self.min_valid_count
        actual_k = torch.clamp(daily_valid_count, max=self.top_k)
        
        weights = torch.zeros(T, N, device=device)
        
        for t in range(T):
            if not valid_trading_day[t]:
                continue
            k = int(actual_k[t].item())
            if k == 0:
                continue
            _, top_indices = torch.topk(masked_factors[t], k=k, largest=True)
            weights[t, top_indices] = 1.0 / k
        
        # 换手和交易成本
        prev_weights = torch.roll(weights, 1, dims=0)
        prev_weights[0] = 0
        turnover = torch.abs(weights - prev_weights).sum(dim=1)
        tx_cost = turnover * self.fee_rate * 2
        
        gross_ret = (weights * target_ret).sum(dim=1)
        net_ret = gross_ret - tx_cost
        
        # 2. 分段计算 Sharpe
        # Train: [0, split_idx), Val: [split_idx, T)
        train_mask = valid_trading_day.clone()
        train_mask[split_idx:] = False
        val_mask = valid_trading_day.clone()
        val_mask[:split_idx] = False
        
        train_ret = net_ret[train_mask]
        val_ret = net_ret[val_mask]
        
        sharpe_train = self._calc_sharpe(train_ret)
        sharpe_val = self._calc_sharpe(val_ret)
        
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
            cum_ret = (1 + valid_net_ret).prod() - 1
            n_days = len(valid_net_ret)
            # 年化: (1 + 累计收益)^(252/交易天数) - 1
            annualized_ret = ((1 + cum_ret) ** (252.0 / n_days) - 1).item() if hasattr(cum_ret, 'item') else ((1 + cum_ret) ** (252.0 / n_days) - 1)
        else:
            annualized_ret = 0.0
        
        # 7. 全局 Sharpe
        sharpe_all = self._calc_sharpe(valid_net_ret)
        
        return {
            'sharpe_train': sharpe_train,
            'sharpe_val': sharpe_val,
            'sharpe_all': sharpe_all,
            'stability_metric': stability_metric,
            'sharpe_std': sharpe_std,
            'max_drawdown': max_drawdown,
            'active_ratio': active_ratio,
            'annualized_ret': annualized_ret,  # 年化收益率
            'valid_days_train': int(train_mask.sum().item()),
            'valid_days_val': int(val_mask.sum().item()),
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


# 保留旧类名以兼容现有代码
MemeBacktest = CBBacktest