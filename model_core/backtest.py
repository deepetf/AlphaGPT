"""
可转债回测引擎 (Convertible Bond Backtest Engine)

支持 Top-K 轮动策略的向量化回测，用于 AlphaGPT 的 RL 训练。
"""
import torch
from .config import ModelConfig


class CBBacktest:
    """
    可转债 Top-K 轮动回测器
    
    核心逻辑:
    1. 每天根据因子值对所有有效转债排序
    2. 选出因子值最高的 Top-K 个标的
    3. 等权持仓 (1/K)
    4. 计算组合收益和夏普比率作为 Reward
    """
    
    def __init__(self, top_k: int = 20, fee_rate: float = 0.0001):
        """
        Args:
            top_k: 每天持有的转债数量
            fee_rate: 单边交易费率 (万分之一 = 0.0001)
        """
        self.top_k = top_k
        self.fee_rate = fee_rate
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


# 保留旧类名以兼容现有代码
MemeBacktest = CBBacktest