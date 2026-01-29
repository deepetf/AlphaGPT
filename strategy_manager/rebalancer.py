"""
可转债调仓逻辑 (CB Rebalancer)

提供调仓计算的纯函数和 Rebalancer 类。
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from execution.cb_trader import Order, OrderSide

logger = logging.getLogger(__name__)


def compute_rebalance(
    current_codes: List[str],
    target_codes: List[str]
) -> Tuple[List[str], List[str]]:
    """
    计算调仓差异 (纯函数)
    
    Args:
        current_codes: 当前持仓代码列表
        target_codes: 目标持仓代码列表
    
    Returns:
        (sell_list, buy_list): 需要卖出和买入的代码列表
    """
    current_set = set(current_codes)
    target_set = set(target_codes)
    
    sell_list = list(current_set - target_set)
    buy_list = list(target_set - current_set)
    
    return sell_list, buy_list


@dataclass
class RebalanceResult:
    """调仓结果"""
    sell_codes: List[str]
    buy_codes: List[str]
    hold_codes: List[str]
    total_changes: int
    
    @property
    def has_changes(self) -> bool:
        return self.total_changes > 0


@dataclass
class AssetInfo:
    """资产信息"""
    code: str
    name: str
    price: float
    score: float = 0.0
    rank: int = 0


class CBRebalancer:
    """
    可转债调仓器
    
    负责:
    1. 计算调仓差异
    2. 生成交易订单
    3. 计算各项数量 (等权分配)
    """
    
    def __init__(self, total_capital: float = 100000.0):
        """
        Args:
            total_capital: 总资金 (元)
        """
        self.total_capital = total_capital
    
    def compute(
        self,
        current_codes: List[str],
        target_codes: List[str]
    ) -> RebalanceResult:
        """
        计算调仓结果
        
        Args:
            current_codes: 当前持仓代码列表
            target_codes: 目标持仓代码列表
        
        Returns:
            RebalanceResult: 调仓结果
        """
        sell_list, buy_list = compute_rebalance(current_codes, target_codes)
        
        current_set = set(current_codes)
        target_set = set(target_codes)
        hold_list = list(current_set & target_set)
        
        return RebalanceResult(
            sell_codes=sell_list,
            buy_codes=buy_list,
            hold_codes=hold_list,
            total_changes=len(sell_list) + len(buy_list)
        )
    
    def generate_orders(
        self,
        current_codes: List[str],
        target_info: List[AssetInfo],
        current_holdings: Optional[Dict[str, int]] = None,
        sell_prices: Optional[Dict[str, float]] = None
    ) -> List[Order]:
        """
        生成交易订单
        
        Args:
            current_codes: 当前持仓代码列表
            target_info: 目标资产信息列表 (包含 code, name, price, score, rank)
            current_holdings: 当前持仓数量 {code: shares}，用于生成卖出数量
            sell_prices: 卖出标的参考价格 {code: price}
        
        Returns:
            订单列表 (先卖后买)
        """
        target_codes = [info.code for info in target_info]
        target_map = {info.code: info for info in target_info}
        sell_prices = sell_prices or {}
        
        sell_list, buy_list = compute_rebalance(current_codes, target_codes)
        
        orders: List[Order] = []
        
        # 生成卖出订单
        for code in sell_list:
            shares = 0
            if current_holdings and code in current_holdings:
                shares = current_holdings[code]
            
            # 优先从 sell_prices 获取，其次从 target_map 获取 (虽然卖出通常意味不在 target)
            price = 0.0
            name = code
            
            if code in sell_prices:
                price = sell_prices[code]
            elif code in target_map:
                price = target_map[code].price
                name = target_map[code].name
            
            orders.append(Order(
                code=code,
                name=name,
                side=OrderSide.SELL,
                quantity=shares,
                price=price,
                reason="剔除出Top-K",
                rank=0
            ))
        
        # 计算每个目标资产的分配金额 (等权)
        # 注意: 这里假设 self.total_capital 是 "目标总权益" (Total Portfolio Value)，而非仅 "可用现金"。
        # 因此，每个标的的目标市值 = 总权益 / 标的数量。
        if buy_list and len(target_info) > 0:
            capital_per_asset = self.total_capital / len(target_info)
        else:
            capital_per_asset = 0
            
        # TODO: 未来对接实盘时，需在此处引入 available_cash (可用现金) 约束。
        # 当前逻辑：假设可以足额买入 (Total Equity / K)。
        # 实盘逻辑：Buy_Amount = min(Target_Amount, Available_Cash)，或者进行动态持仓再平衡(Rebalance)释放现金。
        
        # 生成买入订单
        for code in buy_list:
            if code not in target_map:
                continue
            
            info = target_map[code]
            
            # 计算可买入张数 (向下取整，可转债最小交易单位是 10 张)
            if info.price > 0:
                shares = int(capital_per_asset / info.price / 10) * 10
            else:
                shares = 0
            
            if shares <= 0:
                continue
            
            orders.append(Order(
                code=info.code,
                name=info.name,
                side=OrderSide.BUY,
                quantity=shares,
                price=info.price,
                reason="Top-K入选",
                rank=info.rank
            ))
        
        return orders
    
    def summary(self, result: RebalanceResult) -> str:
        """生成调仓摘要"""
        lines = [
            f"Rebalance Summary:",
            f"  Sell: {len(result.sell_codes)} - {result.sell_codes[:5]}{'...' if len(result.sell_codes) > 5 else ''}",
            f"  Buy:  {len(result.buy_codes)} - {result.buy_codes[:5]}{'...' if len(result.buy_codes) > 5 else ''}",
            f"  Hold: {len(result.hold_codes)}",
            f"  Total Changes: {result.total_changes}"
        ]
        return "\n".join(lines)
