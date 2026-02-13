"""
SimTrader - 模拟成交引擎

在模拟盘环境中执行订单，不涉及真实交易。
以传入的实时价格模拟成交，更新组合状态。
"""
import logging
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

from strategy_manager.cb_portfolio import CBPortfolioManager, CBPosition
from strategy_manager.nav_tracker import NavTracker

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class SimOrder:
    """模拟订单"""
    code: str
    name: str
    side: OrderSide
    shares: int          # 张数
    target_price: float  # 目标价格 (用于模拟成交)
    is_take_profit: bool = False  # 是否止盈卖出
    
    
@dataclass
class TradeRecord:
    """成交记录"""
    date: str            # 交易日期 (YYYY-MM-DD)
    code: str
    name: str
    side: OrderSide
    shares: int
    price: float
    amount: float        # 成交金额 (shares * price)
    timestamp: str       # 交易时间 (YYYY-MM-DDTHH:MM:SS)
    

class SimTrader:
    """
    模拟成交引擎
    
    职责:
    1. 接收订单列表
    2. 以实时价格模拟成交
    3. 更新组合和现金状态
    """
    
    DEFAULT_HISTORY_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "execution", "portfolio", "trade_history.json"
    )
    
    def __init__(
        self, 
        portfolio: CBPortfolioManager,
        nav_tracker: NavTracker,
        fee_rate: float = 0.0005,  # 单边费率
        history_path: Optional[str] = None
    ):
        """
        初始化模拟交易器
        
        Args:
            portfolio: 组合管理器
            nav_tracker: 净值追踪器
            fee_rate: 交易费率 (单边)
            history_path: 交易记录保存路径
        """
        self.portfolio = portfolio
        self.nav_tracker = nav_tracker
        self.fee_rate = fee_rate
        self.history_path = history_path or self.DEFAULT_HISTORY_PATH
        self.trade_history: List[TradeRecord] = []
        
        self.load_history()
    
    def save_history(self):
        """保存交易纪录到 JSON"""
        data = [asdict(t) for t in self.trade_history]
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def load_history(self):
        """从 JSON 加载交易纪录"""
        if not os.path.exists(self.history_path):
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.trade_history = [TradeRecord(**t) for t in data]
            logger.info(f"Loaded {len(self.trade_history)} trade records from {self.history_path}")
        except Exception as e:
            logger.error(f"Failed to load trade history: {e}")

    def execute(
        self, 
        orders: List[SimOrder], 
        prices: Dict[str, float],
        date: str
    ) -> List[TradeRecord]:
        """
        执行订单列表
        
        Args:
            orders: 订单列表
            prices: 实时价格字典 {code: price}
            date: 成交日期 (YYYY-MM-DD)
            
        Returns:
            成交记录列表
        """
        records = []
        
        for order in orders:
            # 获取成交价
            if getattr(order, "is_take_profit", False) and order.target_price > 0:
                exec_price = order.target_price
            else:
                exec_price = prices.get(order.code, order.target_price)
            
            if order.side == OrderSide.SELL:
                record = self._execute_sell(order, exec_price, date)
            else:
                record = self._execute_buy(order, exec_price, date)
            
            if record:
                records.append(record)
                self.trade_history.append(record)
        
        if records:
            self.save_history()
            
        return records
    
    def _execute_buy(
        self, 
        order: SimOrder, 
        price: float, 
        date: str
    ) -> Optional[TradeRecord]:
        """执行买入"""
        amount = order.shares * price
        fee = amount * self.fee_rate
        total_cost = amount + fee
        
        # 检查现金是否足够
        if self.nav_tracker.cash < total_cost:
            # logger.warning(f"[SimTrader] 现金不足: 需要 {total_cost:.2f}, 可用 {self.nav_tracker.cash:.2f}")
            return None
        
        # 扣减现金
        self.nav_tracker.adjust_cash(-total_cost)
        
        # 更新持仓
        existing = self.portfolio.get_position(order.code)
        if existing:
            # 加仓
            new_shares = existing.shares + order.shares
            self.portfolio.update_position(order.code, new_shares, price)
        else:
            # 新建持仓
            self.portfolio.add_position(order.code, order.name, order.shares, price, date)
        
        # 模拟 14:55 成交
        sim_time = f"{date}T14:55:00"
        
        record = TradeRecord(
            date=date,
            code=order.code,
            name=order.name,
            side=OrderSide.BUY,
            shares=order.shares,
            price=price,
            amount=total_cost,
            timestamp=sim_time
        )
        
        logger.info(f"[SimTrader] BUY {order.code} {order.name} x{order.shares} @ {price:.2f} = {total_cost:.2f}")
        return record
    
    def _execute_sell(
        self, 
        order: SimOrder, 
        price: float, 
        date: str
    ) -> Optional[TradeRecord]:
        """执行卖出"""
        existing = self.portfolio.get_position(order.code)
        if not existing:
            logger.warning(f"[SimTrader] 卖出失败: 无持仓 {order.code}")
            return None
        
        # 实际卖出张数 (不能超过持仓)
        actual_shares = min(order.shares, existing.shares)
        amount = actual_shares * price
        fee = amount * self.fee_rate
        net_proceeds = amount - fee
        
        # 增加现金
        self.nav_tracker.adjust_cash(net_proceeds)
        
        # 更新持仓
        remaining = existing.shares - actual_shares
        if remaining > 0:
            self.portfolio.update_position(order.code, remaining, price)
        else:
            self.portfolio.remove_position(order.code)
        
        # 模拟 14:55 成交
        sim_time = f"{date}T14:55:00"
        
        record = TradeRecord(
            date=date,
            code=order.code,
            name=order.name,
            side="SELL-TP" if getattr(order, "is_take_profit", False) else OrderSide.SELL,
            shares=actual_shares,
            price=price,
            amount=net_proceeds,
            timestamp=sim_time
        )
        
        logger.info(f"[SimTrader] SELL {order.code} {order.name} x{actual_shares} @ {price:.2f} = {net_proceeds:.2f}")
        return record
    
    def get_trade_count(self) -> int:
        """获取成交笔数"""
        return len(self.trade_history)
    
    def get_total_volume(self) -> float:
        """获取总成交金额"""
        return sum(t.amount for t in self.trade_history)
