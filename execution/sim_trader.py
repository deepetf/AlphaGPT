"""
SimTrader - 模拟成交引擎

在模拟盘环境中执行订单，不涉及真实交易。
以传入的实时价格模拟成交，更新组合状态。
"""
import json
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional

from strategy_manager.cb_portfolio import CBPortfolioManager
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
    shares: int
    target_price: float
    is_take_profit: bool = False


@dataclass
class TradeRecord:
    """成交记录"""

    date: str
    code: str
    name: str
    side: str
    shares: int
    price: float
    amount: float
    timestamp: str


class SimTrader:
    """模拟成交引擎"""

    DEFAULT_HISTORY_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "execution",
        "portfolio",
        "trade_history.json",
    )

    def __init__(
        self,
        portfolio: CBPortfolioManager,
        nav_tracker: NavTracker,
        fee_rate: float = 0.0005,
        history_path: Optional[str] = None,
    ):
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

    def execute(self, orders: List[SimOrder], prices: Dict[str, float], date: str) -> List[TradeRecord]:
        """执行订单列表"""
        records: List[TradeRecord] = []
        for order in orders:
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

    def _execute_buy(self, order: SimOrder, price: float, date: str) -> Optional[TradeRecord]:
        amount = order.shares * price
        fee = amount * self.fee_rate
        total_cost = amount + fee

        if self.nav_tracker.cash < total_cost:
            logger.warning(
                f"[SimTrader] BUY rejected: cash insufficient for {order.code} "
                f"(need={total_cost:.2f}, cash={self.nav_tracker.cash:.2f}, "
                f"shares={order.shares}, price={price:.2f})"
            )
            return None

        self.nav_tracker.adjust_cash(-total_cost)
        existing = self.portfolio.get_position(order.code)
        if existing:
            self.portfolio.update_position(order.code, existing.shares + order.shares, price)
        else:
            self.portfolio.add_position(order.code, order.name, order.shares, price, date)

        record = TradeRecord(
            date=date,
            code=order.code,
            name=order.name,
            side=OrderSide.BUY.value,
            shares=order.shares,
            price=price,
            amount=total_cost,
            timestamp=f"{date}T14:55:00",
        )
        logger.info(f"[SimTrader] BUY {order.code} {order.name} x{order.shares} @ {price:.2f} = {total_cost:.2f}")
        return record

    def _execute_sell(self, order: SimOrder, price: float, date: str) -> Optional[TradeRecord]:
        existing = self.portfolio.get_position(order.code)
        if not existing:
            logger.warning(f"[SimTrader] 卖出失败: 无持仓 {order.code}")
            return None

        actual_shares = min(order.shares, existing.shares)
        amount = actual_shares * price
        fee = amount * self.fee_rate
        net_proceeds = amount - fee

        self.nav_tracker.adjust_cash(net_proceeds)
        remaining = existing.shares - actual_shares
        if remaining > 0:
            self.portfolio.update_position(order.code, remaining, price)
        else:
            self.portfolio.remove_position(order.code)

        side = "SELL-TP" if getattr(order, "is_take_profit", False) else OrderSide.SELL.value
        record = TradeRecord(
            date=date,
            code=order.code,
            name=order.name,
            side=side,
            shares=actual_shares,
            price=price,
            amount=net_proceeds,
            timestamp=f"{date}T14:55:00",
        )
        logger.info(f"[SimTrader] SELL {order.code} {order.name} x{actual_shares} @ {price:.2f} = {net_proceeds:.2f}")
        return record

    def get_trade_count(self) -> int:
        """获取成交笔数"""
        return len(self.trade_history)

    def get_total_volume(self) -> float:
        """获取总成交金额"""
        return sum(t.amount for t in self.trade_history)
