"""
可转债交易接口抽象 (CB Trader Interface)

提供交易接口抽象，支持文件输出 (FileTrader) 和未来的实盘扩展。
"""
import os
import csv
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """交易订单"""
    code: str           # 转债代码 (113025.SH)
    name: str           # 转债名称
    side: OrderSide     # 买卖方向
    quantity: int       # 交易数量 (张)
    price: float        # 参考价格 (通常是收盘价)
    reason: str = ""    # 入选/剔除原因
    rank: int = 0       # 排名 (对于买入订单)
    
    def to_csv_row(self) -> dict:
        """转换为 CSV 行"""
        return {
            "code": self.code,
            "name": self.name,
            "side": self.side.value if isinstance(self.side, OrderSide) else self.side,
            "quantity": self.quantity,
            "price": self.price,
            "reason": self.reason,
            "rank": self.rank
        }


@dataclass
class OrderResult:
    """订单提交结果"""
    success: bool
    submitted_count: int
    failed_count: int
    message: str
    orders: List[Order]
    
    @property
    def total_count(self) -> int:
        return self.submitted_count + self.failed_count


class BaseCBTrader(ABC):
    """交易接口基类"""
    
    @abstractmethod
    def get_positions(self) -> List[str]:
        """获取当前持仓代码列表"""
        pass
    
    @abstractmethod
    def submit_orders(self, orders: List[Order], date: str) -> OrderResult:
        """提交订单"""
        pass


class FileTrader(BaseCBTrader):
    """
    文件交易器 - 将订单写入 CSV 文件
    
    适用于:
    - 人工检查/模拟运行
    - 对接外部交易终端 (如 QMT 扫描 CSV 下单)
    """
    
    DEFAULT_OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "execution", "orders"
    )
    
    # CSV 列定义
    CSV_COLUMNS = ["date", "code", "name", "side", "quantity", "price", "reason", "rank"]
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_positions(self) -> List[str]:
        """
        获取持仓信息
        
        注意: FileTrader 仅负责生成订单文件，不持有状态。
        状态管理应使用 CBPortfolioManager。
        """
        logger.warning("FileTrader does not support get_positions(). Use CBPortfolioManager instead.")
        raise NotImplementedError("FileTrader.get_positions is not supported. Use CBPortfolioManager.")
    
    def submit_orders(self, orders: List[Order], date: str) -> OrderResult:
        """
        将订单写入 CSV 文件
        
        Args:
            orders: 订单列表
            date: 交易日期 (YYYY-MM-DD)
        
        Returns:
            OrderResult: 提交结果
        """
        if not orders:
            return OrderResult(
                success=True,
                submitted_count=0,
                failed_count=0,
                message="No orders to submit",
                orders=[]
            )
        
        # 生成文件名
        filename = f"orders_{date}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()
                
                for order in orders:
                    row = order.to_csv_row()
                    row["date"] = date
                    writer.writerow(row)
            
            logger.info(f"Orders written to: {filepath}")
            
            return OrderResult(
                success=True,
                submitted_count=len(orders),
                failed_count=0,
                message=f"Orders saved to {filepath}",
                orders=orders
            )
        except Exception as e:
            logger.error(f"Failed to write orders: {e}")
            return OrderResult(
                success=False,
                submitted_count=0,
                failed_count=len(orders),
                message=f"Failed to write orders: {e}",
                orders=orders
            )
    
    def read_orders(self, date: str) -> List[Order]:
        """
        读取指定日期的订单文件
        
        Args:
            date: 交易日期 (YYYY-MM-DD)
        
        Returns:
            订单列表
        """
        filename = f"orders_{date}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            return []
        
        orders = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orders.append(Order(
                        code=row["code"],
                        name=row["name"],
                        side=OrderSide(row["side"]),
                        quantity=int(row["quantity"]),
                        price=float(row["price"]),
                        reason=row.get("reason", ""),
                        rank=int(row.get("rank", 0))
                    ))
        except Exception as e:
            logger.error(f"Failed to read orders: {e}")
        
        return orders
    
    def get_order_files(self) -> List[str]:
        """获取所有订单文件列表"""
        files = []
        for f in os.listdir(self.output_dir):
            if f.startswith("orders_") and f.endswith(".csv"):
                files.append(f)
        return sorted(files)
