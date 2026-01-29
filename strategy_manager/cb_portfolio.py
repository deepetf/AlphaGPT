"""
可转债组合管理器 (CB Portfolio Manager)

管理可转债持仓状态，负责 CRUD 操作。调仓逻辑由 Rebalancer 模块处理。
"""
import json
import os
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from datetime import datetime

# 使用标准 logging (兼容性更好)
logger = logging.getLogger(__name__)


@dataclass
class CBPosition:
    """可转债持仓"""
    code: str           # 113025.SH
    name: str           # 包钢转债
    shares: int         # 持仓张数
    avg_cost: float     # 平均成本价
    last_price: float   # 最新价格 (用于计算市值)
    entry_date: str     # 入场日期 (YYYY-MM-DD)
    
    @property
    def market_value(self) -> float:
        """持仓市值 (元) - 注: price 为 元/张"""
        # shares (张) * last_price (元/张)
        return self.shares * self.last_price
    
    @property
    def cost_value(self) -> float:
        """持仓成本 (元)"""
        return self.shares * self.avg_cost
    
    @property
    def pnl(self) -> float:
        """浮动盈亏 (元)"""
        return self.market_value - self.cost_value
    
    @property
    def pnl_pct(self) -> float:
        """浮动盈亏率"""
        if self.cost_value == 0:
            return 0.0
        return self.pnl / self.cost_value


class CBPortfolioManager:
    """
    可转债组合管理器
    
    职责: 仅管理持仓状态 (CRUD)，不涉及调仓逻辑。
    """
    
    # 默认持久化路径
    DEFAULT_STATE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "execution", "portfolio", "portfolio_state.json"
    )
    
    # 状态文件版本号 (用于未来兼容性升级)
    VERSION = "1.0"
    
    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path or self.DEFAULT_STATE_PATH
        self.positions: Dict[str, CBPosition] = {}
        self._ensure_dir()
        self.load_state()
    
    def _ensure_dir(self):
        """确保持久化目录存在"""
        dir_path = os.path.dirname(self.state_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    # ==================== Query Methods ====================
    
    def get_position_codes(self) -> List[str]:
        """获取当前持仓代码列表"""
        return list(self.positions.keys())
    
    def get_position(self, code: str) -> Optional[CBPosition]:
        """获取单个持仓"""
        return self.positions.get(code)
    
    def get_all_positions(self) -> List[CBPosition]:
        """获取所有持仓"""
        return list(self.positions.values())
    
    def get_holdings_value(self) -> float:
        """获取总持仓市值"""
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_holdings_count(self) -> int:
        """获取持仓数量"""
        return len(self.positions)
    
    # ==================== Mutation Methods ====================
    
    def add_position(self, code: str, name: str, shares: int, 
                     price: float, date: str) -> CBPosition:
        """
        新增持仓
        
        Args:
            code: 转债代码
            name: 转债名称
            shares: 张数
            price: 成交价
            date: 成交日期
        """
        if code in self.positions:
            logger.warning(f"Position {code} already exists, use update_position instead")
            return self.positions[code]
        
        pos = CBPosition(
            code=code,
            name=name,
            shares=shares,
            avg_cost=price,
            last_price=price,
            entry_date=date
        )
        self.positions[code] = pos
        self.save_state()
        logger.info(f"[+] Added position: {code} {name} x{shares} @ {price}")
        return pos
    
    def update_position(self, code: str, shares: int, price: float) -> Optional[CBPosition]:
        """
        更新持仓 (加仓/减仓)
        
        Args:
            code: 转债代码
            shares: 新的持仓张数 (0 表示清仓)
            price: 最新价格
        """
        if code not in self.positions:
            logger.warning(f"Position {code} not found")
            return None
        
        pos = self.positions[code]
        
        if shares <= 0:
            # 清仓
            return self.remove_position(code)
        
        # 更新持仓数量和最新价
        old_shares = pos.shares
        pos.shares = shares
        pos.last_price = price
        
        # 如果是加仓，更新平均成本
        if shares > old_shares:
            # 简化处理: 加权平均
            added = shares - old_shares
            pos.avg_cost = (pos.avg_cost * old_shares + price * added) / shares
        
        self.save_state()
        logger.info(f"[~] Updated position: {code} {old_shares} -> {shares} @ {price}")
        return pos
    
    def update_price(self, code: str, price: float) -> Optional[CBPosition]:
        """仅更新最新价格"""
        if code not in self.positions:
            return None
        self.positions[code].last_price = price
        self.save_state()
        return self.positions[code]
    
    def remove_position(self, code: str) -> Optional[CBPosition]:
        """移除持仓"""
        if code not in self.positions:
            logger.warning(f"Position {code} not found")
            return None
        
        pos = self.positions.pop(code)
        self.save_state()
        logger.info(f"[-] Removed position: {code} {pos.name}")
        return pos
    
    def clear_all(self):
        """清空所有持仓"""
        count = len(self.positions)
        self.positions.clear()
        self.save_state()
        logger.info(f"[x] Cleared {count} positions")
    
    # ==================== Persistence ====================
    
    def save_state(self):
        """保存状态到 JSON"""
        data = {
            "version": self.VERSION,
            "updated_at": datetime.now().isoformat(),
            "positions": {k: asdict(v) for k, v in self.positions.items()}
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """从 JSON 加载状态"""
        if not os.path.exists(self.state_path):
            self.positions = {}
            return
        
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 版本检查 (未来可用于迁移)
            version = data.get("version", "1.0")
            if version != self.VERSION:
                logger.warning(f"Portfolio state version mismatch: {version} vs {self.VERSION}")
            
            positions_data = data.get("positions", {})
            self.positions = {}
            for code, pos_dict in positions_data.items():
                # 过滤掉 dataclass 中不存在的字段
                valid_fields = {f.name for f in CBPosition.__dataclass_fields__.values()}
                filtered = {k: v for k, v in pos_dict.items() if k in valid_fields}
                self.positions[code] = CBPosition(**filtered)
            
            logger.info(f"Loaded {len(self.positions)} positions from {self.state_path}")
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
            self.positions = {}
    
    # ==================== Utils ====================
    
    def summary(self) -> str:
        """生成持仓摘要"""
        if not self.positions:
            return "Empty portfolio"
        
        lines = [f"Portfolio: {len(self.positions)} positions, Total: ¥{self.get_holdings_value():,.2f}"]
        for pos in self.positions.values():
            pnl_str = f"+{pos.pnl:.2f}" if pos.pnl >= 0 else f"{pos.pnl:.2f}"
            lines.append(f"  {pos.code} {pos.name}: {pos.shares}张 @ {pos.last_price:.2f} ({pnl_str})")
        return "\n".join(lines)
