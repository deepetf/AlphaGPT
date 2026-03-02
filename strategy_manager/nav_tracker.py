"""
NavTracker - 净值追踪器

记录每日组合净值变化，计算收益率和最大回撤等绩效指标。
"""
import json
import os
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DailyRecord:
    """每日净值记录"""
    date: str              # YYYY-MM-DD
    nav: float             # 净值 (现金 + 持仓市值)
    cash: float            # 现金余额
    holdings_value: float  # 持仓市值
    holdings_count: int    # 持仓数量
    daily_ret: float       # 当日收益率
    cum_ret: float         # 累计收益率
    mdd: float             # 当前最大回撤


class NavTracker:
    """
    净值追踪器
    
    职责:
    1. 记录每日净值快照
    2. 计算收益率、最大回撤等绩效指标
    3. 持久化历史记录
    """
    
    DEFAULT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "execution", "portfolio", "nav_history.json"
    )
    
    VERSION = "1.0"
    
    def __init__(
        self, 
        initial_capital: float = 1_000_000.0,
        state_path: Optional[str] = None
    ):
        """
        初始化净值追踪器
        
        Args:
            initial_capital: 初始资金
            state_path: 状态文件路径
        """
        self.initial_capital = initial_capital
        self.state_path = state_path or self.DEFAULT_PATH
        self.records: List[DailyRecord] = []
        self.cash: float = initial_capital  # 当前现金
        self.peak_nav: float = initial_capital  # 历史最高净值
        
        self._ensure_dir()
        self.load_state()
    
    def _ensure_dir(self):
        """确保目录存在"""
        dir_path = os.path.dirname(self.state_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    # ==================== Core Methods ====================
    
    def calculate_nav(self, holdings_value: float) -> float:
        """
        计算当前净值
        
        Args:
            holdings_value: 当前持仓市值
            
        Returns:
            净值 = 现金 + 持仓市值
        """
        return self.cash + holdings_value
    
    def record_daily(
        self, 
        date: str, 
        holdings_value: float, 
        holdings_count: int
    ) -> DailyRecord:
        """
        记录每日净值快照
        
        Args:
            date: 日期 (YYYY-MM-DD)
            holdings_value: 持仓市值
            holdings_count: 持仓数量
            
        Returns:
            DailyRecord 对象
        """
        nav = self.calculate_nav(holdings_value)

        # 同日重复运行时，daily_ret 必须相对“上一交易日”而不是同日旧记录。
        overwrite_same_day = bool(self.records and self.records[-1].date == date)
        baseline_records = self.records[:-1] if overwrite_same_day else self.records

        if baseline_records:
            prev_nav = baseline_records[-1].nav
        else:
            prev_nav = self.initial_capital

        daily_ret = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
        cum_ret = (nav - self.initial_capital) / self.initial_capital

        # 同日覆盖时重建历史峰值，避免使用已被覆盖记录导致的 peak/mdd 偏差。
        historical_peak = max([self.initial_capital] + [r.nav for r in baseline_records])
        self.peak_nav = max(historical_peak, nav)
        mdd = (self.peak_nav - nav) / self.peak_nav if self.peak_nav > 0 else 0.0
        
        # 创建记录
        record = DailyRecord(
            date=date,
            nav=nav,
            cash=self.cash,
            holdings_value=holdings_value,
            holdings_count=holdings_count,
            daily_ret=daily_ret,
            cum_ret=cum_ret,
            mdd=mdd
        )
        
        # 避免同一天重复记录
        if overwrite_same_day:
            self.records[-1] = record
        else:
            self.records.append(record)
        
        self.save_state()
        logger.info(f"[NAV] {date}: ¥{nav:,.2f} (daily: {daily_ret*100:+.2f}%, cum: {cum_ret*100:+.2f}%, MDD: {mdd*100:.2f}%)")
        
        return record
    
    def adjust_cash(self, amount: float):
        """
        调整现金 (买入减少，卖出增加)
        
        Args:
            amount: 变动金额 (正数增加，负数减少)
        """
        self.cash += amount
        self.save_state()
        logger.debug(f"[CASH] Adjusted by {amount:+,.2f}, new balance: {self.cash:,.2f}")
    
    # ==================== Query Methods ====================
    
    def get_latest_nav(self) -> float:
        """获取最新净值"""
        if self.records:
            return self.records[-1].nav
        return self.initial_capital
    
    def get_latest_record(self) -> Optional[DailyRecord]:
        """获取最新记录"""
        return self.records[-1] if self.records else None
    
    def get_total_return(self) -> float:
        """获取累计收益率"""
        if self.records:
            return self.records[-1].cum_ret
        return 0.0
    
    def get_max_drawdown(self) -> float:
        """获取历史最大回撤"""
        if not self.records:
            return 0.0
        return max(r.mdd for r in self.records)
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率 (年化)
        
        Args:
            risk_free_rate: 无风险利率 (年化)
        """
        if len(self.records) < 2:
            return 0.0
        
        daily_rets = [r.daily_ret for r in self.records]
        
        import numpy as np
        mean_ret = np.mean(daily_rets)
        std_ret = np.std(daily_rets)
        
        if std_ret == 0:
            return 0.0
        
        # 年化
        annual_ret = mean_ret * 252
        annual_std = std_ret * (252 ** 0.5)
        
        return (annual_ret - risk_free_rate) / annual_std
    
    # ==================== Persistence ====================
    
    def save_state(self):
        """保存状态"""
        data = {
            "version": self.VERSION,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "peak_nav": self.peak_nav,
            "updated_at": datetime.now().isoformat(),
            "records": [asdict(r) for r in self.records]
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """加载状态"""
        if not os.path.exists(self.state_path):
            return
        
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.initial_capital = data.get("initial_capital", 1_000_000.0)
            self.cash = data.get("cash", self.initial_capital)
            self.peak_nav = data.get("peak_nav", self.initial_capital)
            
            self.records = []
            for rec in data.get("records", []):
                self.records.append(DailyRecord(**rec))
            
            logger.info(f"Loaded {len(self.records)} NAV records from {self.state_path}")
        except Exception as e:
            logger.error(f"Failed to load NAV state: {e}")
    
    # ==================== Utils ====================
    
    def summary(self) -> str:
        """生成绩效摘要"""
        if not self.records:
            return f"NAV Tracker: Initial Capital ¥{self.initial_capital:,.2f}, No records yet."
        
        latest = self.records[-1]
        sharpe = self.get_sharpe_ratio()
        max_mdd = self.get_max_drawdown()
        
        return (
            f"NAV Tracker Summary ({len(self.records)} days)\n"
            f"  Initial: ¥{self.initial_capital:,.2f}\n"
            f"  Current: ¥{latest.nav:,.2f}\n"
            f"  Return:  {latest.cum_ret*100:+.2f}%\n"
            f"  MaxDD:   {max_mdd*100:.2f}%\n"
            f"  Sharpe:  {sharpe:.2f}"
        )
    
    def reset(self):
        """重置追踪器"""
        self.records = []
        self.cash = self.initial_capital
        self.peak_nav = self.initial_capital
        self.save_state()
        logger.info("NAV Tracker reset.")
