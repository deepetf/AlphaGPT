"""
NavTracker 单元测试

测试净值追踪器的核心功能:
1. NAV 计算
2. 每日记录
3. 收益率与最大回撤计算
"""
import pytest
import os
import tempfile
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_manager.nav_tracker import NavTracker, DailyRecord


class TestNavTracker:
    """测试 NavTracker"""
    
    @pytest.fixture
    def temp_path(self):
        """创建临时文件路径"""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def tracker(self, temp_path):
        """创建 NavTracker 实例"""
        return NavTracker(initial_capital=1_000_000.0, state_path=temp_path)
    
    def test_initial_state(self, tracker):
        """测试初始状态"""
        assert tracker.initial_capital == 1_000_000.0
        assert tracker.cash == 1_000_000.0
        assert tracker.peak_nav == 1_000_000.0
        assert len(tracker.records) == 0
    
    def test_calculate_nav(self, tracker):
        """测试 NAV 计算"""
        # 初始: 现金 100 万，无持仓
        nav = tracker.calculate_nav(holdings_value=0)
        assert nav == 1_000_000.0
        
        # 假设买入后: 现金 50 万，持仓市值 52 万
        tracker.cash = 500_000.0
        nav = tracker.calculate_nav(holdings_value=520_000.0)
        assert nav == 1_020_000.0
    
    def test_record_daily_first_day(self, tracker):
        """测试首日记录"""
        record = tracker.record_daily(
            date="2026-02-08",
            holdings_value=0,
            holdings_count=0
        )
        
        assert record.date == "2026-02-08"
        assert record.nav == 1_000_000.0
        assert record.daily_ret == 0.0
        assert record.cum_ret == 0.0
        assert record.mdd == 0.0
        assert len(tracker.records) == 1
    
    def test_record_daily_with_gain(self, tracker):
        """测试盈利情况"""
        # 第一天: 无持仓
        tracker.record_daily(date="2026-02-07", holdings_value=0, holdings_count=0)
        
        # 第二天: 买入后持仓市值 52 万 (假设买入花了 50 万)
        tracker.cash = 500_000.0
        record = tracker.record_daily(
            date="2026-02-08",
            holdings_value=520_000.0,
            holdings_count=10
        )
        
        assert record.nav == 1_020_000.0
        assert record.daily_ret == pytest.approx(0.02, rel=1e-4)  # 2% 收益
        assert record.cum_ret == pytest.approx(0.02, rel=1e-4)
        assert record.mdd == 0.0  # 还没回撤
    
    def test_record_daily_with_drawdown(self, tracker):
        """测试回撤情况"""
        # 第一天: 净值 102 万
        tracker.cash = 500_000.0
        tracker.record_daily(date="2026-02-07", holdings_value=520_000.0, holdings_count=10)
        
        # 第二天: 持仓亏损，市值变成 48 万
        record = tracker.record_daily(
            date="2026-02-08",
            holdings_value=480_000.0,
            holdings_count=10
        )
        
        assert record.nav == 980_000.0
        assert record.mdd > 0  # 应该有回撤
        # MDD = (1020000 - 980000) / 1020000 = 0.0392
        assert record.mdd == pytest.approx(0.0392, rel=1e-2)
    
    def test_adjust_cash(self, tracker):
        """测试现金调整"""
        assert tracker.cash == 1_000_000.0
        
        # 买入花费 50 万
        tracker.adjust_cash(-500_000.0)
        assert tracker.cash == 500_000.0
        
        # 卖出收入 20 万
        tracker.adjust_cash(200_000.0)
        assert tracker.cash == 700_000.0
    
    def test_get_max_drawdown(self, tracker):
        """测试最大回撤获取"""
        # 模拟一系列记录
        tracker.cash = 500_000.0
        tracker.record_daily(date="2026-02-05", holdings_value=500_000.0, holdings_count=10)  # NAV=100万
        tracker.record_daily(date="2026-02-06", holdings_value=550_000.0, holdings_count=10)  # NAV=105万
        tracker.record_daily(date="2026-02-07", holdings_value=450_000.0, holdings_count=10)  # NAV=95万
        tracker.record_daily(date="2026-02-08", holdings_value=480_000.0, holdings_count=10)  # NAV=98万
        
        max_mdd = tracker.get_max_drawdown()
        # 峰值 105 万，最低 95 万，MDD = 10/105 ≈ 0.095
        assert max_mdd == pytest.approx(0.095, rel=1e-2)
    
    def test_persistence(self, temp_path):
        """测试持久化"""
        # 创建并记录
        tracker1 = NavTracker(initial_capital=1_000_000.0, state_path=temp_path)
        tracker1.cash = 500_000.0
        tracker1.record_daily(date="2026-02-08", holdings_value=520_000.0, holdings_count=10)
        
        # 新实例加载
        tracker2 = NavTracker(initial_capital=1_000_000.0, state_path=temp_path)
        
        assert tracker2.cash == 500_000.0
        assert len(tracker2.records) == 1
        assert tracker2.records[0].nav == 1_020_000.0
    
    def test_same_day_update(self, tracker):
        """测试同一天重复记录应覆盖"""
        tracker.record_daily(date="2026-02-08", holdings_value=500_000.0, holdings_count=10)
        tracker.record_daily(date="2026-02-08", holdings_value=550_000.0, holdings_count=10)
        
        assert len(tracker.records) == 1
        assert tracker.records[0].holdings_value == 550_000.0
    
    def test_summary(self, tracker):
        """测试摘要生成"""
        tracker.cash = 500_000.0
        tracker.record_daily(date="2026-02-08", holdings_value=520_000.0, holdings_count=10)
        
        summary = tracker.summary()
        assert "NAV Tracker Summary" in summary
        assert "1,020,000" in summary
    
    def test_reset(self, tracker):
        """测试重置"""
        tracker.cash = 500_000.0
        tracker.record_daily(date="2026-02-08", holdings_value=520_000.0, holdings_count=10)
        
        tracker.reset()
        
        assert tracker.cash == 1_000_000.0
        assert len(tracker.records) == 0
        assert tracker.peak_nav == 1_000_000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
