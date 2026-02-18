"""
SimTrader 单元测试

测试模拟成交引擎的核心功能:
1. 买入执行
2. 卖出执行
3. 现金与持仓更新
"""
import pytest
import os
import tempfile

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.sim_trader import SimTrader, SimOrder, OrderSide, TradeRecord
from strategy_manager.cb_portfolio import CBPortfolioManager
from strategy_manager.nav_tracker import NavTracker


class TestSimTrader:
    """测试 SimTrader"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        import tempfile
        import shutil
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)
    
    @pytest.fixture
    def portfolio(self, temp_dir):
        """创建 Portfolio"""
        path = os.path.join(temp_dir, "portfolio.json")
        return CBPortfolioManager(state_path=path)
    
    @pytest.fixture
    def nav_tracker(self, temp_dir):
        """创建 NavTracker"""
        path = os.path.join(temp_dir, "nav.json")
        return NavTracker(initial_capital=1_000_000.0, state_path=path)
    
    @pytest.fixture
    def trader(self, portfolio, nav_tracker, temp_dir):
        """创建 SimTrader"""
        history_path = os.path.join(temp_dir, "test_history.json")
        return SimTrader(
            portfolio=portfolio, 
            nav_tracker=nav_tracker, 
            fee_rate=0.0005,
            history_path=history_path
        )
    
    def test_execute_buy_success(self, trader, nav_tracker, portfolio):
        """测试买入成功"""
        orders = [
            SimOrder(code="123001.SZ", name="测试转债", side=OrderSide.BUY, shares=100, target_price=100.0)
        ]
        prices = {"123001.SZ": 100.0}
        
        records = trader.execute(orders, prices, date="2026-02-08")
        
        assert len(records) == 1
        assert records[0].side == OrderSide.BUY
        assert records[0].shares == 100
        
        # 检查现金: 100 * 100 * (1 + 0.0005) = 10,005
        expected_cost = 100 * 100 * 1.0005
        assert nav_tracker.cash == pytest.approx(1_000_000 - expected_cost, rel=1e-4)
        
        # 检查持仓
        pos = portfolio.get_position("123001.SZ")
        assert pos is not None
        assert pos.shares == 100
    
    def test_execute_buy_insufficient_cash(self, trader, nav_tracker):
        """测试现金不足"""
        # 设置现金很少
        nav_tracker.cash = 1000.0
        
        orders = [
            SimOrder(code="123001.SZ", name="测试转债", side=OrderSide.BUY, shares=100, target_price=100.0)
        ]
        prices = {"123001.SZ": 100.0}
        
        records = trader.execute(orders, prices, date="2026-02-08")
        
        assert len(records) == 0  # 买入失败
    
    def test_execute_sell_success(self, trader, nav_tracker, portfolio):
        """测试卖出成功"""
        # 先建立持仓
        portfolio.add_position("123001.SZ", "测试转债", 100, 100.0, "2026-02-07")
        nav_tracker.cash = 500_000.0  # 模拟买入后剩余现金
        
        orders = [
            SimOrder(code="123001.SZ", name="测试转债", side=OrderSide.SELL, shares=50, target_price=110.0)
        ]
        prices = {"123001.SZ": 110.0}
        
        records = trader.execute(orders, prices, date="2026-02-08")
        
        assert len(records) == 1
        assert records[0].side == OrderSide.SELL
        assert records[0].shares == 50
        
        # 检查现金: 50 * 110 * (1 - 0.0005) = 5,497.25
        expected_proceeds = 50 * 110 * 0.9995
        assert nav_tracker.cash == pytest.approx(500_000 + expected_proceeds, rel=1e-4)
        
        # 检查持仓
        pos = portfolio.get_position("123001.SZ")
        assert pos is not None
        assert pos.shares == 50
    
    def test_execute_sell_full_position(self, trader, portfolio):
        """测试全部卖出"""
        portfolio.add_position("123001.SZ", "测试转债", 100, 100.0, "2026-02-07")
        
        orders = [
            SimOrder(code="123001.SZ", name="测试转债", side=OrderSide.SELL, shares=100, target_price=110.0)
        ]
        prices = {"123001.SZ": 110.0}
        
        trader.execute(orders, prices, date="2026-02-08")
        
        # 持仓应该被移除
        assert portfolio.get_position("123001.SZ") is None
    
    def test_execute_sell_no_position(self, trader):
        """测试卖出无持仓"""
        orders = [
            SimOrder(code="123001.SZ", name="测试转债", side=OrderSide.SELL, shares=100, target_price=110.0)
        ]
        prices = {"123001.SZ": 110.0}
        
        records = trader.execute(orders, prices, date="2026-02-08")
        
        assert len(records) == 0  # 卖出失败
    
    def test_execute_multiple_orders(self, trader, nav_tracker, portfolio):
        """测试多笔订单"""
        orders = [
            SimOrder(code="123001.SZ", name="转债A", side=OrderSide.BUY, shares=50, target_price=100.0),
            SimOrder(code="127050.SZ", name="转债B", side=OrderSide.BUY, shares=50, target_price=120.0),
        ]
        prices = {"123001.SZ": 100.0, "127050.SZ": 120.0}
        
        records = trader.execute(orders, prices, date="2026-02-08")
        
        assert len(records) == 2
        assert portfolio.get_holdings_count() == 2
    
    def test_add_to_existing_position(self, trader, portfolio):
        """测试加仓"""
        # 先建仓
        portfolio.add_position("123001.SZ", "测试转债", 100, 100.0, "2026-02-07")
        
        orders = [
            SimOrder(code="123001.SZ", name="测试转债", side=OrderSide.BUY, shares=50, target_price=110.0)
        ]
        prices = {"123001.SZ": 110.0}
        
        trader.execute(orders, prices, date="2026-02-08")
        
        pos = portfolio.get_position("123001.SZ")
        assert pos.shares == 150  # 100 + 50
    
    def test_get_trade_count(self, trader, nav_tracker):
        """测试成交计数"""
        assert trader.get_trade_count() == 0
        
        orders = [
            SimOrder(code="123001.SZ", name="转债A", side=OrderSide.BUY, shares=10, target_price=100.0),
        ]
        trader.execute(orders, {"123001.SZ": 100.0}, date="2026-02-08")
        
        assert trader.get_trade_count() == 1

    def test_trade_history_persistence(self, temp_dir, portfolio, nav_tracker):
        """测试交易纪录持久化"""
        history_path = os.path.join(temp_dir, "history.json")
        trader1 = SimTrader(portfolio=portfolio, nav_tracker=nav_tracker, history_path=history_path)
        
        orders = [
            SimOrder(code="123001.SZ", name="转债A", side=OrderSide.BUY, shares=10, target_price=100.0),
        ]
        trader1.execute(orders, {"123001.SZ": 100.0}, date="2026-02-08")
        
        # 新实例加载
        trader2 = SimTrader(portfolio=portfolio, nav_tracker=nav_tracker, history_path=history_path)
        assert len(trader2.trade_history) == 1
        assert trader2.trade_history[0].code == "123001.SZ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
