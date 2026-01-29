"""
CB Strategy 单元测试

运行: python -m pytest tests/test_cb_strategy.py -v
"""
import os
import sys
import json
import csv
import tempfile
import shutil
import pytest

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_manager.cb_portfolio import CBPosition, CBPortfolioManager
from execution.cb_trader import Order, OrderSide, OrderResult, FileTrader
from strategy_manager.rebalancer import compute_rebalance, RebalanceResult, AssetInfo, CBRebalancer


class TestCBPosition:
    """测试 CBPosition 数据类"""
    
    def test_position_creation(self):
        """测试持仓创建"""
        pos = CBPosition(
            code="113025.SH",
            name="包钢转债",
            shares=100,
            avg_cost=105.0,
            last_price=110.0,
            entry_date="2024-01-15"
        )
        assert pos.code == "113025.SH"
        assert pos.name == "包钢转债"
        assert pos.shares == 100
        assert pos.avg_cost == 105.0
        assert pos.last_price == 110.0
    
    def test_market_value(self):
        """测试市值计算"""
        pos = CBPosition(
            code="113025.SH",
            name="包钢转债",
            shares=100,
            avg_cost=100.0,
            last_price=110.0,
            entry_date="2024-01-15"
        )
        # 100张 * 110元 = 11000元
        assert pos.market_value == 11000.0
    
    def test_pnl_calculation(self):
        """测试盈亏计算"""
        pos = CBPosition(
            code="113025.SH",
            name="包钢转债",
            shares=100,
            avg_cost=100.0,
            last_price=110.0,
            entry_date="2024-01-15"
        )
        # 成本: 100 * 100 = 10000
        # 市值: 100 * 110 = 11000
        # 盈亏: 1000
        assert pos.cost_value == 10000.0
        assert pos.pnl == 1000.0
        assert pos.pnl_pct == 0.1  # 10%
    
    def test_pnl_loss(self):
        """测试亏损情况"""
        pos = CBPosition(
            code="113025.SH",
            name="包钢转债",
            shares=100,
            avg_cost=110.0,
            last_price=100.0,
            entry_date="2024-01-15"
        )
        assert pos.pnl == -1000.0
        assert pos.pnl_pct == pytest.approx(-0.0909, rel=0.01)


class TestCBPortfolioManager:
    """测试 CBPortfolioManager"""
    
    @pytest.fixture
    def temp_state_file(self):
        """创建临时状态文件"""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield path
        # 清理
        if os.path.exists(path):
            os.remove(path)
    
    def test_empty_portfolio(self, temp_state_file):
        """测试空组合"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        assert pm.get_holdings_count() == 0
        assert pm.get_position_codes() == []
        assert pm.get_holdings_value() == 0.0
    
    def test_add_position(self, temp_state_file):
        """测试添加持仓"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        pos = pm.add_position(
            code="113025.SH",
            name="包钢转债",
            shares=100,
            price=105.0,
            date="2024-01-15"
        )
        assert pos is not None
        assert pos.code == "113025.SH"
        assert pm.get_holdings_count() == 1
        assert "113025.SH" in pm.get_position_codes()
    
    def test_get_position(self, temp_state_file):
        """测试获取持仓"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        pm.add_position("113025.SH", "包钢转债", 100, 105.0, "2024-01-15")
        
        pos = pm.get_position("113025.SH")
        assert pos is not None
        assert pos.name == "包钢转债"
        
        # 不存在的持仓
        assert pm.get_position("999999.SH") is None
    
    def test_update_position(self, temp_state_file):
        """测试更新持仓"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        pm.add_position("113025.SH", "包钢转债", 100, 100.0, "2024-01-15")
        
        # 更新持仓数量和价格
        pos = pm.update_position("113025.SH", shares=150, price=110.0)
        assert pos is not None
        assert pos.shares == 150
        assert pos.last_price == 110.0
    
    def test_update_price_only(self, temp_state_file):
        """测试仅更新价格"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        pm.add_position("113025.SH", "包钢转债", 100, 100.0, "2024-01-15")
        
        pos = pm.update_price("113025.SH", 115.0)
        assert pos is not None
        assert pos.last_price == 115.0
        assert pos.shares == 100  # 数量不变
    
    def test_remove_position(self, temp_state_file):
        """测试移除持仓"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        pm.add_position("113025.SH", "包钢转债", 100, 100.0, "2024-01-15")
        
        removed = pm.remove_position("113025.SH")
        assert removed is not None
        assert pm.get_holdings_count() == 0
        
        # 再次移除应返回 None
        assert pm.remove_position("113025.SH") is None
    
    def test_clear_all(self, temp_state_file):
        """测试清空所有持仓"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        pm.add_position("113025.SH", "包钢转债", 100, 100.0, "2024-01-15")
        pm.add_position("127050.SZ", "蓝帆转债", 50, 120.0, "2024-01-16")
        
        assert pm.get_holdings_count() == 2
        pm.clear_all()
        assert pm.get_holdings_count() == 0
    
    def test_persistence(self, temp_state_file):
        """测试持久化"""
        # 创建并保存
        pm1 = CBPortfolioManager(state_path=temp_state_file)
        pm1.add_position("113025.SH", "包钢转债", 100, 105.0, "2024-01-15")
        pm1.add_position("127050.SZ", "蓝帆转债", 50, 120.0, "2024-01-16")
        
        # 重新加载
        pm2 = CBPortfolioManager(state_path=temp_state_file)
        assert pm2.get_holdings_count() == 2
        
        pos = pm2.get_position("113025.SH")
        assert pos is not None
        assert pos.name == "包钢转债"
        assert pos.shares == 100
    
    def test_holdings_value(self, temp_state_file):
        """测试总市值计算"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        pm.add_position("113025.SH", "包钢转债", 100, 100.0, "2024-01-15")
        pm.add_position("127050.SZ", "蓝帆转债", 50, 120.0, "2024-01-16")
        
        # 100 * 100 + 50 * 120 = 10000 + 6000 = 16000
        assert pm.get_holdings_value() == 16000.0
    
    def test_state_file_format(self, temp_state_file):
        """测试状态文件格式"""
        pm = CBPortfolioManager(state_path=temp_state_file)
        pm.add_position("113025.SH", "包钢转债", 100, 105.0, "2024-01-15")
        
        # 检查文件内容
        with open(temp_state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "version" in data
        assert "updated_at" in data
        assert "positions" in data
        assert data["version"] == "1.0"
        assert "113025.SH" in data["positions"]


class TestOrder:
    """测试 Order 数据类"""
    
    def test_order_creation(self):
        """测试订单创建"""
        order = Order(
            code="113025.SH",
            name="包钢转债",
            side=OrderSide.BUY,
            quantity=100,
            price=105.0,
            reason="Top-K入选",
            rank=1
        )
        assert order.code == "113025.SH"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.rank == 1
    
    def test_order_to_csv_row(self):
        """测试订单转 CSV 行"""
        order = Order(
            code="113025.SH",
            name="包钢转债",
            side=OrderSide.SELL,
            quantity=50,
            price=110.0,
            reason="剔除出Top-K",
            rank=0
        )
        row = order.to_csv_row()
        assert row["code"] == "113025.SH"
        assert row["side"] == "SELL"
        assert row["quantity"] == 50
        assert row["reason"] == "剔除出Top-K"


class TestFileTrader:
    """测试 FileTrader"""
    
    @pytest.fixture
    def temp_orders_dir(self):
        """创建临时订单目录"""
        path = tempfile.mkdtemp()
        yield path
        # 清理
        shutil.rmtree(path, ignore_errors=True)
    
    def test_submit_empty_orders(self, temp_orders_dir):
        """测试提交空订单"""
        trader = FileTrader(output_dir=temp_orders_dir)
        result = trader.submit_orders([], "2024-01-15")
        assert result.success
        assert result.submitted_count == 0
        assert result.message == "No orders to submit"
    
    def test_submit_orders(self, temp_orders_dir):
        """测试提交订单"""
        trader = FileTrader(output_dir=temp_orders_dir)
        orders = [
            Order("113025.SH", "包钢转债", OrderSide.BUY, 100, 105.0, "Top-K入选", 1),
            Order("127050.SZ", "蓝帆转债", OrderSide.SELL, 50, 120.0, "剔除Top-K", 0),
        ]
        result = trader.submit_orders(orders, "2024-01-15")
        
        assert result.success
        assert result.submitted_count == 2
        assert result.failed_count == 0
    
    def test_csv_file_created(self, temp_orders_dir):
        """测试 CSV 文件创建"""
        trader = FileTrader(output_dir=temp_orders_dir)
        orders = [
            Order("113025.SH", "包钢转债", OrderSide.BUY, 100, 105.0, "Top-K入选", 1),
        ]
        trader.submit_orders(orders, "2024-01-15")
        
        # 检查文件存在
        filepath = os.path.join(temp_orders_dir, "orders_2024-01-15.csv")
        assert os.path.exists(filepath)
    
    def test_csv_file_format(self, temp_orders_dir):
        """测试 CSV 文件格式"""
        trader = FileTrader(output_dir=temp_orders_dir)
        orders = [
            Order("113025.SH", "包钢转债", OrderSide.BUY, 100, 105.0, "Top-K入选", 1),
        ]
        trader.submit_orders(orders, "2024-01-15")
        
        # 读取并检查格式
        filepath = os.path.join(temp_orders_dir, "orders_2024-01-15.csv")
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert rows[0]["date"] == "2024-01-15"
        assert rows[0]["code"] == "113025.SH"
        assert rows[0]["side"] == "BUY"
        assert rows[0]["quantity"] == "100"
        assert rows[0]["rank"] == "1"
    
    def test_read_orders(self, temp_orders_dir):
        """测试读取订单"""
        trader = FileTrader(output_dir=temp_orders_dir)
        orders = [
            Order("113025.SH", "包钢转债", OrderSide.BUY, 100, 105.0, "Top-K入选", 1),
            Order("127050.SZ", "蓝帆转债", OrderSide.SELL, 50, 120.0, "剔除Top-K", 0),
        ]
        trader.submit_orders(orders, "2024-01-15")
        
        # 读取回来
        read_orders = trader.read_orders("2024-01-15")
        assert len(read_orders) == 2
        assert read_orders[0].code == "113025.SH"
        assert read_orders[0].side == OrderSide.BUY
        assert read_orders[1].side == OrderSide.SELL
    
    def test_read_nonexistent_orders(self, temp_orders_dir):
        """测试读取不存在的订单"""
        trader = FileTrader(output_dir=temp_orders_dir)
        orders = trader.read_orders("1999-01-01")
        assert orders == []
    
    def test_get_order_files(self, temp_orders_dir):
        """测试获取订单文件列表"""
        trader = FileTrader(output_dir=temp_orders_dir)
        trader.submit_orders([Order("113025.SH", "包钢转债", OrderSide.BUY, 100, 105.0, "", 1)], "2024-01-15")
        trader.submit_orders([Order("113025.SH", "包钢转债", OrderSide.SELL, 100, 110.0, "", 0)], "2024-01-16")
        
        files = trader.get_order_files()
        assert len(files) == 2
        assert "orders_2024-01-15.csv" in files
        assert "orders_2024-01-16.csv" in files


class TestComputeRebalance:
    """测试 compute_rebalance 纯函数"""
    
    def test_no_change(self):
        """测试无变化情况"""
        current = ["A", "B", "C"]
        target = ["A", "B", "C"]
        sell, buy = compute_rebalance(current, target)
        assert sell == []
        assert buy == []
    
    def test_all_sell(self):
        """测试全部卖出"""
        current = ["A", "B", "C"]
        target = []
        sell, buy = compute_rebalance(current, target)
        assert set(sell) == {"A", "B", "C"}
        assert buy == []
    
    def test_all_buy(self):
        """测试全部买入"""
        current = []
        target = ["X", "Y", "Z"]
        sell, buy = compute_rebalance(current, target)
        assert sell == []
        assert set(buy) == {"X", "Y", "Z"}
    
    def test_partial_change(self):
        """测试部分调仓"""
        current = ["A", "B", "C"]
        target = ["B", "C", "D"]
        sell, buy = compute_rebalance(current, target)
        assert sell == ["A"]
        assert buy == ["D"]
    
    def test_complete_rotation(self):
        """测试完全轮动"""
        current = ["A", "B", "C"]
        target = ["X", "Y", "Z"]
        sell, buy = compute_rebalance(current, target)
        assert set(sell) == {"A", "B", "C"}
        assert set(buy) == {"X", "Y", "Z"}


class TestCBRebalancer:
    """测试 CBRebalancer"""
    
    def test_compute_no_change(self):
        """测试无变化计算"""
        rebalancer = CBRebalancer()
        result = rebalancer.compute(["A", "B"], ["A", "B"])
        assert result.total_changes == 0
        assert result.has_changes == False
    
    def test_compute_with_changes(self):
        """测试有变化计算"""
        rebalancer = CBRebalancer()
        result = rebalancer.compute(["A", "B", "C"], ["B", "C", "D"])
        assert result.sell_codes == ["A"]
        assert result.buy_codes == ["D"]
        assert result.hold_codes == ["B", "C"] or set(result.hold_codes) == {"B", "C"}
        assert result.total_changes == 2
    
    def test_generate_orders_buy_only(self):
        """测试生成买入订单"""
        rebalancer = CBRebalancer(total_capital=100000)
        target_info = [
            AssetInfo(code="A", name="债A", price=100.0, score=1.0, rank=1),
            AssetInfo(code="B", name="债B", price=100.0, score=0.9, rank=2),
        ]
        orders = rebalancer.generate_orders([], target_info)
        
        assert len(orders) == 2
        assert all(o.side == OrderSide.BUY for o in orders)
    
    def test_generate_orders_sell_only(self):
        """测试生成卖出订单"""
        rebalancer = CBRebalancer()
        target_info = []
        current_holdings = {"A": 100, "B": 50}
        
        orders = rebalancer.generate_orders(["A", "B"], target_info, current_holdings)
        
        assert len(orders) == 2
        assert all(o.side == OrderSide.SELL for o in orders)
        # 检查数量
        order_a = next(o for o in orders if o.code == "A")
        assert order_a.quantity == 100
    
    def test_generate_orders_mixed(self):
        """测试混合订单"""
        rebalancer = CBRebalancer(total_capital=100000)
        target_info = [
            AssetInfo(code="B", name="债B", price=100.0, score=1.0, rank=1),
            AssetInfo(code="C", name="债C", price=100.0, score=0.9, rank=2),
        ]
        current_holdings = {"A": 100, "B": 50}
        
        orders = rebalancer.generate_orders(["A", "B"], target_info, current_holdings)
        
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        
        assert len(sell_orders) == 1
        assert sell_orders[0].code == "A"
        assert len(buy_orders) == 1
        assert buy_orders[0].code == "C"



from unittest.mock import MagicMock, patch
from model_core.config import RobustConfig

class TestCBStrategyRunner:
    """测试 CBStrategyRunner (集成测试)"""
    
    @patch("strategy_manager.cb_runner.CBDataLoader")
    @patch("strategy_manager.cb_runner.StackVM")
    def test_active_ratio_circuit_breaker(self, mock_vm, mock_loader):
        """测试 Active Ratio 熔断"""
        from strategy_manager.cb_runner import CBStrategyRunner
        import torch
        
        # Mock Loader
        loader_instance = mock_loader.return_value
        loader_instance.dates_list = ["2024-01-15"]
        loader_instance.assets_list = ["A", "B", "C", "D"]
        
        # Mock Valid Mask [Time=1, Assets=4]
        # Only 1 valid asset -> 25% < 50%
        loader_instance.valid_mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
        
        # Run
        runner = CBStrategyRunner(strategy_path="dummy.json")
        runner.load_strategy = MagicMock(return_value=True)
        runner.formula = ["CLOSE"]
        
        # Capture logs
        with patch("strategy_manager.cb_runner.logger") as mock_logger:
            runner.run()
            
            assert mock_logger.critical.called
            args, _ = mock_logger.critical.call_args
            assert "CIRCUIT BREAKER" in args[0]
            assert "Active Ratio" in args[0]
            
    def test_save_plan_schema(self):
        """测试 plan.json 输出格式"""
        from strategy_manager.cb_runner import CBStrategyRunner
        from execution.cb_trader import Order, OrderSide
        
        runner = CBStrategyRunner(strategy_path="dummy.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.output_dir = tmpdir
            
            assets = [{"code": "A", "rank": 1, "score": 1.0}]
            orders = [Order("A", "NameA", OrderSide.BUY, 10, 100.0, "reason", 1)]
            
            runner.save_plan("2024-01-01", assets, orders)
            
            filepath = os.path.join(tmpdir, "plan_2024-01-01.json")
            assert os.path.exists(filepath)
            
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            assert data["date"] == "2024-01-01"
            assert len(data["assets"]) == 1
            assert len(data["orders"]) == 1
            
    @patch("strategy_manager.cb_runner.CBDataLoader")
    @patch("strategy_manager.cb_runner.StackVM")
    def test_run_success_flow(self, mock_vm, mock_loader):
        """测试完整运行流程 (Mock 数据)"""
        from strategy_manager.cb_runner import CBStrategyRunner
        import torch
        
        # 1. Mock Data Setup
        loader = mock_loader.return_value
        loader.dates_list = ["2024-01-01"]
        loader.assets_list = ["A", "B"]
        loader.names_dict = {"A": "债A", "B": "债B"}
        
        # Mask [Time=1, Assets=2]
        loader.valid_mask = torch.ones((1, 2), dtype=torch.bool)
        
        # Raw Prices [Time=1, Assets=2]
        loader.raw_data_cache = {
            'CLOSE': torch.tensor([[100.0, 100.0]])
        }
        
        # Feat Tensor
        loader.feat_tensor = MagicMock()
        loader.feat_tensor.to.return_value = "dummy"
        
        # 2. Mock VM
        vm = mock_vm.return_value
        vm.execute.return_value = torch.tensor([[1.0, 0.9]])
        
        # 3. Setup Runner
        runner = CBStrategyRunner(strategy_path="dummy.json")
        runner.load_strategy = MagicMock(return_value=True)
        runner.formula = ["CLOSE"]
        runner.top_k = 1  # Logic: min_required = max(0, 1*2) = 2. valid=2. 2 < 2 False. Pass.
        
        # Mock Components
        runner.portfolio = MagicMock()
        runner.portfolio.get_position_codes.return_value = []
        runner.portfolio.positions = {}
        
        runner.rebalancer = MagicMock()
        from execution.cb_trader import Order, OrderSide
        runner.rebalancer.generate_orders.return_value = [
            Order("A", "债A", OrderSide.BUY, 10, 100.0, "Top-K", 1)
        ]
        
        runner.trader = MagicMock()
        runner.trader.submit_orders.return_value.success = True
        runner.trader.submit_orders.return_value.message = "OK"
        
        runner.save_plan = MagicMock()
        
        # 4. Run with Config Patch to pass Circuit Breaker
        with patch("strategy_manager.cb_runner.RobustConfig") as mock_config:
            mock_config.MIN_VALID_COUNT = 0
            mock_config.MIN_ACTIVE_RATIO = 0.0
            
            with patch("strategy_manager.cb_runner.logger"):
                runner.run()
        
        # 5. Verify
        runner.rebalancer.generate_orders.assert_called_once()
        runner.trader.submit_orders.assert_called_once()
        runner.save_plan.assert_called_once()
            


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



