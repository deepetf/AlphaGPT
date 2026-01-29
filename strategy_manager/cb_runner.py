import os
import sys
import json
import torch
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add success level for compatibility
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kws)
logging.Logger.success = success

# 添加项目根目录到 Path (如果作为脚本直接运行)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from model_core.config import ModelConfig, RobustConfig
from model_core.data_loader import CBDataLoader
from model_core.vm import StackVM

# Strategy Components
from strategy_manager.cb_portfolio import CBPortfolioManager
from strategy_manager.rebalancer import CBRebalancer, AssetInfo
from execution.cb_trader import FileTrader, OrderSide

class CBStrategyRunner:
    """
    可转债 AlphaGPT 策略执行器
    
    功能:
    1. 加载最佳公式 (best_cb_formula.json)
    2. 加载全量历史数据 (CBDataLoader)
    3. 执行因子计算
    4. 生成实盘交易计划 (Trade Plan)
    5. 执行稳健性风控 (Active Ratio Check)
    """
    
    def __init__(self, strategy_path=None):
        if strategy_path is None:
            # 默认寻找 model_core 下的 best_cb_formula.json
            strategy_path = os.path.join(project_root, "model_core", "best_cb_formula.json")
            
        self.strategy_path = strategy_path
        self.formula = None
        # 优先使用配置中的 Top-K，否则默认 10
        self.top_k = getattr(RobustConfig, "TOP_K", 10)
        self.output_dir = os.path.join(project_root, "execution", "plans")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Components
        self.portfolio = CBPortfolioManager()
        self.rebalancer = CBRebalancer() # Default 100k capital, can appear from config later
        self.trader = FileTrader()
        
    def load_strategy(self):
        """加载策略公式"""
        if not os.path.exists(self.strategy_path):
            logger.error(f"Strategy file not found: {self.strategy_path}")
            return False
            
        try:
            with open(self.strategy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 兼容 V2 格式 (best -> formula)
                if "best" in data:
                    self.formula = data["best"]["formula"]
                    score = data["best"].get("score", 0)
                    logger.info(f"Loaded Best Formula (Score: {score:.2f})")
                else:
                    logger.error("Invalid strategy file format (missing 'best' key)")
                    return False
                    
            logger.info(f"Formula: {' '.join(self.formula) if isinstance(self.formula, list) else self.formula}")
            return True
        except Exception as e:
            logger.exception(f"Failed to load strategy: {e}")
            return False

    def run(self):
        """执行每日选股"""
        logger.info("🚀 Starting CB Strategy Runner...")
        
        # 1. 加载数据
        loader = CBDataLoader()
        logger.info("Loading Data...")
        loader.load_data()
        
        # 2. 获取最新日期
        latest_date_idx = -1
        latest_date = loader.dates_list[latest_date_idx]
        logger.info(f"Latest Data Date: {latest_date}")
        
        # 3. 风控检查: Active Ratio + Min Valid Count
        # 检查当日有效标的数量
        valid_mask = loader.valid_mask[latest_date_idx, :]
        valid_count = valid_mask.sum().item()
        total_assets = len(loader.assets_list)
        active_ratio = valid_count / total_assets if total_assets > 0 else 0
        
        logger.info(f"Valid Assets Today: {valid_count}/{total_assets} (Active Ratio: {active_ratio:.2%})")
        
        # 熔断条件 1: Active Ratio 低于配置阈值
        if active_ratio < RobustConfig.MIN_ACTIVE_RATIO:
            logger.critical(f"⛔ CIRCUIT BREAKER: Active Ratio too low ({active_ratio:.2%} < {RobustConfig.MIN_ACTIVE_RATIO:.0%}). Trading HALTED.")
            return
        
        # 熔断条件 2: 有效标的数量低于最小阈值
        min_required = max(RobustConfig.MIN_VALID_COUNT, self.top_k * 2)
        if valid_count < min_required:
            logger.critical(f"⛔ CIRCUIT BREAKER: Too few valid assets ({valid_count} < {min_required}). Trading HALTED.")
            return
        
        # 4. 执行因子计算
        logger.info("Executing Alpha Formula...")
        vm = StackVM()
        
        # 转为 CPU 执行推理
        feat_tensor = loader.feat_tensor.to('cpu')
        
        try:
            # [1, Time] -> [Assets, Time] ? No, execute returns [Time, Assets] usually?
            # features is [Assets, Features, Time]
            # StackVM execute returns [Time, Assets]
            factors = vm.execute(self.formula, feat_tensor) # -> [Time, Assets]
            
            if factors is None:
                logger.error("Formula execution failed (returned None).")
                return
                
        except Exception as e:
            logger.exception(f"Formula execution error: {e}")
            return
            
        # 5. 提取最新因子值 并 选股
        # factors: [Time, Assets]
        latest_factors = factors[latest_date_idx, :] # [Assets]
        
        # 应用有效性掩码
        # loader.valid_mask is [Time, Assets]
        today_mask = loader.valid_mask[latest_date_idx, :].to('cpu') # [Assets]
        
        # Mask invalids
        latest_factors[~today_mask] = -1e9
        
        # Top-K Slicing
        # values, indices
        top_k_vals, top_k_indices = torch.topk(latest_factors, k=self.top_k)
        
        selected_assets = []
        for rank, idx in enumerate(top_k_indices):
            idx = idx.item()
            val = top_k_vals[rank].item()
            
            # double check validity
            if val <= -1e8: 
                break # 不足 K 个有效
            
            code = loader.assets_list[idx]
            name = loader.names_dict.get(code, "Unknown")
            
            selected_assets.append({
                "rank": rank + 1,
                "code": code,
                "name": name,
                "score": round(val, 4)
            })
            
        # 6. 生成调仓计划
        logger.success(f"Selected {len(selected_assets)} assets for {latest_date}")
        
        # 优化: 建立 code -> index 映射以加速查找 (O(1))
        # 虽然 Top-K 较小，但对于后续查询持仓价格很有用
        code_to_idx = {code: i for i, code in enumerate(loader.assets_list)}
        
        target_info_list = []
        for asset in selected_assets:
            # 获取收盘价
            asset_idx = code_to_idx.get(asset['code'])
            price = 0.0
            if asset_idx is not None:
                price = loader.raw_data_cache['CLOSE'][latest_date_idx, asset_idx].item()
            else:
                logger.warning(f"Price not found for asset {asset['code']}")
            
            logger.info(f"  #{asset['rank']} {asset['code']} {asset['name']} (Score: {asset['score']} Price: {price:.2f})")
            
            target_info_list.append(AssetInfo(
                code=asset['code'],
                name=asset['name'],
                price=price,
                score=asset['score'],
                rank=asset['rank']
            ))
            
        # 获取当前持仓
        current_codes = self.portfolio.get_position_codes()
        current_holdings = {code: pos.shares for code, pos in self.portfolio.positions.items()}
        
        # 获取卖出标的价格 (用于生成卖单)
        sell_prices = {}
        for code in current_codes:
            # 只有当持仓标的不在 Top-K (target_info_list) 中时，才需要这里单独获取价格
            # 但为了简单，我们可以获取所有持仓标的的价格
            idx = code_to_idx.get(code)
            if idx is not None:
                sell_prices[code] = loader.raw_data_cache['CLOSE'][latest_date_idx, idx].item()
        
        # 计算订单
        orders = self.rebalancer.generate_orders(
            current_codes=current_codes,
            target_info=target_info_list,
            current_holdings=current_holdings,
            sell_prices=sell_prices
        )
        
        # 7. 提交订单 (生成 CSV)
        if orders:
            logger.info(f"Generated {len(orders)} orders")
            result = self.trader.submit_orders(orders, latest_date)
            if result.success:
                logger.success(f"Orders submitted: {result.message}")
            else:
                logger.error(f"Order submission failed: {result.message}")
        else:
            logger.info("No orders generated (Holdings match Target)")
            
        self.save_plan(latest_date, selected_assets, orders)
        
        # 可选: 模拟成交更新状态
        # self.simulate_execution(orders) # 默认关闭，需手动开启或配置

        
    def save_plan(self, date_str, assets, orders=None):
        """保存交易计划"""
        plan = {
            "date": date_str,
            "created_at": datetime.now().isoformat(),
            "strategy": "AlphaGPT_CB_Top10",
            "assets": assets,
            "orders": [o.to_csv_row() for o in orders] if orders else []
        }
        
        filename = f"plan_{date_str}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Plan saved to: {filepath}")
        
    def simulate_execution(self, orders: list, date: str):
        """
        模拟成交并更新持仓 (Closed Loop)
        
        Args:
            orders: 订单列表
            date: 成交日期
        """
        if not orders:
            return
            
        logger.info(f"Simulating execution for {len(orders)} orders on {date}...")
        
        for order in orders:
            # 假设全部以收盘价成交
            
            # 使用 Enum 值比较，更安全
            # 兼容 Enum 和 字符串 判断 (支持小写容错)
            # OrderSide 继承自 str, 但转为 str 并 upper() 可以兼容 "buy", "Buy" 等非标准输入
            side_str = str(order.side).upper()
            # 处理可能的 Enum 前缀 (如 "ORDERSIDE.BUY")
            if "." in side_str:
                side_str = side_str.split(".")[-1]
            
            is_buy = (side_str == "BUY")
            is_sell = (side_str == "SELL")
            
            if is_buy:
                # 检查是否已有持仓
                pos = self.portfolio.get_position(order.code)
                if pos:
                    # 加仓 (Update)
                    new_shares = pos.shares + order.quantity
                    self.portfolio.update_position(order.code, new_shares, order.price)
                else:
                    # 建仓 (Add)
                    self.portfolio.add_position(
                        code=order.code,
                        name=order.name,
                        shares=order.quantity,
                        price=order.price,
                        date=date
                    )
            elif is_sell:
                # 检查当前持仓，计算剩余数量
                pos = self.portfolio.get_position(order.code)
                if pos:
                    new_shares = max(0, pos.shares - order.quantity)
                    if new_shares > 0:
                        # 减仓
                        self.portfolio.update_position(order.code, new_shares, order.price)
                    else:
                        # 清仓
                        self.portfolio.remove_position(order.code)
                else:
                    logger.warning(f"Cannot sell {order.code}: Position not found")
            else:
                logger.warning(f"Skipping order with unknown side '{order.side}': {order}")
                
        self.portfolio.save_state()
        logger.success("Portfolio updated (Simulated)")


if __name__ == "__main__":
    runner = CBStrategyRunner()
    if runner.load_strategy():
        runner.run()
