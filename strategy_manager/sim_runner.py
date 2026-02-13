"""
SimulationRunner - 模拟盘主控逻辑

每日 14:50 运行，执行以下流程：
1. 获取实时行情和 CB 特性数据
2. 构建因子张量并计算因子值
3. 检测止盈信号
4. 筛选 Top-K 标的
5. 生成调仓订单并执行
6. 记录每日净值
"""
import logging
import json
import os
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from model_core.vm import StackVM
from model_core.config import ModelConfig, RobustConfig
from data_pipeline.realtime_provider import RealtimeDataProvider
from strategy_manager.cb_portfolio import CBPortfolioManager
from strategy_manager.nav_tracker import NavTracker
from strategy_manager.rebalancer import CBRebalancer, AssetInfo
from execution.sim_trader import SimTrader, SimOrder, OrderSide
from execution.cb_trader import OrderSide as CBOrderSide

# 延迟导入避免循环引用
StrategyConfig = None

logger = logging.getLogger(__name__)


def _get_strategy_config_class():
    """延迟导入 StrategyConfig"""
    global StrategyConfig
    if StrategyConfig is None:
        from strategy_manager.strategy_config import StrategyConfig as SC
        StrategyConfig = SC
    return StrategyConfig


class SimulationRunner:
    """
    模拟盘运行器
    
    职责:
    1. 每日调度数据获取和因子计算
    2. 执行止盈逻辑
    3. Top-K 选股与调仓
    4. 记录净值变化
    
    支持两种初始化方式:
    1. 传统模式: 传入独立的 portfolio, nav_tracker, formula_path 等参数
    2. 策略模式: 传入 StrategyConfig，自动创建隔离的组件实例
    """
    
    # 默认公式文件 (现已支持 TS_* 时序算子)
    DEFAULT_FORMULA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model_core", "best_cb_formula.json"
    )
    
    PORTFOLIO_BASE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "execution", "portfolio"
    )
    
    def __init__(
        self,
        data_provider: RealtimeDataProvider,
        portfolio: Optional[CBPortfolioManager] = None,
        nav_tracker: Optional[NavTracker] = None,
        formula_path: Optional[str] = None,
        top_k: int = 10,
        take_profit_ratio: float = 0.0,
        strategy_config = None,  # StrategyConfig instance
        replay_strict: bool = False,  # strict replay alignment mode
        replay_source: str = "sql_eod",  # strict replay backend
    ):
        """
        初始化模拟盘运行器
        
        Args:
            data_provider: 实时数据提供者
            portfolio: 组合管理器 (传统模式)
            nav_tracker: 净值追踪器 (传统模式)
            formula_path: 因子公式路径 (传统模式)
            top_k: 每日持仓数量 (传统模式)
            take_profit_ratio: 止盈比例 (传统模式)
            strategy_config: 策略配置 (策略模式，优先级高于传统参数)
            replay_source: 严格回放数据源 (sql_eod/parquet)
        """
        self.data_provider = data_provider
        
        if strategy_config is not None:
            # 策略模式: 从 StrategyConfig 初始化
            self._init_from_strategy_config(strategy_config)
        else:
            # 传统模式: 使用传入的参数
            self._init_legacy(
                portfolio,
                nav_tracker,
                formula_path,
                top_k,
                take_profit_ratio,
                replay_strict,
                replay_source,
            )
        
        # 初始化虚拟机
        self.vm = StackVM()

        # strict replay caches (used only in historical replay mode)
        self._bt_loader = None
        self._bt_factors = None
        self._bt_code_to_idx = {}

        # Reuse shared rebalance logic to align with CBStrategyRunner/verify flow.
        self.rebalancer = CBRebalancer(total_capital=self.nav_tracker.initial_capital)
    
    def _init_from_strategy_config(self, strategy_config):
        """从 StrategyConfig 初始化 (策略模式)"""
        self.strategy_id = strategy_config.id
        self.strategy_name = strategy_config.name
        
        # 创建策略专属的数据目录
        strategy_dir = os.path.join(self.PORTFOLIO_BASE_DIR, self.strategy_id)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # 初始化组合管理器
        holdings_path = os.path.join(strategy_dir, "holdings.json")
        self.portfolio = CBPortfolioManager(state_path=holdings_path)
        
        # 初始化净值追踪器
        nav_path = os.path.join(strategy_dir, "nav_history.json")
        self.nav_tracker = NavTracker(
            initial_capital=strategy_config.params.initial_capital,
            state_path=nav_path
        )
        
        # 初始化模拟交易器
        history_path = os.path.join(strategy_dir, "trade_history.json")
        self.trader = SimTrader(
            portfolio=self.portfolio,
            nav_tracker=self.nav_tracker,
            fee_rate=strategy_config.params.fee_rate,
            history_path=history_path
        )
        
        # 加载公式
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.formula = strategy_config.get_formula(base_dir)
        
        # 校验公式 (检测 TS_ 时序算子)
        self._validate_formula()
        
        # 策略参数
        self.top_k = strategy_config.params.top_k
        self.take_profit_ratio = strategy_config.params.take_profit_ratio
        self.replay_strict = getattr(strategy_config.params, "replay_strict", False)
        self.replay_source = getattr(strategy_config.params, "replay_source", "sql_eod")
        
        logger.info(
            f"[{self.strategy_id}] 绛栫暐鍒濆鍖栧畬鎴? "
            f"top_k={self.top_k}, tp={self.take_profit_ratio}, "
            f"replay_strict={self.replay_strict}, replay_source={self.replay_source}"
        )

    
    def _init_legacy(
        self,
        portfolio,
        nav_tracker,
        formula_path,
        top_k,
        take_profit_ratio,
        replay_strict: bool,
        replay_source: str,
    ):
        """传统模式初始化"""
        self.strategy_id = "default"
        self.strategy_name = "Default Strategy"
        
        self.portfolio = portfolio
        self.nav_tracker = nav_tracker
        self.top_k = top_k
        self.take_profit_ratio = take_profit_ratio
        self.replay_strict = replay_strict
        self.replay_source = replay_source
        
        # 加载因子公式
        self.formula_path = formula_path or self.DEFAULT_FORMULA_PATH
        self.formula = self._load_formula()
        
        # 校验公式 (检测 TS_ 时序算子)
        self._validate_formula()
        
        # 初始化模拟交易器
        self.trader = SimTrader(
            portfolio=portfolio, 
            nav_tracker=nav_tracker,
            fee_rate=RobustConfig.FEE_RATE
        )
    
    def _load_formula(self) -> List[str]:
        """加载因子公式 (传统模式使用)"""
        if not os.path.exists(self.formula_path):
            raise FileNotFoundError(f"公式文件不存在: {self.formula_path}")
        
        with open(self.formula_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formula = data.get('best', {}).get('formula', [])
        logger.info(f"加载公式: {' '.join(formula)}")
        return formula
    
    def _get_required_window(self) -> int:
        """
        根据公式中的 TS 算子确定所需的历史窗口大小
        
        Returns:
            1: 纯横截面公式 (无 TS_* 算子)
            2: 包含 TS_DELAY/TS_DELTA/TS_RET
            5: 包含 TS_MEAN5/TS_STD5/TS_BIAS5
        """
        ts_5day = ['TS_MEAN5', 'TS_STD5', 'TS_BIAS5']
        ts_2day = ['TS_DELAY', 'TS_DELTA', 'TS_RET']
        
        if any(op in self.formula for op in ts_5day):
            return 5
        elif any(op in self.formula for op in ts_2day):
            return 2
        else:
            return 1
    
    def _validate_formula(self):
        """
        校验公式并记录窗口需求
        """
        self.required_window = self._get_required_window()
        if self.required_window > 1:
            ts_ops = [op for op in self.formula if op.startswith('TS_')]
            logger.info(
                f"[{self.strategy_id}] 公式包含时序算子: {ts_ops}, "
                f"将使用 {self.required_window} 天历史窗口"
            )
        else:
            logger.info(f"[{self.strategy_id}] 纯横截面公式，使用单日数据")


    
    def run_daily(self, date: str) -> Dict:
        """
        执行每日模拟
        
        Args:
            date: 日期 (YYYY-MM-DD)
            
        Returns:
            运行结果摘要
        """
        logger.info(f"=== 模拟盘运行 {date} ===")
        
        # 判断是否为实盘模式 (今天) 或回放模式 (历史日期)
        today_str = datetime.now().strftime('%Y-%m-%d')
        is_live_mode = (date == today_str)
        
        if is_live_mode:
            logger.info("[实盘模式] 将获取 Mini QMT 实时行情")
        else:
            logger.info(f"[回放模式] 使用 SQL 历史数据，不调用实时行情接口")
        
        # 1. 获取基础特征
        if self.replay_strict and not is_live_mode:
            logger.info(
                f"[{self.strategy_id}] strict replay mode enabled: source={self.replay_source}"
            )
            return self._run_daily_replay_strict(date)
        
        cb_features = self.data_provider.get_cb_features(date)
        if cb_features.empty:
            logger.warning(f"无法获取 {date} 的 SQL 基础数据")
            return {"status": "no_data"}
            
        code_list = cb_features['code'].tolist()
        
        # 2. 根据模式决定是否获取实时行情
        if is_live_mode:
            # 建立实时推送连接 (Mini QMT 推荐用法)
            self.data_provider.subscribe_quotes(code_list)
            # 获取实时快照 (使用 get_full_tick)
            realtime_quotes = self.data_provider.get_realtime_quotes(code_list)
        else:
            # 回放模式: 不调用 QMT，使用空 DataFrame (让 build_feat_tensor 完全使用 SQL 数据)
            realtime_quotes = pd.DataFrame()
        
        # 3. 构建因子张量 (根据公式类型选择构建方式)
        if self.required_window > 1:
            # 时序公式: 需要历史窗口
            feat_tensor, asset_list = self.data_provider.build_feat_tensor_with_history(
                date=date,
                realtime_quotes=realtime_quotes,
                window=self.required_window,
                strict_date_mode=is_live_mode
            )
            names_dict = self.data_provider.get_names_dict(cb_features)
        else:
            # 横截面公式: 单日数据
            feat_tensor = self.data_provider.build_feat_tensor(
                realtime_quotes, cb_features, strict_date_mode=is_live_mode
            )
            # 需要扩展为 [1, Assets, Features] 供 VM 使用
            feat_tensor = feat_tensor.unsqueeze(0)
            asset_list = self.data_provider.get_asset_list(cb_features)
            names_dict = self.data_provider.get_names_dict(cb_features)

        
        # 3. 构建价格字典
        prices = self._build_price_dict(cb_features, realtime_quotes)
        
        # 4. 止盈检测 (传入 realtime_quotes 以便实盘模式使用 QMT 数据)
        tp_orders = []
        if self.take_profit_ratio > 0:
            tp_orders = self._check_take_profit(cb_features, realtime_quotes, prices, date)
            if tp_orders:
                logger.info(f"止盈订单: {len(tp_orders)} 笔")
                self.trader.execute(tp_orders, prices, date)
        
        # 5. 计算因子值并选股
        factor_values = self._compute_factor(feat_tensor)
        target_codes = self._select_top_k(
            factor_values, 
            asset_list, 
            prices,
            names_dict=names_dict,
            date=date
        )
        
        # 6. 生成调仓订单
        rebalance_orders = self._generate_rebalance_orders(
            target_codes, prices, names_dict, date
        )
        
        # 7. 执行调仓
        if rebalance_orders:
            logger.info(f"调仓订单: {len(rebalance_orders)} 笔")
            self.trader.execute(rebalance_orders, prices, date)
        
        # 8. 更新持仓价格
        self._update_position_prices(prices)
        
        # 9. 记录净值
        holdings_value = self.portfolio.get_holdings_value()
        holdings_count = self.portfolio.get_holdings_count()
        record = self.nav_tracker.record_daily(date, holdings_value, holdings_count)
        
        return {
            "status": "success",
            "date": date,
            "nav": record.nav,
            "daily_ret": record.daily_ret,
            "holdings_count": holdings_count,
            "tp_orders": len(tp_orders),
            "rebalance_orders": len(rebalance_orders),
        }
    
    def _ensure_backtest_context(self):
        """Load strict replay context on demand."""
        if self._bt_loader is None:
            if self.replay_source == "sql_eod":
                from data_pipeline.sql_strict_loader import SQLStrictLoader

                self._bt_loader = SQLStrictLoader()
            else:
                # fallback for compatibility
                from model_core.data_loader import CBDataLoader

                self._bt_loader = CBDataLoader()

            self._bt_loader.load_data()
            self._bt_code_to_idx = {
                code: i for i, code in enumerate(self._bt_loader.assets_list)
            }
        
        if self._bt_factors is None:
            feat_tensor_cpu = self._bt_loader.feat_tensor.to("cpu")
            factors = self.vm.execute(self.formula, feat_tensor_cpu)
            if factors is None:
                raise ValueError("Formula execution failed: VM returned None")
            self._bt_factors = factors.to("cpu")
    
    def _build_prices_from_loader(self, date_idx: int) -> Dict[str, float]:
        """Build close price dict from strict loader cache."""
        close_row = self._bt_loader.raw_data_cache["CLOSE"][date_idx].to("cpu")
        prices: Dict[str, float] = {}
        for i, code in enumerate(self._bt_loader.assets_list):
            prices[code] = float(close_row[i].item())
        return prices
    
    def _check_take_profit_from_loader(
        self,
        date_idx: int,
    ) -> List[SimOrder]:
        """Check take-profit signals using strict loader OHLC data."""
        if date_idx <= 0:
            return []
        
        close = self._bt_loader.raw_data_cache["CLOSE"].to("cpu")
        open_ = self._bt_loader.raw_data_cache["OPEN"].to("cpu")
        high = self._bt_loader.raw_data_cache["HIGH"].to("cpu")
        
        orders: List[SimOrder] = []
        for pos in self.portfolio.get_all_positions():
            asset_idx = self._bt_code_to_idx.get(pos.code)
            if asset_idx is None:
                continue
            
            prev_close = float(close[date_idx - 1, asset_idx].item())
            today_open = float(open_[date_idx, asset_idx].item())
            today_high = float(high[date_idx, asset_idx].item())
            
            if prev_close <= 0:
                continue
            
            tp_trigger = prev_close * (1 + self.take_profit_ratio)
            if today_open >= tp_trigger:
                orders.append(
                    SimOrder(
                        code=pos.code,
                        name=pos.name,
                        side=OrderSide.SELL,
                        shares=pos.shares,
                        target_price=today_open,
                        is_take_profit=True,
                    )
                )
            elif today_high >= tp_trigger:
                orders.append(
                    SimOrder(
                        code=pos.code,
                        name=pos.name,
                        side=OrderSide.SELL,
                        shares=pos.shares,
                        target_price=tp_trigger,
                        is_take_profit=True,
                    )
                )
        
        return orders
    
    def _run_daily_replay_strict(self, date: str) -> Dict:
        """Run strict replay logic with loader-aligned pipeline (SQL EOD by default)."""
        self._ensure_backtest_context()
        
        if date not in self._bt_loader.dates_list:
            logger.warning(f"[{self.strategy_id}] date not found in backtest data: {date}")
            return {"status": "no_data"}
        
        date_idx = self._bt_loader.dates_list.index(date)
        prices = self._build_prices_from_loader(date_idx)
        names_dict = self._bt_loader.names_dict
        
        tp_orders: List[SimOrder] = []
        if self.take_profit_ratio > 0:
            tp_orders = self._check_take_profit_from_loader(date_idx)
            if tp_orders:
                logger.info(f"止盈订单: {len(tp_orders)} 笔")
                self.trader.execute(tp_orders, prices, date)
        
        factor_row = self._bt_factors[date_idx]
        valid_mask = self._bt_loader.valid_mask[date_idx].to("cpu")
        target_codes = self._select_top_k(
            factor_values=factor_row,
            asset_list=self._bt_loader.assets_list,
            prices=prices,
            names_dict=names_dict,
            date=date,
            valid_mask=valid_mask,
        )
        
        rebalance_orders = self._generate_rebalance_orders(
            target_codes, prices, names_dict, date
        )
        if rebalance_orders:
            logger.info(f"调仓订单: {len(rebalance_orders)} 笔")
            self.trader.execute(rebalance_orders, prices, date)
        
        self._update_position_prices(prices)
        holdings_value = self.portfolio.get_holdings_value()
        holdings_count = self.portfolio.get_holdings_count()
        record = self.nav_tracker.record_daily(date, holdings_value, holdings_count)
        
        return {
            "status": "success",
            "date": date,
            "nav": record.nav,
            "daily_ret": record.daily_ret,
            "holdings_count": holdings_count,
            "tp_orders": len(tp_orders),
            "rebalance_orders": len(rebalance_orders),
            "mode": "replay_strict",
        }
    
    def _build_price_dict(
        self, 
        cb_features: pd.DataFrame, 
        realtime_quotes: pd.DataFrame
    ) -> Dict[str, float]:
        """构建价格字典"""
        prices = {}
        
        # 从 cb_features 获取基础价格
        for _, row in cb_features.iterrows():
            prices[row['code']] = float(row.get('close', 0))
        
        # 用实时报价覆盖
        if not realtime_quotes.empty:
            for _, row in realtime_quotes.iterrows():
                if row['close'] > 0:
                    prices[row['code']] = float(row['close'])
        
        return prices
    
    def _check_take_profit(
        self, 
        cb_features: pd.DataFrame,
        realtime_quotes: pd.DataFrame,
        prices: Dict[str, float],
        date: str
    ) -> List[SimOrder]:
        """
        检测止盈信号 (严格对齐 backtest.py 逻辑)
        
        逻辑:
        - 基准价格 (prev_close): T-1 日收盘价 (从 SQL 独立查询)
        - 止盈价格 (tp_trigger): prev_close * (1 + TP_RATIO)
        - 跳空止盈: 当日 open >= tp_trigger -> 以 open 成交
        - 盘中止盈: 当日 high >= tp_trigger -> 以 tp_trigger 成交
        
        注意:
        - 实盘模式优先使用 QMT 实时 open/high
        - 回放模式使用 SQL 中的 open/high
        """
        orders = []
        
        # 1. 获取 T-1 日收盘价 (严格来源)
        prev_close_dict = self.data_provider.get_prev_close(date)
        if not prev_close_dict:
            logger.warning(f"无法获取 T-1 收盘价，跳过止盈检测")
            return orders
        
        # 2. 构建当日 open/high 查找字典 (优先使用 QMT 实时数据)
        today_ohlc = {}
        
        # 先从 SQL (cb_features) 获取基础数据
        for _, row in cb_features.iterrows():
            today_ohlc[row['code']] = {
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
            }
        
        # 如果有 QMT 实时数据，优先使用
        if not realtime_quotes.empty:
            for _, row in realtime_quotes.iterrows():
                code = row['code']
                if code in today_ohlc:
                    qmt_open = float(row.get('open', 0))
                    qmt_high = float(row.get('high', 0))
                    if qmt_open > 0:
                        today_ohlc[code]['open'] = qmt_open
                    if qmt_high > 0:
                        today_ohlc[code]['high'] = qmt_high
        
        # 3. 遍历持仓检测止盈
        for pos in self.portfolio.get_all_positions():
            if pos.code not in prev_close_dict:
                continue
            if pos.code not in today_ohlc:
                continue
            
            prev_close = prev_close_dict[pos.code]
            data = today_ohlc[pos.code]
            
            if prev_close <= 0:
                continue
                
            tp_trigger = prev_close * (1 + self.take_profit_ratio)
            
            # 跳空止盈
            if data['open'] >= tp_trigger:
                exec_price = data['open']
                orders.append(SimOrder(
                    code=pos.code,
                    name=pos.name,
                    side=OrderSide.SELL,
                    shares=pos.shares,
                    target_price=exec_price,
                    is_take_profit=True,
                ))
                logger.info(f"[止盈] {pos.code} 跳空止盈 (PrevClose:{prev_close:.2f}, Open:{data['open']:.2f} >= Target:{tp_trigger:.2f})")
            
            # 盘中止盈
            elif data['high'] >= tp_trigger:
                exec_price = tp_trigger
                orders.append(SimOrder(
                    code=pos.code,
                    name=pos.name,
                    side=OrderSide.SELL,
                    shares=pos.shares,
                    target_price=exec_price,
                    is_take_profit=True,
                ))
                logger.info(f"[止盈] {pos.code} 盘中止盈 (PrevClose:{prev_close:.2f}, High:{data['high']:.2f} >= Target:{tp_trigger:.2f})")
        
        return orders

    
    def _compute_factor(self, feat_tensor: torch.Tensor) -> torch.Tensor:
        """
        计算因子值
        
        Args:
            feat_tensor: [Time, Assets, Features] 格式的张量 (由 run_daily 预处理)
            
        Returns:
            [Assets] 格式的因子值 (取最后一个时间点)
        """
        # feat_tensor 已经是 [Time, Assets, Features] 格式
        factor_values = self.vm.execute(self.formula, feat_tensor)
        if factor_values is None:
            raise ValueError("Formula execution failed: VM returned None")
        # return latest-day factor values
        return factor_values[-1] if factor_values.dim() == 2 else factor_values

    
    def _select_top_k(
        self, 
        factor_values: torch.Tensor, 
        asset_list: List[str],
        prices: Dict[str, float],
        names_dict: Dict[str, str] = None,
        date: str = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """选择 Top-K 标的 (支持回测 valid_mask 对齐)"""
        # 统一到 1D CPU/Device 向量
        values = factor_values
        if values.dim() > 1:
            values = values.reshape(-1)

        # 优先使用外部传入的可交易掩码（回测严格对齐模式）
        if valid_mask is not None:
            mask = valid_mask
            if mask.dim() > 1:
                mask = mask.reshape(-1)
            mask = mask.to(device=values.device, dtype=torch.bool)
        else:
            # 兼容旧逻辑：仅以价格 > 0 作为可交易约束
            mask = torch.tensor(
                [prices.get(code, 0) > 0 for code in asset_list],
                device=values.device,
                dtype=torch.bool,
            )

        # 长度保护：避免外部数据长度不一致导致索引越界
        expected_len = min(values.numel(), len(asset_list), mask.numel())
        if expected_len <= 0:
            return []

        if values.numel() != expected_len or len(asset_list) != expected_len or mask.numel() != expected_len:
            logger.warning(
                f"[{self.strategy_id}] Top-K input size mismatch: "
                f"factors={values.numel()}, assets={len(asset_list)}, mask={mask.numel()}, "
                f"use_first={expected_len}"
            )

        values = values[:expected_len]
        asset_list = asset_list[:expected_len]
        mask = mask[:expected_len]

        valid_count = int(mask.sum().item())

        # 与回测保持一致：仅在 strict/显式 valid_mask 模式下启用最小可交易数量阈值
        if valid_mask is not None or self.replay_strict:
            min_required = max(30, self.top_k * 2)
            if valid_count < min_required:
                logger.warning(
                    f"[{self.strategy_id}] Skip trading: valid assets too few "
                    f"({valid_count} < {min_required})"
                )
                return []

        masked_values = values.clone()
        masked_values[~mask] = float('-inf')

        # 屏蔽非有限值，避免 topk 不稳定
        finite_mask = torch.isfinite(masked_values)
        masked_values[~finite_mask] = float('-inf')

        # 取 Top-K
        k = min(self.top_k, valid_count)
        if k == 0:
            return []

        scores, indices = torch.topk(masked_values, k, largest=True)
        target_codes = [asset_list[i] for i in indices.tolist()]
        
        # 保存候选记录 (如果提供了 date 和 names_dict)
        if date and names_dict:
            self._save_candidates(date, target_codes, scores.tolist(), names_dict)
            
        return target_codes

    def _save_candidates(
        self,
        date: str,
        codes: List[str],
        scores: List[float],
        names_dict: Dict[str, str]
    ):
        """保存每日 Top-K 候选记录"""
        candidates = []
        for code, score in zip(codes, scores):
            candidates.append({
                "code": code,
                "name": names_dict.get(code, code),
                "factor_score": round(float(score), 4)
            })
            
        record = {
            "date": date,
            "candidates": candidates
        }
        
        # 确定保存路径
        strategy_dir = os.path.join(self.PORTFOLIO_BASE_DIR, self.strategy_id)
        file_path = os.path.join(strategy_dir, "candidates_history.json")
        
        # 读取现有记录 (如果存在)
        history = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        
        # 如果当日已有记录，则更新；否则追加
        updated = False
        for i, item in enumerate(history):
            if item.get("date") == date:
                history[i] = record
                updated = True
                break
        
        if not updated:
            history.append(record)
            
        # 写入文件
        os.makedirs(strategy_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def _generate_rebalance_orders(
        self, 
        target_codes: List[str],
        prices: Dict[str, float],
        names_dict: Dict[str, str],
        date: str
    ) -> List[SimOrder]:
        """生成调仓订单（复用 CBRebalancer，再适配为 SimOrder）"""
        target_info: List[AssetInfo] = []
        for rank, code in enumerate(target_codes, start=1):
            target_info.append(
                AssetInfo(
                    code=code,
                    name=names_dict.get(code, code),
                    price=float(prices.get(code, 0.0)),
                    score=0.0,
                    rank=rank,
                )
            )

        current_codes = self.portfolio.get_position_codes()
        current_holdings = {}
        sell_prices = {}
        total_equity = self.nav_tracker.cash
        for code in current_codes:
            pos = self.portfolio.get_position(code)
            if not pos:
                continue
            current_holdings[code] = pos.shares
            px = float(prices.get(code, pos.last_price))
            sell_prices[code] = px
            total_equity += pos.shares * px

        self.rebalancer.total_capital = max(float(total_equity), 0.0)
        raw_orders = self.rebalancer.generate_orders(
            current_codes=current_codes,
            target_info=target_info,
            current_holdings=current_holdings,
            sell_prices=sell_prices,
        )

        # execution.cb_trader.Order -> execution.sim_trader.SimOrder
        sim_orders: List[SimOrder] = []
        for order in raw_orders:
            side = OrderSide.BUY
            if order.side == CBOrderSide.SELL:
                side = OrderSide.SELL

            shares = int(getattr(order, "quantity", 0))
            if shares <= 0:
                continue

            target_price = float(getattr(order, "price", 0.0) or 0.0)
            if target_price <= 0:
                target_price = float(prices.get(order.code, 0.0))
            if target_price <= 0:
                continue

            name = order.name if getattr(order, "name", "") else names_dict.get(order.code, order.code)
            if name == order.code:
                # For sell orders, prefer existing position name if available.
                pos = self.portfolio.get_position(order.code)
                if pos and pos.name:
                    name = pos.name

            sim_orders.append(
                SimOrder(
                    code=order.code,
                    name=name,
                    side=side,
                    shares=shares,
                    target_price=target_price,
                )
            )

        return sim_orders
    
    def _update_position_prices(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for code in self.portfolio.get_position_codes():
            if code in prices:
                self.portfolio.update_price(code, prices[code])
