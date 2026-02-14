"""
SimulationRunner - 妯℃嫙鐩樹富鎺ч€昏緫

姣忔棩 14:50 杩愯锛屾墽琛屼互涓嬫祦绋嬶細
1. 鑾峰彇瀹炴椂琛屾儏鍜?CB 鐗规€ф暟鎹?
2. 鏋勫缓鍥犲瓙寮犻噺骞惰绠楀洜瀛愬€?
3. 妫€娴嬫鐩堜俊鍙?
4. 绛涢€?Top-K 鏍囩殑
5. 鐢熸垚璋冧粨璁㈠崟骞舵墽琛?
6. 璁板綍姣忔棩鍑€鍊?
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
from strategy_manager.sql_state_store import SQLStateStore
from execution.sim_trader import SimTrader, SimOrder, OrderSide, TradeRecord
from execution.cb_trader import OrderSide as CBOrderSide

# 寤惰繜瀵煎叆閬垮厤寰幆寮曠敤
StrategyConfig = None

logger = logging.getLogger(__name__)


def _get_strategy_config_class():
    """寤惰繜瀵煎叆 StrategyConfig"""
    global StrategyConfig
    if StrategyConfig is None:
        from strategy_manager.strategy_config import StrategyConfig as SC
        StrategyConfig = SC
    return StrategyConfig


class SimulationRunner:
    """
    妯℃嫙鐩樿繍琛屽櫒
    
    鑱岃矗:
    1. 姣忔棩璋冨害鏁版嵁鑾峰彇鍜屽洜瀛愯绠?
    2. 鎵ц姝㈢泩閫昏緫
    3. Top-K 閫夎偂涓庤皟浠?
    4. 璁板綍鍑€鍊煎彉鍖?
    
    鏀寔涓ょ鍒濆鍖栨柟寮?
    1. 浼犵粺妯″紡: 浼犲叆鐙珛鐨?portfolio, nav_tracker, formula_path 绛夊弬鏁?
    2. 绛栫暐妯″紡: 浼犲叆 StrategyConfig锛岃嚜鍔ㄥ垱寤洪殧绂荤殑缁勪欢瀹炰緥
    """
    
    # 榛樿鍏紡鏂囦欢 (鐜板凡鏀寔 TS_* 鏃跺簭绠楀瓙)
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
        state_backend: str = "sql",  # sql/json
    ):
        """
        鍒濆鍖栨ā鎷熺洏杩愯鍣?
        
        Args:
            data_provider: 瀹炴椂鏁版嵁鎻愪緵鑰?
            portfolio: 缁勫悎绠＄悊鍣?(浼犵粺妯″紡)
            nav_tracker: 鍑€鍊艰拷韪櫒 (浼犵粺妯″紡)
            formula_path: 鍥犲瓙鍏紡璺緞 (浼犵粺妯″紡)
            top_k: 姣忔棩鎸佷粨鏁伴噺 (浼犵粺妯″紡)
            take_profit_ratio: 姝㈢泩姣斾緥 (浼犵粺妯″紡)
            strategy_config: 绛栫暐閰嶇疆 (绛栫暐妯″紡锛屼紭鍏堢骇楂樹簬浼犵粺鍙傛暟)
            replay_source: 涓ユ牸鍥炴斁鏁版嵁婧?(sql_eod/parquet)
        """
        self.data_provider = data_provider
        self.state_backend = state_backend
        
        if strategy_config is not None:
            # strategy mode
            self._init_from_strategy_config(strategy_config)
        else:
            # legacy mode
            self._init_legacy(
                portfolio,
                nav_tracker,
                formula_path,
                top_k,
                take_profit_ratio,
                replay_strict,
                replay_source,
                state_backend,
            )
        
        # 鍒濆鍖栬櫄鎷熸満
        self.vm = StackVM()

        # strict replay caches (used only in historical replay mode)
        self._bt_loader = None
        self._bt_factors = None
        self._bt_code_to_idx = {}

        # Reuse shared rebalance logic to align with CBStrategyRunner/verify flow.
        self.rebalancer = CBRebalancer(total_capital=self.nav_tracker.initial_capital)

        self.sql_state_store = None
        if self.state_backend == "sql":
            self.sql_state_store = SQLStateStore(sql_engine=self.data_provider.sql_engine)
            self._hydrate_state_from_sql()

    def _hydrate_state_from_sql(self):
        """Load latest strategy state from SQL into in-memory components."""
        if self.sql_state_store is None:
            return

        state = self.sql_state_store.load_runtime_state(
            strategy_id=self.strategy_id,
            initial_capital=self.nav_tracker.initial_capital,
        )

        self.portfolio.positions = {p.code: p for p in state["positions"]}
        self.nav_tracker.records = state["records"]
        self.nav_tracker.cash = float(state["cash"])
        self.nav_tracker.peak_nav = float(state["peak_nav"])
        self.trader.trade_history = state["trade_history"]

        logger.info(
            f"[{self.strategy_id}] SQL state loaded: "
            f"positions={len(self.portfolio.positions)}, "
            f"nav_records={len(self.nav_tracker.records)}, "
            f"trades={len(self.trader.trade_history)}"
        )

    def _persist_day_state(self, date: str, record, trade_records: List[TradeRecord]):
        """Persist daily state to SQL if SQL backend is enabled."""
        if self.sql_state_store is None:
            return
        self.sql_state_store.save_daily_state(
            strategy_id=self.strategy_id,
            trade_date=date,
            nav_record=record,
            positions=self.portfolio.get_all_positions(),
            trade_records=trade_records,
        )
    
    def _init_from_strategy_config(self, strategy_config):
        """浠?StrategyConfig 鍒濆鍖?(绛栫暐妯″紡)"""
        self.strategy_id = strategy_config.id
        self.strategy_name = strategy_config.name
        
        # 鍒涘缓绛栫暐涓撳睘鐨勬暟鎹洰褰?
        strategy_dir = os.path.join(self.PORTFOLIO_BASE_DIR, self.strategy_id)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # 鍒濆鍖栫粍鍚堢鐞嗗櫒
        holdings_path = os.path.join(strategy_dir, "holdings.json")
        self.portfolio = CBPortfolioManager(state_path=holdings_path)
        
        # 鍒濆鍖栧噣鍊艰拷韪櫒
        nav_path = os.path.join(strategy_dir, "nav_history.json")
        self.nav_tracker = NavTracker(
            initial_capital=strategy_config.params.initial_capital,
            state_path=nav_path
        )
        
        # 鍒濆鍖栨ā鎷熶氦鏄撳櫒
        history_path = os.path.join(strategy_dir, "trade_history.json")
        self.trader = SimTrader(
            portfolio=self.portfolio,
            nav_tracker=self.nav_tracker,
            fee_rate=strategy_config.params.fee_rate,
            history_path=history_path
        )
        
        # 鍔犺浇鍏紡
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.formula = strategy_config.get_formula(base_dir)
        
        # 鏍￠獙鍏紡 (妫€娴?TS_ 鏃跺簭绠楀瓙)
        self._validate_formula()
        
        # 绛栫暐鍙傛暟
        self.top_k = strategy_config.params.top_k
        self.take_profit_ratio = strategy_config.params.take_profit_ratio
        self.replay_strict = getattr(strategy_config.params, "replay_strict", False)
        self.replay_source = getattr(strategy_config.params, "replay_source", "sql_eod")
        self.state_backend = getattr(strategy_config.params, "state_backend", self.state_backend)
        
        logger.info(
            f"[{self.strategy_id}] 缁涙牜鏆愰崚婵嗩潗閸栨牕鐣幋? "
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
        state_backend: str,
    ):
        """Initialize runner in legacy single-strategy mode."""
        self.strategy_id = "default"
        self.strategy_name = "Default Strategy"
        
        self.portfolio = portfolio
        self.nav_tracker = nav_tracker
        self.top_k = top_k
        self.take_profit_ratio = take_profit_ratio
        self.replay_strict = replay_strict
        self.replay_source = replay_source
        self.state_backend = state_backend
        
        # 鍔犺浇鍥犲瓙鍏紡
        self.formula_path = formula_path or self.DEFAULT_FORMULA_PATH
        self.formula = self._load_formula()
        
        # 鏍￠獙鍏紡 (妫€娴?TS_ 鏃跺簭绠楀瓙)
        self._validate_formula()
        
        # 鍒濆鍖栨ā鎷熶氦鏄撳櫒
        self.trader = SimTrader(
            portfolio=portfolio, 
            nav_tracker=nav_tracker,
            fee_rate=RobustConfig.FEE_RATE
        )
    
    def _load_formula(self) -> List[str]:
        """鍔犺浇鍥犲瓙鍏紡 (浼犵粺妯″紡浣跨敤)"""
        if not os.path.exists(self.formula_path):
            raise FileNotFoundError(f"鍏紡鏂囦欢涓嶅瓨鍦? {self.formula_path}")
        
        with open(self.formula_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formula = data.get('best', {}).get('formula', [])
        logger.info(f"Load formula: {' '.join(formula)}")
        return formula
    
    def _get_required_window(self) -> int:
        """
        鏍规嵁鍏紡涓殑 TS 绠楀瓙纭畾鎵€闇€鐨勫巻鍙茬獥鍙ｅぇ灏?
        
        Returns:
            1: 绾í鎴潰鍏紡 (鏃?TS_* 绠楀瓙)
            2: 鍖呭惈 TS_DELAY/TS_DELTA/TS_RET
            5: 鍖呭惈 TS_MEAN5/TS_STD5/TS_BIAS5
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
        鏍￠獙鍏紡骞惰褰曠獥鍙ｉ渶姹?
        """
        self.required_window = self._get_required_window()
        if self.required_window > 1:
            ts_ops = [op for op in self.formula if op.startswith('TS_')]
            logger.info(
                f"[{self.strategy_id}] formula contains TS ops: {ts_ops}, "
                f"required_window={self.required_window}"
            )
        else:
            logger.info(f"[{self.strategy_id}] cross-sectional formula, use 1-day window")


    
    def run_daily(self, date: str) -> Dict:
        """
        鎵ц姣忔棩妯℃嫙
        
        Args:
            date: 鏃ユ湡 (YYYY-MM-DD)
            
        Returns:
            杩愯缁撴灉鎽樿
        """
        logger.info(f"=== Simulation Run {date} ===")
        
        # 鍒ゆ柇鏄惁涓哄疄鐩樻ā寮?(浠婂ぉ) 鎴栧洖鏀炬ā寮?(鍘嗗彶鏃ユ湡)
        today_str = datetime.now().strftime('%Y-%m-%d')
        is_live_mode = (date == today_str)
        
        if is_live_mode:
            logger.info("[live mode] use MiniQMT realtime quotes")
        else:
            logger.info("[replay mode] use SQL historical data only")
        
        # 1. 鑾峰彇鍩虹鐗瑰緛
        if self.replay_strict and not is_live_mode:
            logger.info(
                f"[{self.strategy_id}] strict replay mode enabled: source={self.replay_source}"
            )
            return self._run_daily_replay_strict(date)
        
        cb_features = self.data_provider.get_cb_features(date)
        if cb_features.empty:
            logger.warning(f"No SQL feature data for date={date}")
            return {"status": "no_data"}
            
        code_list = cb_features['code'].tolist()
        
        # 2. 鏍规嵁妯″紡鍐冲畾鏄惁鑾峰彇瀹炴椂琛屾儏
        if is_live_mode:
            # 寤虹珛瀹炴椂鎺ㄩ€佽繛鎺?(Mini QMT 鎺ㄨ崘鐢ㄦ硶)
            self.data_provider.subscribe_quotes(code_list)
            # 鑾峰彇瀹炴椂蹇収 (浣跨敤 get_full_tick)
            realtime_quotes = self.data_provider.get_realtime_quotes(code_list)
        else:
            # 鍥炴斁妯″紡: 涓嶈皟鐢?QMT锛屼娇鐢ㄧ┖ DataFrame (璁?build_feat_tensor 瀹屽叏浣跨敤 SQL 鏁版嵁)
            realtime_quotes = pd.DataFrame()
        
        # 3. 鏋勫缓鍥犲瓙寮犻噺 (鏍规嵁鍏紡绫诲瀷閫夋嫨鏋勫缓鏂瑰紡)
        if self.required_window > 1:
            # 鏃跺簭鍏紡: 闇€瑕佸巻鍙茬獥鍙?
            feat_tensor, asset_list = self.data_provider.build_feat_tensor_with_history(
                date=date,
                realtime_quotes=realtime_quotes,
                window=self.required_window,
                strict_date_mode=is_live_mode
            )
            names_dict = self.data_provider.get_names_dict(cb_features)
        else:
            # 妯埅闈㈠叕寮? 鍗曟棩鏁版嵁
            feat_tensor = self.data_provider.build_feat_tensor(
                realtime_quotes, cb_features, strict_date_mode=is_live_mode
            )
            # 闇€瑕佹墿灞曚负 [1, Assets, Features] 渚?VM 浣跨敤
            feat_tensor = feat_tensor.unsqueeze(0)
            asset_list = self.data_provider.get_asset_list(cb_features)
            names_dict = self.data_provider.get_names_dict(cb_features)

        
        # 3. 鏋勫缓浠锋牸瀛楀吀
        prices = self._build_price_dict(cb_features, realtime_quotes)
        
        # 4. 姝㈢泩妫€娴?(浼犲叆 realtime_quotes 浠ヤ究瀹炵洏妯″紡浣跨敤 QMT 鏁版嵁)
        tp_orders = []
        tp_records: List[TradeRecord] = []
        if self.take_profit_ratio > 0:
            tp_orders = self._check_take_profit(cb_features, realtime_quotes, prices, date)
            if tp_orders:
                logger.info(f"止盈订单: {len(tp_orders)} 笔")
                tp_records = self.trader.execute(tp_orders, prices, date)
        
        # 5. 璁＄畻鍥犲瓙鍊煎苟閫夎偂
        factor_values = self._compute_factor(feat_tensor)
        target_codes = self._select_top_k(
            factor_values, 
            asset_list, 
            prices,
            names_dict=names_dict,
            date=date
        )
        
        # 6. 鐢熸垚璋冧粨璁㈠崟
        rebalance_orders = self._generate_rebalance_orders(
            target_codes, prices, names_dict, date
        )
        
        # 7. 鎵ц璋冧粨
        rebalance_records: List[TradeRecord] = []
        if rebalance_orders:
            logger.info(f"调仓订单: {len(rebalance_orders)} 笔")
            rebalance_records = self.trader.execute(rebalance_orders, prices, date)
        
        # 8. 鏇存柊鎸佷粨浠锋牸
        self._update_position_prices(prices)
        
        # 9. 璁板綍鍑€鍊?
        holdings_value = self.portfolio.get_holdings_value()
        holdings_count = self.portfolio.get_holdings_count()
        record = self.nav_tracker.record_daily(date, holdings_value, holdings_count)
        self._persist_day_state(date, record, tp_records + rebalance_records)
        
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
        tp_records: List[TradeRecord] = []
        if self.take_profit_ratio > 0:
            tp_orders = self._check_take_profit_from_loader(date_idx)
            if tp_orders:
                logger.info(f"止盈订单: {len(tp_orders)} 笔")
                tp_records = self.trader.execute(tp_orders, prices, date)
        
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
        rebalance_records: List[TradeRecord] = []
        if rebalance_orders:
            logger.info(f"调仓订单: {len(rebalance_orders)} 笔")
            rebalance_records = self.trader.execute(rebalance_orders, prices, date)
        
        self._update_position_prices(prices)
        holdings_value = self.portfolio.get_holdings_value()
        holdings_count = self.portfolio.get_holdings_count()
        record = self.nav_tracker.record_daily(date, holdings_value, holdings_count)
        self._persist_day_state(date, record, tp_records + rebalance_records)
        
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
        """鏋勫缓浠锋牸瀛楀吀"""
        prices = {}
        
        # 浠?cb_features 鑾峰彇鍩虹浠锋牸
        for _, row in cb_features.iterrows():
            prices[row['code']] = float(row.get('close', 0))
        
        # 鐢ㄥ疄鏃舵姤浠疯鐩?
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
        妫€娴嬫鐩堜俊鍙?(涓ユ牸瀵归綈 backtest.py 閫昏緫)
        
        閫昏緫:
        - 鍩哄噯浠锋牸 (prev_close): T-1 鏃ユ敹鐩樹环 (浠?SQL 鐙珛鏌ヨ)
        - 姝㈢泩浠锋牸 (tp_trigger): prev_close * (1 + TP_RATIO)
        - 璺崇┖姝㈢泩: 褰撴棩 open >= tp_trigger -> 浠?open 鎴愪氦
        - 鐩樹腑姝㈢泩: 褰撴棩 high >= tp_trigger -> 浠?tp_trigger 鎴愪氦
        
        娉ㄦ剰:
        - 瀹炵洏妯″紡浼樺厛浣跨敤 QMT 瀹炴椂 open/high
        - 鍥炴斁妯″紡浣跨敤 SQL 涓殑 open/high
        """
        orders = []
        
        # 1. 鑾峰彇 T-1 鏃ユ敹鐩樹环 (涓ユ牸鏉ユ簮)
        prev_close_dict = self.data_provider.get_prev_close(date)
        if not prev_close_dict:
            logger.warning(f"无法获取 T-1 收盘价，跳过止盈检测: {date}")
            return orders
        
        # 2. 鏋勫缓褰撴棩 open/high 鏌ユ壘瀛楀吀 (浼樺厛浣跨敤 QMT 瀹炴椂鏁版嵁)
        today_ohlc = {}
        
        # 鍏堜粠 SQL (cb_features) 鑾峰彇鍩虹鏁版嵁
        for _, row in cb_features.iterrows():
            today_ohlc[row['code']] = {
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
            }
        
        # 濡傛灉鏈?QMT 瀹炴椂鏁版嵁锛屼紭鍏堜娇鐢?
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
        
        # 3. 閬嶅巻鎸佷粨妫€娴嬫鐩?
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
            
            # 璺崇┖姝㈢泩
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
                logger.info(
                    f"[TP] {pos.code} gap-take-profit "
                    f"(PrevClose:{prev_close:.2f}, Open:{data['open']:.2f} >= Target:{tp_trigger:.2f})"
                )
            
            # 鐩樹腑姝㈢泩
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
                logger.info(
                    f"[TP] {pos.code} intraday-take-profit "
                    f"(PrevClose:{prev_close:.2f}, High:{data['high']:.2f} >= Target:{tp_trigger:.2f})"
                )
        
        return orders

    
    def _compute_factor(self, feat_tensor: torch.Tensor) -> torch.Tensor:
        """
        璁＄畻鍥犲瓙鍊?
        
        Args:
            feat_tensor: [Time, Assets, Features] 鏍煎紡鐨勫紶閲?(鐢?run_daily 棰勫鐞?
            
        Returns:
            [Assets] 鏍煎紡鐨勫洜瀛愬€?(鍙栨渶鍚庝竴涓椂闂寸偣)
        """
        # feat_tensor 宸茬粡鏄?[Time, Assets, Features] 鏍煎紡
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
        """閫夋嫨 Top-K 鏍囩殑 (鏀寔鍥炴祴 valid_mask 瀵归綈)"""
        # 缁熶竴鍒?1D CPU/Device 鍚戦噺
        values = factor_values
        if values.dim() > 1:
            values = values.reshape(-1)

        # 浼樺厛浣跨敤澶栭儴浼犲叆鐨勫彲浜ゆ槗鎺╃爜锛堝洖娴嬩弗鏍煎榻愭ā寮忥級
        if valid_mask is not None:
            mask = valid_mask
            if mask.dim() > 1:
                mask = mask.reshape(-1)
            mask = mask.to(device=values.device, dtype=torch.bool)
        else:
            # Legacy fallback: use price > 0 as tradable mask.
            mask = torch.tensor(
                [prices.get(code, 0) > 0 for code in asset_list],
                device=values.device,
                dtype=torch.bool,
            )

        # Protect against shape mismatch from external inputs.
        expected_len = min(values.numel(), len(asset_list), mask.numel())
        if expected_len <= 0:
            return []
            logger.warning(
                f"[{self.strategy_id}] Top-K input size mismatch: "
                f"factors={values.numel()}, assets={len(asset_list)}, mask={mask.numel()}, "
                f"use_first={expected_len}"
            )

        values = values[:expected_len]
        asset_list = asset_list[:expected_len]
        mask = mask[:expected_len]

        valid_count = int(mask.sum().item())

        # Align with backtest guard when strict mode is enabled.
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

        # Mask non-finite values before top-k.
        finite_mask = torch.isfinite(masked_values)
        masked_values[~finite_mask] = float('-inf')
        # 鍙?Top-K
        k = min(self.top_k, valid_count)
        if k == 0:
            return []

        scores, indices = torch.topk(masked_values, k, largest=True)
        target_codes = [asset_list[i] for i in indices.tolist()]
        
        # 淇濆瓨鍊欓€夎褰?(濡傛灉鎻愪緵浜?date 鍜?names_dict)
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
        """Save daily Top-K candidate list."""
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
        
        # 纭畾淇濆瓨璺緞
        strategy_dir = os.path.join(self.PORTFOLIO_BASE_DIR, self.strategy_id)
        file_path = os.path.join(strategy_dir, "candidates_history.json")
        
        # 璇诲彇鐜版湁璁板綍 (濡傛灉瀛樺湪)
        history = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        
        # 濡傛灉褰撴棩宸叉湁璁板綍锛屽垯鏇存柊锛涘惁鍒欒拷鍔?
        updated = False
        for i, item in enumerate(history):
            if item.get("date") == date:
                history[i] = record
                updated = True
                break
        
        if not updated:
            history.append(record)
            
        # 鍐欏叆鏂囦欢
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
        """Generate rebalance orders via shared CBRebalancer logic."""
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
        # Convert first, then enforce cash-aware buy sizing to avoid silent BUY rejection.
        sell_orders: List[SimOrder] = []
        buy_orders: List[Tuple[int, SimOrder]] = []

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
                pos = self.portfolio.get_position(order.code)
                if pos and pos.name:
                    name = pos.name

            sim_order = SimOrder(
                code=order.code,
                name=name,
                side=side,
                shares=shares,
                target_price=target_price,
            )

            if side == OrderSide.SELL:
                sell_orders.append(sim_order)
            else:
                rank = int(getattr(order, "rank", 999999))
                buy_orders.append((rank, sim_order))

        # Keep SELL first so BUY can consume released cash.
        sim_orders: List[SimOrder] = list(sell_orders)

        # Buy ranking: higher-priority target first (smaller rank).
        buy_orders.sort(key=lambda x: x[0])

        fee_rate = float(getattr(self.trader, "fee_rate", 0.0))
        projected_cash = float(self.nav_tracker.cash)

        for s in sell_orders:
            projected_cash += s.shares * s.target_price * (1.0 - fee_rate)

        remaining_buys = len(buy_orders)
        for rank, b in buy_orders:
            if remaining_buys <= 0:
                break

            # Reserve budget for remaining buy orders to avoid early over-consumption.
            budget_per_order = projected_cash / float(remaining_buys)
            lot = 10
            per_share_cost = b.target_price * (1.0 + fee_rate)

            max_shares_by_budget = int((budget_per_order / per_share_cost) / lot) * lot
            max_shares_by_cash = int((projected_cash / per_share_cost) / lot) * lot
            allowed_shares = min(b.shares, max_shares_by_budget, max_shares_by_cash)

            if allowed_shares <= 0:
                logger.warning(
                    f"[{self.strategy_id}] Skip BUY due cash budget: code={b.code}, rank={rank}, "
                    f"cash={projected_cash:.2f}, price={b.target_price:.2f}, wanted={b.shares}"
                )
                remaining_buys -= 1
                continue

            if allowed_shares < b.shares:
                logger.info(
                    f"[{self.strategy_id}] Trim BUY shares by cash budget: code={b.code}, rank={rank}, "
                    f"{b.shares} -> {allowed_shares}"
                )

            sim_orders.append(
                SimOrder(
                    code=b.code,
                    name=b.name,
                    side=OrderSide.BUY,
                    shares=allowed_shares,
                    target_price=b.target_price,
                )
            )

            projected_cash -= allowed_shares * per_share_cost
            remaining_buys -= 1

        return sim_orders
    
    def _update_position_prices(self, prices: Dict[str, float]):
        """鏇存柊鎸佷粨浠锋牸"""
        for code in self.portfolio.get_position_codes():
            if code in prices:
                self.portfolio.update_price(code, prices[code])


