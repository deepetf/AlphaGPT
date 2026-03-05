"""
绛栫暐楠岃瘉鑴氭湰 (Strategy Verification Script)

鐩爣锛?
1. 楠岃瘉 Event-Driven 绛栫暐寮曟搸涓?Vector Backtest 鐨勪竴鑷存€?
2. 妫€娴?Look-Ahead Bias
3. 鐢熸垚閲忓寲瀵规瘮鎶ュ憡

鎵ц鏂瑰紡锛?
    python tests/verify_strategy.py --start 2024-01-01 --end 2024-12-31
"""

import os
import sys
import json
import logging
from itertools import combinations
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import torch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model_core.config import RobustConfig
from model_core.data_loader import CBDataLoader
from model_core.backtest import CBBacktest
from model_core.factors import FeatureEngineer
from model_core.formula_validator import validate_formula
from model_core.signal_utils import build_topk_weights
from model_core.vm import StackVM
from strategy_manager.strategy_config import load_strategies_config
from strategy_manager.cb_runner import CBStrategyRunner
from strategy_manager.cb_portfolio import CBPortfolioManager
from execution.cb_trader import Order, OrderSide
from tests.perf_visualizer import generate_verification_visuals

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "tests", "verification.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger("StrategyVerification")


@dataclass
class DailyRecord:
    """姣忔棩璁板綍"""
    date: str
    sim_equity: float
    sim_return: float
    sim_holdings: List[str]
    sim_cash: float
    orders_count: int
    tx_fee: float = 0.0


class SimAccount:
    """妯℃嫙璐︽埛"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.holdings: Dict[str, int] = {}  # {code: shares}
        self.initial_cash = initial_cash
        
    def get_equity(self, prices: Dict[str, float]) -> float:
        """璁＄畻鎬绘潈鐩?"""
        holdings_value = sum(self.holdings.get(code, 0) * prices.get(code, 0.0) 
                            for code in self.holdings)
        return self.cash + holdings_value
    
    def execute_order(self, order: Order, fee_rate: float = 0.001):
        """鎵ц璁㈠崟"""
        if order.side == OrderSide.BUY:
            cost = order.quantity * order.price * (1 + fee_rate)
            if cost > self.cash:
                logger.warning(f"Insufficient cash for {order.code}: need {cost:.2f}, have {self.cash:.2f}")
                return False
            self.cash -= cost
            self.holdings[order.code] = self.holdings.get(order.code, 0) + order.quantity
            logger.debug(f"BUY {order.code} x{order.quantity} @ {order.price:.2f}, cost={cost:.2f}")
            return True
            
        elif order.side == OrderSide.SELL:
            if order.code not in self.holdings or self.holdings[order.code] < order.quantity:
                logger.warning(f"Insufficient holdings for {order.code}")
                return False
            proceeds = order.quantity * order.price * (1 - fee_rate)
            self.cash += proceeds
            self.holdings[order.code] -= order.quantity
            if self.holdings[order.code] == 0:
                del self.holdings[order.code]
            logger.debug(f"SELL {order.code} x{order.quantity} @ {order.price:.2f}, proceeds={proceeds:.2f}")
            return True
        
        return False


class MockTrader:
    """Mock Trader - 鍙湪鍐呭瓨涓褰曡鍗?"""
    
    def __init__(self):
        self.orders = []
    
    def submit_orders(self, orders, date):
        """璁板綍璁㈠崟浣嗕笉鐢熸垚鏂囦欢"""
        self.orders.extend(orders)
        return type('Result', (), {'success': True, 'message': f'Recorded {len(orders)} orders'})()


class StrategyVerifier:
    """绛栫暐楠岃瘉鍣?"""
    WARMUP_DAYS = 65
    
    def __init__(
        self,
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None,
        king_step: Optional[int] = None,
        take_profit_ratio: Optional[float] = None,
        strategies_config: Optional[str] = None,
        strategy_id: Optional[str] = None,
        top_k: Optional[int] = None,
        fee_rate: Optional[float] = None,
        initial_cash: Optional[float] = None,
        verify_cash_aware: bool = True,
    ):
        """
        鍒濆鍖栭獙璇佸櫒
        
        Args:
            start_date: 寮€濮嬫棩鏈?
            end_date: 缁撴潫鏃ユ湡锛圢one琛ㄧず鏈€鏂版棩鏈燂級
            king_step: King鍥犲瓙鐨剆tep缂栧彿锛圢one琛ㄧず浣跨敤best锛?
        """
        self.start_date = start_date
        self.end_date = end_date
        self.king_step = king_step
        self.strategies_config = strategies_config
        self.strategy_id = strategy_id
        self.top_k = top_k
        self.fee_rate = fee_rate
        self.take_profit_ratio = take_profit_ratio
        self.initial_cash = initial_cash
        self.verify_cash_aware = bool(verify_cash_aware)
        self.formula_source = "unknown"
        self.strategy_name = ""
        self.visual_outputs: Dict[str, str] = {}
        self.daily_trades: List[Dict] = []
        
        # Setup artifacts directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.artifacts_dir = os.path.join(current_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Load data
        logger.info("Loading data...")
        self.loader = CBDataLoader()
        loader_start_date = start_date
        try:
            # Keep enough history before verify start for rolling features and warmup alignment.
            loader_start_date = (
                pd.to_datetime(start_date) - pd.Timedelta(days=200)
            ).strftime("%Y-%m-%d")
        except Exception:
            logger.warning(
                "Failed to derive loader start date from start=%s, fallback to start date.",
                start_date,
            )
        logger.info(
            "Verify loader start_date=%s (requested_start=%s, warmup_days=%d)",
            loader_start_date,
            start_date,
            self.WARMUP_DAYS,
        )
        self.loader.load_data(start_date=loader_start_date)
        
        # Filter dates
        all_dates = self.loader.dates_list
        self.dates = [d for d in all_dates if d >= start_date]
        if end_date:
            self.dates = [d for d in self.dates if d <= end_date]
        if not self.dates:
            raise ValueError(
                f"No trading dates found in verification range: "
                f"start={start_date}, end={end_date or 'latest'}"
            )

        actual_start_date = self.dates[0]
        actual_end_date = self.dates[-1]
        warmup_start_date = self._resolve_warmup_start_date(
            all_dates=all_dates,
            anchor_date=actual_start_date,
            warmup_days=self.WARMUP_DAYS,
        )
        self._trim_loader_and_recompute(
            load_start_date=warmup_start_date,
            load_end_date=actual_end_date,
            feature_warmup_anchor_date=actual_start_date,
        )

        # Rebuild verification dates on trimmed loader.
        self.dates = [d for d in self.loader.dates_list if d >= start_date]
        if end_date:
            self.dates = [d for d in self.dates if d <= end_date]
        if not self.dates:
            raise ValueError(
                f"No trading dates left after warmup trim: "
                f"start={start_date}, end={end_date or 'latest'}"
            )

        self.code_to_idx = {c: i for i, c in enumerate(self.loader.assets_list)}
        logger.info(
            f"Loader window (aligned): {self.loader.dates_list[0]} to "
            f"{self.loader.dates_list[-1]} ({len(self.loader.dates_list)} days, "
            f"warmup_days={self.WARMUP_DAYS})"
        )
        
        logger.info(f"Verification period: {self.dates[0]} to {self.dates[-1]} ({len(self.dates)} days)")
        
        # Load formula and strategy params
        self._load_strategy()

    def _resolve_warmup_start_date(
        self,
        all_dates: List[str],
        anchor_date: str,
        warmup_days: int = 65,
    ) -> str:
        """Return warmup start date using latest warmup_days up to anchor_date (inclusive)."""
        if anchor_date not in all_dates:
            raise ValueError(f"anchor_date not found in trading dates: {anchor_date}")
        anchor_idx = all_dates.index(anchor_date)
        start_idx = max(0, anchor_idx - max(1, int(warmup_days)) + 1)
        return all_dates[start_idx]

    def _trim_loader_and_recompute(
        self,
        load_start_date: str,
        load_end_date: str,
        feature_warmup_anchor_date: Optional[str] = None,
    ) -> None:
        """Trim loader tensors by date range and recompute feature/target tensors for alignment."""
        dates = self.loader.dates_list
        if load_start_date not in dates or load_end_date not in dates:
            raise ValueError(
                f"Trim dates not found in loader range: "
                f"start={load_start_date}, end={load_end_date}"
            )
        start_idx = dates.index(load_start_date)
        end_idx = dates.index(load_end_date)
        if start_idx > end_idx:
            raise ValueError(
                f"Invalid trim range: start={load_start_date} > end={load_end_date}"
            )
        time_slice = slice(start_idx, end_idx + 1)

        # 1) Trim raw market tensors first.
        trimmed_raw = {}
        for key, tensor in self.loader.raw_data_cache.items():
            trimmed_raw[key] = tensor[time_slice]
        self.loader.raw_data_cache = trimmed_raw

        # 2) Trim valid mask and date list.
        self.loader.valid_mask = self.loader.valid_mask[time_slice]
        self.loader.dates_list = dates[time_slice]

        # 3) Recompute features on trimmed window with explicit warmup anchor.
        warmup_rows = 0
        if feature_warmup_anchor_date:
            if feature_warmup_anchor_date in self.loader.dates_list:
                warmup_rows = self.loader.dates_list.index(feature_warmup_anchor_date)
            else:
                logger.warning(
                    "feature_warmup_anchor_date not found in trimmed dates, fallback warmup_rows=0: %s",
                    feature_warmup_anchor_date,
                )
        self.loader.feat_tensor = FeatureEngineer.compute_features(
            self.loader.raw_data_cache,
            warmup_rows=warmup_rows,
        )
        logger.info(
            "Recomputed features with warmup_rows=%d (anchor=%s, range=%s~%s)",
            warmup_rows,
            feature_warmup_anchor_date or "None",
            self.loader.dates_list[0] if self.loader.dates_list else "N/A",
            self.loader.dates_list[-1] if self.loader.dates_list else "N/A",
        )

        # 4) Recompute target return on trimmed window (last day should be zero).
        close = self.loader.raw_data_cache["CLOSE"]
        ret_1d = (torch.roll(close, -1, dims=0) / (close + 1e-9)) - 1.0
        ret_1d[~self.loader.valid_mask] = 0.0
        ret_1d[-1] = 0.0
        self.loader.target_ret = ret_1d

        # Keep split index inside trimmed range for compatibility.
        if getattr(self.loader, "split_idx", None) is not None:
            new_split = self.loader.split_idx - start_idx
            self.loader.split_idx = max(0, min(len(self.loader.dates_list) - 1, new_split))

    def _strategy_tag(self) -> str:
        if self.strategy_id:
            return self.strategy_id
        if self.king_step is not None:
            return f"king_{self.king_step}"
        return "default"

    def _artifact_suffix(self) -> str:
        end = self.end_date or (self.dates[-1] if self.dates else "end")
        return f"_{self._strategy_tag()}_{self.start_date}_{end}"
    
    def _load_strategy(self):
        """Load formula and parameters from strategy config or legacy king."""
        if self.king_step is not None:
            if self.strategy_id:
                raise ValueError("--king cannot be used together with --strategy-id")
            self._load_formula_legacy()
            self.top_k = self.top_k if self.top_k is not None else int(RobustConfig.TOP_K)
            self.fee_rate = self.fee_rate if self.fee_rate is not None else float(RobustConfig.FEE_RATE)
            if self.take_profit_ratio is None:
                self.take_profit_ratio = 0.0
            if self.initial_cash is None:
                self.initial_cash = 100000.0
            self.formula_source = "best_cb_formula.json(legacy)"
            self.strategy_name = f"king_step_{self.king_step}"
            return

        # Config-driven mode (default)
        config_path = self.strategies_config or os.path.join(
            project_root, "strategy_manager", "strategies_config.json"
        )
        cfg = load_strategies_config(config_path)
        enabled = cfg.get_enabled_strategies()
        if not enabled:
            raise ValueError(f"No enabled strategies found in config: {config_path}")

        selected = None
        if self.strategy_id:
            for s in enabled:
                if s.id == self.strategy_id:
                    selected = s
                    break
            if selected is None:
                available = ", ".join([s.id for s in enabled])
                raise ValueError(f"strategy_id '{self.strategy_id}' not found. enabled: {available}")
        else:
            selected = enabled[0]

        self.strategy_id = selected.id
        self.strategy_name = selected.name
        self.formula = selected.get_formula(project_root)
        self.formula_info = {"id": selected.id, "name": selected.name}

        self.top_k = self.top_k if self.top_k is not None else int(selected.params.top_k)
        self.fee_rate = self.fee_rate if self.fee_rate is not None else float(selected.params.fee_rate)
        if self.take_profit_ratio is None:
            self.take_profit_ratio = float(selected.params.take_profit_ratio)
        if self.initial_cash is None:
            self.initial_cash = float(selected.params.initial_capital)

        self.formula_source = config_path
        logger.info(
            f"Using strategy config: id={selected.id}, name={selected.name}, "
            f"top_k={self.top_k}, fee_rate={self.fee_rate}, "
            f"tp={self.take_profit_ratio}, initial_cash={self.initial_cash}"
        )
        logger.info(f"Formula: {' '.join(self.formula)}")
        self._validate_formula_or_raise()

    def _load_formula_legacy(self):
        """鍔犺浇鍥犲瓙鍏紡 (legacy king mode)"""
        strategy_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model_core",
            "best_cb_formula.json"
        )
        
        with open(strategy_path, 'r', encoding='utf-8') as f:
            formula_data = json.load(f)
        
        if self.king_step is None:
            # Use best formula
            self.formula = formula_data['best']['formula']
            self.formula_info = formula_data['best']
            logger.info(f"Using BEST formula (Step {self.formula_info.get('step', 'N/A')})")
        else:
            # Find formula by step
            found = False
            # Check history
            for king in formula_data.get('history', []):
                if king.get('step') == self.king_step:
                    self.formula = king['formula']
                    self.formula_info = king
                    found = True
                    logger.info(f"Using King formula from Step {self.king_step} (found in history)")
                    break
            
            # Check diversity pool if not found
            if not found:
                for king in formula_data.get('diverse_top_50', []):
                    if king.get('step') == self.king_step:
                        self.formula = king['formula']
                        self.formula_info = king
                        found = True
                        logger.info(f"Using Diverse formula from Step {self.king_step} (found in diverse_top_50)")
                        break
            
            if not found:
                history_steps = [k['step'] for k in formula_data.get('history', []) if 'step' in k]
                diverse_steps = [k['step'] for k in formula_data.get('diverse_top_50', []) if 'step' in k]
                all_steps = sorted(list(set(history_steps + diverse_steps)))
                raise ValueError(
                    f"King step {self.king_step} not found in history or diversity pool. "
                    f"Available steps: {all_steps}"
                )
        
        logger.info(f"Formula: {' '.join(self.formula)}")
        logger.info(f"Sharpe: {self.formula_info.get('sharpe', 'N/A'):.4f}, "
                   f"Ann. Return: {self.formula_info.get('annualized_ret', 'N/A'):.2%}")
        self._validate_formula_or_raise()

    def _validate_formula_or_raise(self):
        """Validate formula structure before long-running verification."""
        if not self.formula:
            raise ValueError("Formula is empty.")
        if not isinstance(self.formula, list) or not isinstance(self.formula[0], str):
            raise TypeError("Formula must be a list of string tokens.")
        ok, _, reason = validate_formula(self.formula)
        if not ok:
            formula_str = " ".join(self.formula)
            msg = (
                f"Invalid formula for verification (source={self.formula_source}): {reason}. "
                f"formula={formula_str}"
            )
            logger.error(msg)
            raise ValueError(msg)
        
    def print_config_alignment(self):
        """鎵撳嵃鍙傛暟瀵归綈妫€鏌?"""
        logger.info("="*60)
        logger.info("PARAMETER ALIGNMENT CHECK")
        logger.info("="*60)
        logger.info(f"STRATEGY_ID: {self.strategy_id or 'legacy'}")
        logger.info(f"STRATEGY_NAME: {self.strategy_name or 'legacy'}")
        logger.info(f"FORMULA_SOURCE: {self.formula_source}")
        logger.info(f"TOP_K: {self.top_k}")
        logger.info(f"FEE_RATE: {self.fee_rate} (single-sided, total={self.fee_rate*2})")
        logger.info(f"INITIAL_CASH: {self.initial_cash}")
        logger.info(f"MIN_ACTIVE_RATIO: {RobustConfig.MIN_ACTIVE_RATIO}")
        min_valid_count = max(30, int(self.top_k) * 2)
        logger.info(f"MIN_VALID_COUNT (Computed): {min_valid_count}")
        logger.info(f"TRAIN_TEST_SPLIT_DATE: {RobustConfig.TRAIN_TEST_SPLIT_DATE}")
        logger.info(f"TAKE_PROFIT_RATIO: {self.take_profit_ratio}")
        logger.info(f"VERIFY_CASH_AWARE: {self.verify_cash_aware}")
        logger.info("="*60)

    def _build_prices(self, date_idx: int) -> Dict[str, float]:
        close_row = self.loader.raw_data_cache['CLOSE'][date_idx]
        return {
            code: float(close_row[asset_idx].item())
            for asset_idx, code in enumerate(self.loader.assets_list)
        }

    def _sync_portfolio_with_account(
        self,
        portfolio: CBPortfolioManager,
        account: SimAccount,
        order: Order,
        date: str,
    ):
        current_shares = account.holdings.get(order.code, 0)
        if current_shares > 0:
            if portfolio.get_position(order.code):
                portfolio.update_position(order.code, current_shares, order.price)
            else:
                portfolio.add_position(
                    code=order.code,
                    name=getattr(order, 'name', order.code),
                    shares=current_shares,
                    price=order.price,
                    date=date
                )
        else:
            if portfolio.get_position(order.code):
                portfolio.remove_position(order.code)

    def _execute_orders(
        self,
        account: SimAccount,
        portfolio: CBPortfolioManager,
        orders: List[Order],
        date: str,
    ) -> tuple[int, float]:
        executed = 0
        total_fee = 0.0
        for order in orders:
            success = account.execute_order(order, fee_rate=self.fee_rate)
            if success:
                executed += 1
                total_fee += float(order.quantity) * float(order.price) * self.fee_rate
                self._sync_portfolio_with_account(portfolio, account, order, date)
        return executed, total_fee

    def _is_buy_order(self, order: Order) -> bool:
        side = str(getattr(order, "side", "")).upper()
        return side == "ORDERside.BUY".upper() or side.endswith("BUY")

    def _is_sell_order(self, order: Order) -> bool:
        side = str(getattr(order, "side", "")).upper()
        return side == "ORDERside.SELL".upper() or side.endswith("SELL")

    def _cash_aware_trim_rebalance_orders(
        self,
        orders: List[Order],
        account_cash: float,
    ) -> Tuple[List[Order], Dict[str, object]]:
        """
        Cash-aware buy trimming for verify only.
        Keep sell orders unchanged; trim BUY quantities by projected cash budget.
        """
        if not orders:
            return [], {
                "enabled": self.verify_cash_aware,
                "requested_orders": 0,
                "requested_buys": 0,
                "requested_sells": 0,
                "submitted_orders": 0,
                "submitted_buys": 0,
                "submitted_sells": 0,
                "trimmed_buy_count": 0,
                "skipped_buy_count": 0,
                "trim_events": [],
            }

        clean_orders: List[Order] = []
        sell_orders: List[Order] = []
        buy_orders: List[Order] = []
        for o in orders:
            qty = int(getattr(o, "quantity", 0) or 0)
            price = float(getattr(o, "price", 0.0) or 0.0)
            if qty <= 0 or price <= 0:
                continue
            clean_orders.append(o)
            if self._is_sell_order(o):
                sell_orders.append(o)
            elif self._is_buy_order(o):
                buy_orders.append(o)

        if not self.verify_cash_aware:
            submitted_buys = len([o for o in clean_orders if self._is_buy_order(o)])
            submitted_sells = len([o for o in clean_orders if self._is_sell_order(o)])
            return clean_orders, {
                "enabled": False,
                "requested_orders": len(clean_orders),
                "requested_buys": submitted_buys,
                "requested_sells": submitted_sells,
                "submitted_orders": len(clean_orders),
                "submitted_buys": submitted_buys,
                "submitted_sells": submitted_sells,
                "trimmed_buy_count": 0,
                "skipped_buy_count": 0,
                "trim_events": [],
            }

        projected_cash = float(account_cash)
        fee_rate = float(self.fee_rate)
        for s in sell_orders:
            projected_cash += float(s.quantity) * float(s.price) * (1.0 - fee_rate)

        buy_orders = sorted(buy_orders, key=lambda x: int(getattr(x, "rank", 999999) or 999999))
        remaining_buys = len(buy_orders)
        lot_size = 10
        trimmed_buy_count = 0
        skipped_buy_count = 0
        trim_events: List[Dict] = []
        trimmed_buys: List[Order] = []

        for b in buy_orders:
            if remaining_buys <= 0:
                break

            budget_per_order = projected_cash / float(max(remaining_buys, 1))
            per_share_cost = float(b.price) * (1.0 + fee_rate)

            max_shares_by_budget = int((budget_per_order / per_share_cost) / lot_size) * lot_size
            max_shares_by_cash = int((projected_cash / per_share_cost) / lot_size) * lot_size
            requested = int(b.quantity)
            allowed = min(requested, max_shares_by_budget, max_shares_by_cash)

            if allowed <= 0:
                skipped_buy_count += 1
                trim_events.append({
                    "code": b.code,
                    "rank": int(getattr(b, "rank", 0) or 0),
                    "requested_qty": requested,
                    "submitted_qty": 0,
                    "status": "skipped_due_cash",
                })
                remaining_buys -= 1
                continue

            if allowed < requested:
                trimmed_buy_count += 1
                trim_events.append({
                    "code": b.code,
                    "rank": int(getattr(b, "rank", 0) or 0),
                    "requested_qty": requested,
                    "submitted_qty": allowed,
                    "status": "trimmed_due_cash",
                })

            trimmed_buys.append(
                Order(
                    code=b.code,
                    name=b.name,
                    side=b.side,
                    quantity=allowed,
                    price=b.price,
                    reason=b.reason,
                    rank=int(getattr(b, "rank", 0) or 0),
                )
            )
            projected_cash -= float(allowed) * per_share_cost
            remaining_buys -= 1

        final_orders = sell_orders + trimmed_buys
        submitted_buys = len(trimmed_buys)
        submitted_sells = len(sell_orders)
        audit = {
            "enabled": True,
            "requested_orders": len(clean_orders),
            "requested_buys": len(buy_orders),
            "requested_sells": len(sell_orders),
            "submitted_orders": len(final_orders),
            "submitted_buys": submitted_buys,
            "submitted_sells": submitted_sells,
            "trimmed_buy_count": trimmed_buy_count,
            "skipped_buy_count": skipped_buy_count,
            "trim_events": trim_events,
        }
        return final_orders, audit

    def _generate_take_profit_orders(
        self,
        portfolio: CBPortfolioManager,
        date_idx: int,
    ) -> List[Order]:
        """涓?sim_runner 瀵归綈: 寮€鐩樿Е鍙戜紭鍏? 鍏舵鐩樹腑 high 瑙﹀彂銆?"""
        if self.take_profit_ratio <= 0 or date_idx <= 0:
            return []

        close = self.loader.raw_data_cache['CLOSE']
        open_ = self.loader.raw_data_cache['OPEN']
        high = self.loader.raw_data_cache['HIGH']

        tp_orders: List[Order] = []
        for pos in portfolio.get_all_positions():
            asset_idx = self.code_to_idx.get(pos.code)
            if asset_idx is None:
                continue

            prev_close = float(close[date_idx - 1, asset_idx].item())
            today_open = float(open_[date_idx, asset_idx].item())
            today_high = float(high[date_idx, asset_idx].item())
            if prev_close <= 0:
                continue

            tp_trigger = prev_close * (1 + self.take_profit_ratio)
            if today_open >= tp_trigger:
                tp_orders.append(Order(
                    code=pos.code,
                    name=pos.name,
                    side=OrderSide.SELL,
                    quantity=pos.shares,
                    price=today_open,
                    reason="TP_GAP_OPEN",
                    rank=0
                ))
            elif today_high >= tp_trigger:
                tp_orders.append(Order(
                    code=pos.code,
                    name=pos.name,
                    side=OrderSide.SELL,
                    quantity=pos.shares,
                    price=tp_trigger,
                    reason="TP_INTRADAY",
                    rank=0
                ))

        return tp_orders
        
    def run_simulation(self, initial_cash: Optional[float] = None) -> List[DailyRecord]:
        """杩愯浜嬩欢椹卞姩妯℃嫙锛圱鏃ラ€夎偂鈫扵鏃ュ缓浠撯啋T+1鏃ヨ绠楁敹鐩婏級"""
        resolved_initial_cash = float(initial_cash if initial_cash is not None else self.initial_cash)
        logger.info("\n" + "="*60)
        logger.info(f"RUNNING EVENT-DRIVEN SIMULATION (Initial Cash: {resolved_initial_cash:,.2f})")
        logger.info("="*60)
        
        account = SimAccount(initial_cash=resolved_initial_cash)
        records = []
        
        # Create temporary portfolio for simulation
        temp_portfolio_path = os.path.join(self.artifacts_dir, "temp_portfolio.json")
        portfolio = CBPortfolioManager(state_path=temp_portfolio_path)
        portfolio.clear_all()  # Start fresh
        
        # Create mock trader
        mock_trader = MockTrader()
        
        # Create runner with injected dependencies
        runner = CBStrategyRunner(
            loader=self.loader,
            portfolio=portfolio,
            trader=mock_trader,
            save_plan_enabled=False,
        )
        # Keep simulation formula strictly aligned with verifier formula
        # (especially important when --king is specified).
        runner.formula = self.formula
        runner.top_k = int(self.top_k)
        
        # Storage for detailed records
        daily_trades = []
        daily_holdings_detail = []
        
        prev_equity = account.initial_cash
        
        for i, date in enumerate(self.dates):
            logger.info(f"\n[{i+1}/{len(self.dates)}] Processing {date}...")
            
            # Get current prices for this date
            date_idx = self.loader.dates_list.index(date)
            prices = self._build_prices(date_idx)
            
            # 0) 姝㈢泩鍗栧嚭锛堝悓鏃ュ悗缁彲閫氳繃璋冧粨涔板洖锛?
            tp_orders = self._generate_take_profit_orders(portfolio, date_idx)
            tp_sells = []
            tp_fee = 0.0
            if tp_orders:
                _, tp_fee = self._execute_orders(account, portfolio, tp_orders, date)
                for order in tp_orders:
                    tp_sells.append({
                        'code': order.code,
                        'name': getattr(order, 'name', order.code),
                        'quantity': order.quantity,
                        'price': order.price,
                        'amount': order.quantity * order.price,
                        'reason': getattr(order, 'reason', 'TP')
                    })

            # 1) 璋冧粨璁㈠崟锛圱P 鍚庢寜褰撳墠鏉冪泭鍒嗛厤锛?
            runner.rebalancer.total_capital = account.get_equity(prices)
            mock_trader.orders = []
            runner.run(date=date, simulate=False)
            raw_orders = list(mock_trader.orders)
            orders, cash_audit = self._cash_aware_trim_rebalance_orders(
                raw_orders,
                account_cash=account.cash,
            )
            
            # Categorize orders
            requested_buys = []
            requested_sells = []
            for order in raw_orders:
                order_info = {
                    'code': order.code,
                    'name': getattr(order, 'name', order.code),
                    'quantity': order.quantity,
                    'price': order.price,
                    'amount': order.quantity * order.price
                }
                if self._is_buy_order(order):
                    requested_buys.append(order_info)
                elif self._is_sell_order(order):
                    requested_sells.append(order_info)

            buys = []
            sells = []
            for order in orders:
                order_info = {
                    'code': order.code,
                    'name': getattr(order, 'name', order.code),
                    'quantity': order.quantity,
                    'price': order.price,
                    'amount': order.quantity * order.price
                }
                if self._is_buy_order(order):
                    buys.append(order_info)
                elif self._is_sell_order(order):
                    sells.append(order_info)
            
            # Execute orders at T close
            if orders:
                logger.debug(f"  Executing {len(orders)} orders at T close...")
                logger.debug(f"    Sells: {len(sells)}, Buys: {len(buys)}")
                _, rebalance_fee = self._execute_orders(account, portfolio, orders, date)
                
                # CRITICAL: Validate holdings consistency between portfolio and SimAccount
                # Portfolio is used for signal generation, SimAccount for P&L calculation
                # They must be in sync to avoid "correct signal, wrong P&L" issues
                portfolio_positions = {p.code: p.shares for p in portfolio.get_all_positions()}
                simaccount_holdings = account.holdings
                
                if portfolio_positions != simaccount_holdings:
                    logger.warning(f"  鈿狅笍 Holdings mismatch detected!")
                    p_keys = set(portfolio_positions.keys())
                    s_keys = set(simaccount_holdings.keys())
                    logger.warning(f"    Only in Portfolio: {p_keys - s_keys}")
                    logger.warning(f"    Only in SimAccount: {s_keys - p_keys}")
                    
                    # Check share mismatches for common keys
                    for code in p_keys.intersection(s_keys):
                        if portfolio_positions[code] != simaccount_holdings[code]:
                            logger.warning(f"    Share mismatch for {code}: Portfolio={portfolio_positions[code]}, SimAccount={simaccount_holdings[code]}")
                    
                    # This is a critical issue that should be investigated
                    # For now, we log it but continue execution
            else:
                rebalance_fee = 0.0

            # Calculate return AFTER trading so TP execution impact and transaction
            # costs are reflected in the same day simulation return.
            current_equity_after_trade = account.get_equity(prices)
            daily_return = (current_equity_after_trade - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            total_tx_fee = tp_fee + rebalance_fee
            
            # Record trades
            daily_trades.append({
                'date': date,
                'tp_sells': tp_sells,
                'buys': buys,
                'sells': sells,
                'requested_buys': requested_buys,
                'requested_sells': requested_sells,
                'tx_fee_tp': tp_fee,
                'tx_fee_rebalance': rebalance_fee,
                'tx_fee_total': total_tx_fee,
                'cash_aware_enabled': bool(cash_audit.get('enabled', False)),
                'requested_orders_count': int(cash_audit.get('requested_orders', len(raw_orders))),
                'requested_buy_count': int(cash_audit.get('requested_buys', len(requested_buys))),
                'requested_sell_count': int(cash_audit.get('requested_sells', len(requested_sells))),
                'submitted_orders_count': int(cash_audit.get('submitted_orders', len(orders))),
                'submitted_buy_count': int(cash_audit.get('submitted_buys', len(buys))),
                'submitted_sell_count': int(cash_audit.get('submitted_sells', len(sells))),
                'trimmed_buy_count': int(cash_audit.get('trimmed_buy_count', 0)),
                'skipped_buy_count': int(cash_audit.get('skipped_buy_count', 0)),
                'buy_trim_events': list(cash_audit.get('trim_events', [])),
            })
            
            # Record holdings detail
            holdings_snapshot = {}
            # Prefer portfolio for rich info
            for pos in portfolio.get_all_positions():
                holdings_snapshot[pos.code] = {
                    'name': pos.name,
                    'shares': pos.shares,
                    'market_value': pos.shares * prices.get(pos.code, 0)
                }
            
            # Ensure complete coverage from account
            for code, shares in account.holdings.items():
                if code not in holdings_snapshot:
                    name = self.loader.names_dict.get(code, "Unknown")
                    holdings_snapshot[code] = {
                        'name': name,
                        'shares': shares,
                        'market_value': shares * prices.get(code, 0)
                    }

            daily_holdings_detail.append({
                'date': date,
                'holdings': holdings_snapshot,
                'cash': account.cash,
                'equity': current_equity_after_trade
            })
            
            # Record
            record = DailyRecord(
                date=date,
                sim_equity=current_equity_after_trade,
                sim_return=daily_return,
                sim_holdings=list(account.holdings.keys()),
                sim_cash=account.cash,
                orders_count=len(tp_orders) + len(orders),
                tx_fee=total_tx_fee,
            )
            records.append(record)
            
            logger.info(
                f"  Equity: {current_equity_after_trade:,.2f} | Return: {daily_return:+.4%} | "
                f"Holdings: {len(account.holdings)} | Cash: {account.cash:,.2f} | "
                f"TP Sells: {len(tp_orders)} | Rebalance Orders: {len(orders)} | "
                f"TxFee: {total_tx_fee:,.2f}"
            )
            
            prev_equity = current_equity_after_trade
        
        # Save detailed records
        self._save_detailed_records(daily_trades, daily_holdings_detail)
        self.daily_trades = daily_trades
        
        return records
    
    def _save_detailed_records(self, daily_trades, daily_holdings_detail):
        """淇濆瓨璇︾粏鐨勪氦鏄撳拰鎸佷粨璁板綍"""
        suffix = self._artifact_suffix()
        
        # Save trading records
        trades_path = os.path.join(self.artifacts_dir, f"daily_trades{suffix}.json")
        with open(trades_path, 'w', encoding='utf-8') as f:
            json.dump(daily_trades, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved trading records to: {trades_path}")
        
        # Save holdings records
        holdings_path = os.path.join(self.artifacts_dir, f"daily_holdings{suffix}.json")
        with open(holdings_path, 'w', encoding='utf-8') as f:
            json.dump(daily_holdings_detail, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved holdings records to: {holdings_path}")
    
    def run_backtest(self) -> Dict:
        """杩愯鍚戦噺鍖栧洖娴?"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING VECTOR BACKTEST")
        logger.info("="*60)
        
        # Execute formula on full data
        vm = StackVM()
        feat_tensor = self.loader.feat_tensor.to('cpu')
        factors = vm.execute(self.formula, feat_tensor)
        if factors is None:
            ok, _, reason = validate_formula(self.formula)
            reason_msg = reason if not ok else "VM returned None"
            formula_str = " ".join(self.formula)
            raise ValueError(
                f"Formula execution failed in vector backtest: {reason_msg}. "
                f"formula={formula_str}"
            )
        
        # Run backtest (if TP enabled, use TP-aware vector simulation)
        if self.take_profit_ratio > 0:
            result = self._evaluate_with_details_with_tp(factors)
        else:
            backtest = CBBacktest(top_k=int(self.top_k), fee_rate=float(self.fee_rate))
            result = backtest.evaluate_with_details(
                factors=factors,
                target_ret=self.loader.target_ret,
                valid_mask=self.loader.valid_mask
            )
        
        logger.info(f"Backtest Sharpe: {result['sharpe']:.4f}")
        logger.info(f"Backtest Cum Return: {result['cum_ret']:.4%}")

        # Also report metrics on the verification date range for easier comparison.
        period_returns = []
        for date in self.dates:
            date_idx = self.loader.dates_list.index(date)
            period_returns.append(result['daily_returns'][date_idx])
        if period_returns:
            period_arr = np.array(period_returns, dtype=float)
            period_cum_ret = (1 + period_arr).prod() - 1
            if len(period_arr) >= 2:
                period_sharpe = (
                    period_arr.mean() / (period_arr.std() + 1e-9) * np.sqrt(252)
                )
            else:
                period_sharpe = 0.0
            logger.info(f"Backtest Period Sharpe ({self.dates[0]}~{self.dates[-1]}): {period_sharpe:.4f}")
            logger.info(f"Backtest Period Cum Return ({self.dates[0]}~{self.dates[-1]}): {period_cum_ret:.4%}")
        
        return result

    def _evaluate_with_details_with_tp(self, factors: torch.Tensor) -> Dict:
        """
        TP 鍚戦噺鍥炴祴鏄庣粏:
        - t 鏃ユ寔浠撶殑鏀剁泭锛屼娇鐢? t+1 鏃ヤ环鏍艰绠?
        - 寮€鐩樿Е鍙? 浣跨敤 t+1 open 鏀剁泭
        - 鐩樹腑瑙﹀彂: 閿佸畾 take_profit_ratio (on t+1 intraday)
        - 褰撴棩涔板洖: 閫氳繃棰濆鍙岃竟璐圭巼鎵ｅ噺浣撶幇
        """
        if factors is None:
            raise ValueError("TP backtest received None factors (formula execution failed).")
        factors = factors.to('cpu')
        target_ret = self.loader.target_ret.to('cpu')
        valid_mask = self.loader.valid_mask.to('cpu')
        open_prices = self.loader.raw_data_cache['OPEN'].to('cpu')
        high_prices = self.loader.raw_data_cache['HIGH'].to('cpu')
        close_prices = self.loader.raw_data_cache['CLOSE'].to('cpu')

        backtest = CBBacktest(
            top_k=int(self.top_k),
            fee_rate=float(self.fee_rate),
            take_profit=self.take_profit_ratio
        )

        device = factors.device
        T, N = factors.shape
        weights, valid_trading_day, daily_valid_count, daily_holdings = build_topk_weights(
            factors=factors,
            valid_mask=valid_mask,
            top_k=backtest.top_k,
            min_valid_count=backtest.min_valid_count,
            clean_enabled=RobustConfig.SIGNAL_CLEAN_ENABLED,
            winsor_q=RobustConfig.SIGNAL_WINSOR_Q,
            clip_value=RobustConfig.SIGNAL_CLIP,
            rank_output=RobustConfig.SIGNAL_RANK_OUTPUT,
        )

        prev_weights = torch.roll(weights, 1, dims=0)
        prev_weights[0] = 0
        turnover = torch.abs(weights - prev_weights).sum(dim=1)
        tx_cost = turnover * backtest.fee_rate * 2

        effective_ret = target_ret.clone()
        tp_extra_cost = torch.zeros(T, device=device)

        # Time alignment for t -> t+1 return:
        # - weights[t] represents holdings decided at t close
        # - TP should be checked on next day (t+1) open/high against close[t]
        prev_close = close_prices.clone()
        next_open = torch.roll(open_prices, -1, dims=0)
        next_high = torch.roll(high_prices, -1, dims=0)
        next_open[-1] = 0
        next_high[-1] = 0

        valid_price_mask = (
            (prev_close > 0) & (prev_close < 10000) &
            (next_open > 0) & (next_open < 10000) &
            (next_high > 0) & (next_high < 10000)
        )
        tp_trigger_price = prev_close * (1 + self.take_profit_ratio)

        holding_mask = weights > 0
        open_gap_up = (next_open >= tp_trigger_price) & valid_price_mask
        gap_up_mask = open_gap_up & holding_mask
        intraday_tp = (next_high >= tp_trigger_price) & (~open_gap_up) & valid_price_mask
        intra_tp_mask = intraday_tp & holding_mask

        open_ret = (next_open / prev_close) - 1.0
        effective_ret[gap_up_mask] = open_ret[gap_up_mask]
        effective_ret[intra_tp_mask] = self.take_profit_ratio

        daily_k = holding_mask.sum(dim=1).float()
        gap_up_count = gap_up_mask.sum(dim=1).float()
        intra_tp_count = intra_tp_mask.sum(dim=1).float()
        safe_k = torch.where(daily_k > 0, daily_k, torch.ones_like(daily_k))
        tp_extra_cost += (gap_up_count + intra_tp_count) * 2 * backtest.fee_rate / safe_k

        gross_ret = (weights * effective_ret).sum(dim=1)
        net_ret = gross_ret - tx_cost - tp_extra_cost
        valid_net_ret = net_ret[valid_trading_day]

        if len(valid_net_ret) < 10:
            return {
                'reward': -10.0,
                'cum_ret': 0.0,
                'sharpe': 0.0,
                'daily_holdings': daily_holdings,
                'daily_returns': net_ret.tolist()
            }

        cum_ret = (1 + valid_net_ret).prod() - 1
        mean_ret = valid_net_ret.mean()
        std_ret = valid_net_ret.std() + 1e-9
        sharpe = mean_ret / std_ret * (252 ** 0.5)

        valid_turnover = turnover[valid_trading_day]
        avg_turnover = valid_turnover.mean()
        turnover_penalty = torch.clamp(avg_turnover - 0.3, min=0) * 2

        avg_holding = (weights > 0).sum(dim=1).float()[valid_trading_day].mean()
        activity_penalty = 5.0 if avg_holding < backtest.top_k * 0.5 else 0.0

        reward = sharpe * 10 - turnover_penalty - activity_penalty
        if torch.isnan(reward) or torch.isinf(reward):
            reward = torch.tensor(-10.0, device=device)

        return {
            'reward': reward.item() if hasattr(reward, 'item') else reward,
            'cum_ret': cum_ret.item() if hasattr(cum_ret, 'item') else float(cum_ret),
            'sharpe': sharpe.item() if hasattr(sharpe, 'item') else float(sharpe),
            'daily_holdings': daily_holdings,
            'daily_returns': net_ret.tolist()
        }
    
    def compare_results(self, sim_records: List[DailyRecord], backtest_result: Dict):
        """
        瀵规瘮缁撴灉
        
        CRITICAL ASSUMPTIONS:
        1. Simulation: Orders executed at T close, returns calculated on T+1
           - sim_return[i] = (equity[i] - equity[i-1]) / equity[i-1]
           - Represents return from holdings established on day i-1, realized on day i
        
        2. Backtest: Holdings selected on day T, returns realized on T+1
           - backtest_return[t] = target_ret[t] = (close[t+1] / close[t]) - 1
           - Represents return from holdings selected on day t, realized on day t+1
        
        3. Alignment: sim[i+1] should match backtest[i]
           - sim[1] (day 1 return from day 0 holdings) = backtest[0] (day 0鈫? return)
        """
        logger.info("\n" + "="*60)
        logger.info("COMPARISON ANALYSIS")
        logger.info("="*60)
        
        # Store for performance metrics calculation
        self.sim_records = sim_records
        
        # Extract simulation returns
        sim_returns = [r.sim_return for r in sim_records]
        
        # Extract backtest returns (align dates)
        backtest_returns = []
        for date in self.dates:
            date_idx = self.loader.dates_list.index(date)
            backtest_returns.append(backtest_result['daily_returns'][date_idx])
        
        # Validate assumptions
        assert len(sim_returns) == len(self.dates), \
            f"Simulation returns ({len(sim_returns)}) must match dates ({len(self.dates)})"
        assert len(backtest_returns) == len(self.dates), \
            f"Backtest returns ({len(backtest_returns)}) must match dates ({len(self.dates)})"
        
        if len(sim_returns) <= 1:
            logger.error("Insufficient simulation data for comparison")
            return False
        
        # Align: sim[1:] vs backtest[:-1]
        # This implements: sim[i+1] = backtest[i]
        sim_returns_aligned = sim_returns[1:]  # Skip day 0 (no prior holdings)
        backtest_returns_aligned = backtest_returns[:-1]  # Days 0 to N-2
        aligned_dates = self.dates[:-1]  # Dates corresponding to backtest returns
        
        # Ensure same length
        min_len = min(len(sim_returns_aligned), len(backtest_returns_aligned))
        sim_returns_aligned = sim_returns_aligned[:min_len]
        backtest_returns_aligned = backtest_returns_aligned[:min_len]
        aligned_dates = aligned_dates[:min_len]
        
        logger.info(f"Comparing {len(sim_returns_aligned)} aligned returns...")
        logger.info(f"Alignment: sim[1:{min_len+1}] vs backtest[0:{min_len}]")
        
        # Calculate metrics
        sim_returns_arr = np.array(sim_returns_aligned)
        backtest_returns_arr = np.array(backtest_returns_aligned)
        
        # 1. Return difference
        diff = sim_returns_arr - backtest_returns_arr
        mae = np.abs(diff).mean()
        max_abs_error = np.abs(diff).max()
        
        # 2. Correlation
        if len(sim_returns_arr) >= 2:
            correlation = np.corrcoef(sim_returns_arr, backtest_returns_arr)[0, 1]
        else:
            correlation = float("nan")
        
        # 3. Cumulative returns
        sim_cum_ret = (1 + sim_returns_arr).prod() - 1
        backtest_cum_ret = (1 + backtest_returns_arr).prod() - 1
        
        logger.info(f"Return MAE: {mae:.6f} (target: < 1e-4)")
        logger.info(f"Max Abs Error: {max_abs_error:.6f} (target: < 1e-3)")
        logger.info(f"Correlation: {correlation:.6f} (target: > 0.99)")
        logger.info(f"Sim Cum Return: {sim_cum_ret:.4%}")
        logger.info(f"Backtest Cum Return: {backtest_cum_ret:.4%}")

        benchmark_returns_aligned = self._load_benchmark_returns(aligned_dates)
        
        # Save to CSV
        df = pd.DataFrame({
            'Date': aligned_dates,
            'Sim_Return': sim_returns_aligned,
            'Backtest_Return': backtest_returns_aligned,
            'Benchmark_Return': benchmark_returns_aligned,
            'Diff': diff,
            'Sim_Equity': [sim_records[i+1].sim_equity for i in range(min_len)]
        })
        suffix = self._artifact_suffix()
        csv_path = os.path.join(self.artifacts_dir, f"daily_returns{suffix}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved daily returns to: {csv_path}")

        # Generate visualization artifacts from existing comparison outputs.
        aligned_rebalance_counts = self._extract_aligned_rebalance_counts(min_len)
        self._generate_visual_outputs(df, aligned_rebalance_counts)
        
        # Generate report
        self.generate_report(mae, max_abs_error, correlation, sim_cum_ret, backtest_cum_ret)
        
        # Check pass/fail
        passed = (mae < 1e-4 and correlation > 0.99)
        if passed:
            logger.info("\nVERIFICATION PASSED")
        else:
            logger.warning("\nVERIFICATION FAILED - Review metrics above")
        
        return passed

    def _generate_visual_outputs(
        self,
        compare_df: pd.DataFrame,
        aligned_rebalance_counts: Optional[List[int]] = None,
    ) -> None:
        """Generate post-backtest visualization artifacts without changing verify logic."""
        suffix = self._artifact_suffix()
        try:
            strategy_label = self.strategy_name or (self.strategy_id or "legacy")
            self.visual_outputs = generate_verification_visuals(
                compare_df=compare_df,
                output_dir=self.artifacts_dir,
                suffix=suffix,
                strategy_name=strategy_label,
                top_k=int(self.top_k),
                rebalance_counts=aligned_rebalance_counts,
            )
            logger.info(
                "Saved visualization artifacts: %s",
                ", ".join(self.visual_outputs.values()),
            )
        except Exception as exc:
            logger.exception("Failed to generate visualization artifacts: %s", exc)

    def _extract_aligned_rebalance_counts(self, min_len: int) -> List[int]:
        """Build aligned daily rebalance counts from simulation trade records."""
        if min_len <= 0:
            return []
        if not self.daily_trades:
            return [0] * min_len

        aligned = self.daily_trades[1 : 1 + min_len]
        counts: List[int] = []
        for day_trade in aligned:
            buys = day_trade.get("buys", []) if isinstance(day_trade, dict) else []
            sells = day_trade.get("sells", []) if isinstance(day_trade, dict) else []
            buy_codes = {
                item.get("code")
                for item in buys
                if isinstance(item, dict) and item.get("code")
            }
            sell_codes = {
                item.get("code")
                for item in sells
                if isinstance(item, dict) and item.get("code")
            }
            counts.append(max(len(buy_codes), len(sell_codes)))

        if len(counts) < min_len:
            counts.extend([0] * (min_len - len(counts)))
        return counts[:min_len]

    def _load_benchmark_returns(self, aligned_dates: List[str]) -> List[float]:
        """Load benchmark daily returns aligned to next trading day (t -> t+1)."""
        if not aligned_dates:
            return []

        default_returns = [0.0] * len(aligned_dates)
        index_path = os.path.join(project_root, "data", "index.pq")
        if not os.path.exists(index_path):
            logger.warning("Benchmark file not found: %s", index_path)
            return default_returns

        try:
            idx_df = pd.read_parquet(index_path)
        except Exception as exc:
            logger.warning("Failed to read benchmark file %s: %s", index_path, exc)
            return default_returns

        try:
            if "trade_date" in idx_df.columns:
                idx_df["trade_date"] = pd.to_datetime(idx_df["trade_date"], errors="coerce")
                idx_df = idx_df.dropna(subset=["trade_date"]).set_index("trade_date")
            else:
                idx_df.index = pd.to_datetime(idx_df.index, errors="coerce")
                idx_df = idx_df[~idx_df.index.isna()]

            idx_df = idx_df.sort_index()

            if "index_jsl" in idx_df.columns:
                bench_ret = pd.to_numeric(idx_df["index_jsl"], errors="coerce")
            elif "equity_jsl" in idx_df.columns:
                bench_equity = pd.to_numeric(idx_df["equity_jsl"], errors="coerce")
                bench_ret = bench_equity.pct_change()
                logger.warning("index_jsl not found, fallback to equity_jsl pct_change().")
            else:
                logger.warning("Neither index_jsl nor equity_jsl found in %s", index_path)
                return default_returns

            bench_ret = bench_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            bench_ret.index = pd.to_datetime(bench_ret.index, errors="coerce")
            bench_ret = bench_ret[~bench_ret.index.isna()]
            bench_ret = bench_ret.groupby(bench_ret.index.normalize()).last()

            aligned_idx = pd.to_datetime(aligned_dates, errors="coerce")
            strategy_dates = pd.to_datetime(getattr(self, "dates", []), errors="coerce")
            next_trade_map: Dict[pd.Timestamp, pd.Timestamp] = {}
            for i in range(len(strategy_dates) - 1):
                cur = strategy_dates[i]
                nxt = strategy_dates[i + 1]
                if pd.isna(cur) or pd.isna(nxt):
                    continue
                next_trade_map[cur.normalize()] = nxt.normalize()

            bench_index = bench_ret.index
            aligned_returns: List[float] = []
            missing_count = 0
            for dt in aligned_idx:
                if pd.isna(dt):
                    aligned_returns.append(0.0)
                    missing_count += 1
                    continue

                d0 = dt.normalize()
                target_dt = next_trade_map.get(d0)
                if target_dt is None:
                    pos = bench_index.searchsorted(d0, side="right")
                    if pos < len(bench_index):
                        target_dt = bench_index[pos]

                if target_dt is None:
                    aligned_returns.append(0.0)
                    missing_count += 1
                    continue

                v = bench_ret.get(target_dt, np.nan)
                if pd.isna(v):
                    aligned_returns.append(0.0)
                    missing_count += 1
                else:
                    aligned_returns.append(float(v))

            if missing_count > 0:
                logger.warning(
                    "Benchmark alignment has %d missing dates (filled with 0.0).",
                    missing_count,
                )
            return aligned_returns
        except Exception as exc:
            logger.warning("Failed to align benchmark returns: %s", exc)
            return default_returns
    
    def generate_report(self, mae, max_abs_error, correlation, sim_cum_ret, backtest_cum_ret):
        """Generate verification report markdown."""
        suffix = self._artifact_suffix()
        report_path = os.path.join(self.artifacts_dir, f"verification_report{suffix}.md")
        
        # Calculate additional performance metrics
        sim_returns = [r.sim_return for r in self.sim_records] if hasattr(self, 'sim_records') else []
        
        if len(sim_returns) > 1:
            sim_returns_arr = np.array(sim_returns[1:])  # Skip first day (no prior holdings)
            
            # Sharpe Ratio (annualized)
            mean_ret = sim_returns_arr.mean()
            std_ret = sim_returns_arr.std()
            sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252)
            
            # Annualized Return
            trading_days = len(sim_returns_arr)
            ann_ret = (1 + sim_cum_ret) ** (252 / trading_days) - 1
            
            # Max Drawdown
            equity_curve = [100000.0]  # Initial
            for ret in sim_returns_arr:
                equity_curve.append(equity_curve[-1] * (1 + ret))
            
            peak = equity_curve[0]
            max_dd = 0.0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            sharpe = 0.0
            ann_ret = 0.0
            max_dd = 0.0
            sim_returns_arr = np.array([], dtype=float)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Strategy Verification Report\n\n")
            f.write(f"**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(
                f"**Date Range**: {self.dates[0]} to {self.dates[-1]} "
                f"({len(self.dates)} trading days)\n\n"
            )

            f.write("## Parameter Alignment\n")
            f.write(f"- STRATEGY_ID: {self.strategy_id or 'legacy'}\n")
            f.write(f"- STRATEGY_NAME: {self.strategy_name or 'legacy'}\n")
            f.write(f"- TOP_K: {self.top_k}\n")
            f.write(f"- FEE_RATE: {self.fee_rate} (single-sided)\n")
            f.write(f"- TAKE_PROFIT_RATIO: {self.take_profit_ratio}\n")
            f.write(f"- VERIFY_CASH_AWARE: {self.verify_cash_aware}\n")
            f.write(f"- INITIAL_CASH: {self.initial_cash}\n")
            f.write(f"- FORMULA_SOURCE: {self.formula_source}\n\n")

            f.write("## Performance Metrics\n\n")
            f.write("### Event-Driven Simulation\n\n")
            f.write(f"- **Cumulative Return**: {sim_cum_ret:.4%}\n")
            f.write(f"- **Annualized Return**: {ann_ret:.4%}\n")
            f.write(f"- **Sharpe Ratio**: {sharpe:.4f}\n")
            f.write(f"- **Max Drawdown**: {max_dd:.4%}\n")
            f.write(f"- **Trading Days**: {len(sim_returns_arr)}\n\n")

            f.write("### Vector Backtest Baseline\n\n")
            f.write(f"- **Cumulative Return**: {backtest_cum_ret:.4%}\n\n")

            f.write("## Consistency Metrics\n\n")
            f.write("| Metric | Actual | Target | Status |\n")
            f.write("|:---|:---|:---|:---|\n")
            f.write(f"| Return MAE | {mae:.6f} | < 1e-4 | {'PASS' if mae < 1e-4 else 'FAIL'} |\n")
            f.write(f"| Max Abs Error | {max_abs_error:.6f} | < 1e-3 | {'PASS' if max_abs_error < 1e-3 else 'FAIL'} |\n")
            f.write(f"| Correlation | {correlation:.6f} | > 0.99 | {'PASS' if correlation > 0.99 else 'FAIL'} |\n\n")

            if self.visual_outputs:
                f.write("## Visualization Outputs\n\n")
                for name, path in self.visual_outputs.items():
                    rel_path = os.path.relpath(path, project_root)
                    f.write(f"- **{name}**: `{rel_path}`\n")
                f.write("\n")
            
            f.write("## Conclusion\n\n")
            if mae < 1e-4 and correlation > 0.99:
                f.write("PASS: Event-driven simulation is highly aligned with vector backtest.\n")
            else:
                f.write("WARNING: Alignment gap remains. Review TP handling, lot rounding and transaction cost path dependency.\n")
        
        logger.info(f"Saved report to: {report_path}")


def _compute_return_metrics(daily_returns: List[float]) -> Dict[str, float]:
    """Compute cumulative/annualized/sharpe/max-drawdown metrics."""
    arr = np.array(daily_returns, dtype=float)
    if arr.size == 0:
        return {
            "cum_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trading_days": 0.0,
        }

    nav = np.cumprod(1.0 + arr)
    cum_return = float(nav[-1] - 1.0)
    annualized_return = float((1.0 + cum_return) ** (252.0 / arr.size) - 1.0)
    sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252.0))
    peak = np.maximum.accumulate(nav)
    drawdown = nav / (peak + 1e-12) - 1.0
    max_drawdown = float(abs(drawdown.min()))
    return {
        "cum_return": cum_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trading_days": float(arr.size),
    }


def _combine_strategy_trades(trades_by_strategy: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Merge per-strategy daily trades by date (simple concatenation)."""
    date_map: Dict[str, Dict] = {}
    rebalance_count_map: Dict[str, int] = {}

    for sid, daily_trades in trades_by_strategy.items():
        for rec in daily_trades:
            date = rec.get("date")
            if not date:
                continue
            if date not in date_map:
                date_map[date] = {
                    "date": date,
                    "tp_sells": [],
                    "buys": [],
                    "sells": [],
                    "tx_fee_tp": 0.0,
                    "tx_fee_rebalance": 0.0,
                    "tx_fee_total": 0.0,
                }

            buy_codes = set()
            sell_codes = set()
            for b in rec.get("buys", []):
                b2 = dict(b)
                b2["strategy_id"] = sid
                date_map[date]["buys"].append(b2)
                code = b2.get("code")
                if code:
                    buy_codes.add(code)
            for s in rec.get("sells", []):
                s2 = dict(s)
                s2["strategy_id"] = sid
                date_map[date]["sells"].append(s2)
                code = s2.get("code")
                if code:
                    sell_codes.add(code)
            for t in rec.get("tp_sells", []):
                t2 = dict(t)
                t2["strategy_id"] = sid
                date_map[date]["tp_sells"].append(t2)

            date_map[date]["tx_fee_tp"] += float(rec.get("tx_fee_tp", 0.0) or 0.0)
            date_map[date]["tx_fee_rebalance"] += float(rec.get("tx_fee_rebalance", 0.0) or 0.0)
            date_map[date]["tx_fee_total"] += float(rec.get("tx_fee_total", 0.0) or 0.0)

            rebalance_count_map[date] = rebalance_count_map.get(date, 0) + max(
                len(buy_codes), len(sell_codes)
            )

    merged = [date_map[d] for d in sorted(date_map.keys())]
    return {"daily_trades": merged, "rebalance_count_map": rebalance_count_map}


def _combine_strategy_holdings(holdings_by_strategy: Dict[str, List[Dict]]) -> List[Dict]:
    """Merge per-strategy daily holdings by date (simple additive merge)."""
    date_map: Dict[str, Dict] = {}
    for sid, daily_holdings in holdings_by_strategy.items():
        for rec in daily_holdings:
            date = rec.get("date")
            if not date:
                continue
            if date not in date_map:
                date_map[date] = {
                    "date": date,
                    "holdings": {},
                    "cash": 0.0,
                    "equity": 0.0,
                }

            date_map[date]["cash"] += float(rec.get("cash", 0.0) or 0.0)
            date_map[date]["equity"] += float(rec.get("equity", 0.0) or 0.0)

            for code, item in (rec.get("holdings", {}) or {}).items():
                if code not in date_map[date]["holdings"]:
                    base = dict(item) if isinstance(item, dict) else {}
                    base["strategy_ids"] = [sid]
                    date_map[date]["holdings"][code] = base
                else:
                    existing = date_map[date]["holdings"][code]
                    existing["shares"] = float(existing.get("shares", 0.0) or 0.0) + float(
                        (item or {}).get("shares", 0.0) or 0.0
                    )
                    existing["market_value"] = float(
                        existing.get("market_value", 0.0) or 0.0
                    ) + float((item or {}).get("market_value", 0.0) or 0.0)
                    existing.setdefault("strategy_ids", []).append(sid)

    return [date_map[d] for d in sorted(date_map.keys())]


def _build_daily_holding_code_map(daily_holdings: List[Dict]) -> Dict[str, set]:
    """Build date -> holding code set map from daily holdings artifact."""
    code_map: Dict[str, set] = {}
    for rec in daily_holdings:
        date = rec.get("date")
        if not date:
            continue
        holdings = rec.get("holdings", {}) or {}
        if isinstance(holdings, dict):
            codes = {str(code) for code in holdings.keys() if code}
        else:
            codes = set()
        code_map[str(date)] = codes
    return code_map


def _build_multi_strategy_summary_rows(
    merged_returns_df: pd.DataFrame,
    valid_runs: List[Dict],
    holdings_by_strategy: Dict[str, List[Dict]],
) -> List[Tuple[str, float]]:
    """Build extra summary metrics for multi-strategy combined portfolio."""
    extra_rows: List[Tuple[str, float]] = []
    if merged_returns_df.empty or len(valid_runs) < 2:
        return extra_rows

    sid_to_col: Dict[str, str] = {}
    for run in valid_runs:
        sid = str(run.get("strategy_id"))
        sid_key = sid.replace("-", "_")
        col = f"Sim_Return__{sid_key}"
        if col in merged_returns_df.columns:
            sid_to_col[sid] = col

    strategy_ids = [str(r.get("strategy_id")) for r in valid_runs if str(r.get("strategy_id")) in sid_to_col]
    if len(strategy_ids) < 2:
        return extra_rows

    n_days = len(merged_returns_df)
    for sid in strategy_ids:
        arr = (
            pd.to_numeric(merged_returns_df[sid_to_col[sid]], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if arr.size == 0:
            ann = 0.0
            sharpe = 0.0
        else:
            total_ret = float(np.prod(1.0 + arr) - 1.0)
            ann = float((1.0 + total_ret) ** (252.0 / max(n_days, 1)) - 1.0)
            sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252.0))
        extra_rows.append((f"Component {sid} Annualized Return", ann))
        extra_rows.append((f"Component {sid} Sharpe Ratio", sharpe))

    corr_values: List[float] = []
    for sid_a, sid_b in combinations(strategy_ids, 2):
        a = (
            pd.to_numeric(merged_returns_df[sid_to_col[sid_a]], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        b = (
            pd.to_numeric(merged_returns_df[sid_to_col[sid_b]], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if a.size >= 2 and b.size >= 2:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = float("nan")
        extra_rows.append((f"Pair Return Corr [{sid_a} vs {sid_b}]", corr))
        if np.isfinite(corr):
            corr_values.append(corr)
    if corr_values:
        extra_rows.append(("Avg Pair Return Corr", float(np.mean(corr_values))))

    holding_map_by_sid = {
        sid: _build_daily_holding_code_map(holdings_by_strategy.get(sid, []))
        for sid in strategy_ids
    }
    overlap_values: List[float] = []
    for sid_a, sid_b in combinations(strategy_ids, 2):
        map_a = holding_map_by_sid.get(sid_a, {})
        map_b = holding_map_by_sid.get(sid_b, {})
        all_dates = sorted(set(map_a.keys()) | set(map_b.keys()))
        pair_daily_overlap: List[float] = []
        for d in all_dates:
            a_set = map_a.get(d, set())
            b_set = map_b.get(d, set())
            union = a_set | b_set
            if not union:
                continue
            pair_daily_overlap.append(len(a_set & b_set) / float(len(union)))
        overlap = float(np.mean(pair_daily_overlap)) if pair_daily_overlap else float("nan")
        extra_rows.append((f"Pair Holding Overlap [{sid_a} vs {sid_b}]", overlap))
        if np.isfinite(overlap):
            overlap_values.append(overlap)
    if overlap_values:
        extra_rows.append(("Avg Pair Holding Overlap", float(np.mean(overlap_values))))

    return extra_rows


def generate_multi_strategy_artifacts(
    run_infos: List[Dict],
    artifacts_dir: str,
    start_date: str,
    end_date: Optional[str],
) -> Dict[str, str]:
    """Build merged portfolio artifacts from multiple single-strategy outputs."""
    suffix_end = end_date or "end"
    combo_suffix = f"_multi_combo_{start_date}_{suffix_end}"

    valid_runs: List[Dict] = []
    return_frames: List[pd.DataFrame] = []
    trades_by_strategy: Dict[str, List[Dict]] = {}
    holdings_by_strategy: Dict[str, List[Dict]] = {}

    for info in run_infos:
        sid = info.get("strategy_id")
        suffix = info.get("suffix")
        if not sid or not suffix:
            continue
        returns_path = os.path.join(artifacts_dir, f"daily_returns{suffix}.csv")
        trades_path = os.path.join(artifacts_dir, f"daily_trades{suffix}.json")
        holdings_path = os.path.join(artifacts_dir, f"daily_holdings{suffix}.json")
        if not os.path.exists(returns_path):
            logger.warning("Skip strategy %s: missing returns artifact %s", sid, returns_path)
            continue
        if not os.path.exists(trades_path):
            logger.warning("Skip strategy %s: missing trades artifact %s", sid, trades_path)
            continue
        if not os.path.exists(holdings_path):
            logger.warning("Skip strategy %s: missing holdings artifact %s", sid, holdings_path)
            continue

        df = pd.read_csv(returns_path)
        required = {"Date", "Sim_Return", "Backtest_Return"}
        if not required.issubset(set(df.columns)):
            logger.warning("Skip strategy %s: returns columns missing in %s", sid, returns_path)
            continue

        sid_key = str(sid).replace("-", "_")
        keep_cols = ["Date", "Sim_Return", "Backtest_Return"]
        if "Benchmark_Return" in df.columns:
            keep_cols.append("Benchmark_Return")
        sdf = df[keep_cols].copy()
        rename_map = {
            "Sim_Return": f"Sim_Return__{sid_key}",
            "Backtest_Return": f"Backtest_Return__{sid_key}",
        }
        if "Benchmark_Return" in sdf.columns:
            rename_map["Benchmark_Return"] = f"Benchmark_Return__{sid_key}"
        sdf = sdf.rename(columns=rename_map)
        return_frames.append(sdf)

        with open(trades_path, "r", encoding="utf-8") as f:
            trades_by_strategy[str(sid)] = json.load(f)
        with open(holdings_path, "r", encoding="utf-8") as f:
            holdings_by_strategy[str(sid)] = json.load(f)
        valid_runs.append(info)

    if len(valid_runs) < 2:
        raise ValueError("Need at least two valid strategy outputs for multi-strategy combine.")

    merged = return_frames[0]
    for frame in return_frames[1:]:
        merged = merged.merge(frame, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping dates across strategy return artifacts.")

    sim_cols = [c for c in merged.columns if c.startswith("Sim_Return__")]
    backtest_cols = [c for c in merged.columns if c.startswith("Backtest_Return__")]
    benchmark_cols = [c for c in merged.columns if c.startswith("Benchmark_Return__")]
    if not sim_cols or not backtest_cols:
        raise ValueError("Missing merged return columns for multi-strategy combine.")

    combined_df = pd.DataFrame({"Date": merged["Date"]})
    combined_df["Sim_Return"] = merged[sim_cols].mean(axis=1)
    combined_df["Backtest_Return"] = merged[backtest_cols].mean(axis=1)
    if benchmark_cols:
        combined_df["Benchmark_Return"] = merged[benchmark_cols].mean(axis=1)
    else:
        combined_df["Benchmark_Return"] = 0.0
    combined_df["Diff"] = combined_df["Sim_Return"] - combined_df["Backtest_Return"]

    total_initial_cash = sum(float(r.get("initial_cash", 0.0) or 0.0) for r in valid_runs)
    if total_initial_cash <= 0:
        total_initial_cash = 100000.0 * len(valid_runs)
    combined_df["Sim_Equity"] = total_initial_cash * (1.0 + combined_df["Sim_Return"]).cumprod()

    combine_trades_result = _combine_strategy_trades(trades_by_strategy)
    combined_trades = combine_trades_result["daily_trades"]
    rebalance_count_map = combine_trades_result["rebalance_count_map"]
    combined_holdings = _combine_strategy_holdings(holdings_by_strategy)
    aligned_rebalance_counts = [
        int(rebalance_count_map.get(d, 0)) for d in combined_df["Date"].tolist()
    ]

    combo_ids = [str(r.get("strategy_id")) for r in valid_runs]
    combo_top_k = sum(int(r.get("top_k", 0) or 0) for r in valid_runs)
    combo_name = "multi_combo(" + ",".join(combo_ids) + ")"
    extra_summary_rows = _build_multi_strategy_summary_rows(
        merged_returns_df=merged,
        valid_runs=valid_runs,
        holdings_by_strategy=holdings_by_strategy,
    )
    visual_outputs = generate_verification_visuals(
        compare_df=combined_df,
        output_dir=artifacts_dir,
        suffix=combo_suffix,
        strategy_name=combo_name,
        top_k=combo_top_k if combo_top_k > 0 else None,
        rebalance_counts=aligned_rebalance_counts,
        extra_summary_rows=extra_summary_rows,
    )

    returns_path = os.path.join(artifacts_dir, f"daily_returns{combo_suffix}.csv")
    trades_path = os.path.join(artifacts_dir, f"daily_trades{combo_suffix}.json")
    holdings_path = os.path.join(artifacts_dir, f"daily_holdings{combo_suffix}.json")
    report_path = os.path.join(artifacts_dir, f"verification_report{combo_suffix}.md")

    combined_df.to_csv(returns_path, index=False)
    with open(trades_path, "w", encoding="utf-8") as f:
        json.dump(combined_trades, f, indent=2, ensure_ascii=False)
    with open(holdings_path, "w", encoding="utf-8") as f:
        json.dump(combined_holdings, f, indent=2, ensure_ascii=False)

    sim_metrics = _compute_return_metrics(combined_df["Sim_Return"].tolist())
    backtest_metrics = _compute_return_metrics(combined_df["Backtest_Return"].tolist())
    benchmark_metrics = _compute_return_metrics(combined_df["Benchmark_Return"].tolist())
    mae = float(np.abs(combined_df["Diff"].to_numpy(dtype=float)).mean())
    max_abs_error = float(np.abs(combined_df["Diff"].to_numpy(dtype=float)).max())
    if len(combined_df) >= 2:
        correlation = float(
            np.corrcoef(
                combined_df["Sim_Return"].to_numpy(dtype=float),
                combined_df["Backtest_Return"].to_numpy(dtype=float),
            )[0, 1]
        )
    else:
        correlation = float("nan")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Multi-Strategy Verification Report\n\n")
        f.write(f"**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Date Range**: {start_date} to {suffix_end}\n\n")
        f.write("## Combined Strategies\n\n")
        for r in valid_runs:
            f.write(
                f"- `{r.get('strategy_id')}` | "
                f"top_k={r.get('top_k')} | initial_cash={r.get('initial_cash')}\n"
            )
        f.write("\n")

        f.write("## Combined Performance\n\n")
        f.write(f"- **Cumulative Return**: {sim_metrics['cum_return']:.4%}\n")
        f.write(f"- **Annualized Return**: {sim_metrics['annualized_return']:.4%}\n")
        f.write(f"- **Sharpe Ratio**: {sim_metrics['sharpe']:.4f}\n")
        f.write(f"- **Max Drawdown**: {sim_metrics['max_drawdown']:.4%}\n")
        f.write(f"- **Benchmark Cumulative Return**: {benchmark_metrics['cum_return']:.4%}\n")
        f.write(
            f"- **Excess Return vs Benchmark**: "
            f"{(sim_metrics['cum_return'] - benchmark_metrics['cum_return']):.4%}\n"
        )
        f.write("\n")

        f.write("## Consistency Metrics (Combined Sim vs Combined Backtest)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|:---|:---|\n")
        f.write(f"| Return MAE | {mae:.6f} |\n")
        f.write(f"| Max Abs Error | {max_abs_error:.6f} |\n")
        f.write(f"| Correlation | {correlation:.6f} |\n\n")

        f.write("## Objective Check (Drawdown / Sharpe)\n\n")
        f.write(
            "Use this combined report and visualization to compare whether "
            "portfolio combination lowers drawdown and improves Sharpe.\n\n"
        )

        f.write("## Artifacts\n\n")
        f.write(f"- `daily_returns`: `{os.path.relpath(returns_path, project_root)}`\n")
        f.write(f"- `daily_trades`: `{os.path.relpath(trades_path, project_root)}`\n")
        f.write(f"- `daily_holdings`: `{os.path.relpath(holdings_path, project_root)}`\n")
        f.write(f"- `verification_report`: `{os.path.relpath(report_path, project_root)}`\n")
        for k, v in visual_outputs.items():
            f.write(f"- `{k}`: `{os.path.relpath(v, project_root)}`\n")
        f.write("\n")

    return {
        "daily_returns_csv": returns_path,
        "daily_trades_json": trades_path,
        "daily_holdings_json": holdings_path,
        "verification_report_md": report_path,
        **visual_outputs,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Strategy verification script')
    parser.add_argument('--start', type=str, default='2024-01-01', help='start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='2024-12-31', help='end date YYYY-MM-DD')
    parser.add_argument(
        '--strategies-config',
        type=str,
        default=None,
        help='strategy config path, default strategy_manager/strategies_config.json'
    )
    parser.add_argument(
        '--strategy-id',
        type=str,
        default=None,
        help='strategy id from strategy config; if omitted, verify all enabled strategies'
    )
    parser.add_argument('--king', type=int, default=None,
                        help='legacy king step id (cannot combine with --strategy-id)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='override top_k')
    parser.add_argument('--fee-rate', type=float, default=None,
                        help='override fee rate (single-sided)')
    parser.add_argument('--initial-cash', type=float, default=None,
                        help='initial cash override; default uses strategy config initial_capital')
    parser.add_argument('--take-profit', type=float, default=None,
                        help='take profit ratio override, e.g. 0.06 means +6%%')
    parser.add_argument(
        '--verify-cash-aware',
        dest='verify_cash_aware',
        action='store_true',
        default=True,
        help='enable cash-aware buy trimming in verify simulation (default: enabled)'
    )
    parser.add_argument(
        '--verify-no-cash-aware',
        dest='verify_cash_aware',
        action='store_false',
        help='disable cash-aware buy trimming in verify simulation'
    )
    
    args = parser.parse_args()

    def _run_one(selected_strategy_id: Optional[str]) -> Dict:
        verifier = StrategyVerifier(
            start_date=args.start,
            end_date=args.end,
            strategies_config=args.strategies_config,
            strategy_id=selected_strategy_id,
            king_step=args.king,
            take_profit_ratio=args.take_profit,
            top_k=args.top_k,
            fee_rate=args.fee_rate,
            initial_cash=args.initial_cash,
            verify_cash_aware=args.verify_cash_aware,
        )
        verifier.print_config_alignment()
        sim_records = verifier.run_simulation(initial_cash=args.initial_cash)
        backtest_result = verifier.run_backtest()
        passed = verifier.compare_results(sim_records, backtest_result)
        return {
            "strategy_id": verifier.strategy_id or selected_strategy_id or "legacy",
            "strategy_name": verifier.strategy_name or (selected_strategy_id or "legacy"),
            "passed": passed,
            "suffix": verifier._artifact_suffix(),
            "top_k": int(verifier.top_k) if verifier.top_k is not None else 0,
            "initial_cash": float(verifier.initial_cash) if verifier.initial_cash is not None else 0.0,
        }

    # Legacy king mode always runs a single strategy.
    if args.king is not None:
        _run_one(args.strategy_id)
    else:
        if args.strategy_id:
            strategy_ids = [args.strategy_id]
        else:
            config_path = args.strategies_config or os.path.join(
                project_root, "strategy_manager", "strategies_config.json"
            )
            cfg = load_strategies_config(config_path)
            strategy_ids = [s.id for s in cfg.get_enabled_strategies()]
            if not strategy_ids:
                raise ValueError(f"No enabled strategies in config: {config_path}")
            logger.info(f"No --strategy-id provided, verify all enabled strategies: {strategy_ids}")

        run_infos: Dict[str, Dict] = {}
        results = {}
        for sid in strategy_ids:
            logger.info("\n" + "=" * 80)
            logger.info(f"VERIFY STRATEGY: {sid}")
            logger.info("=" * 80)
            run_info = _run_one(sid)
            run_infos[sid] = run_info
            results[sid] = bool(run_info.get("passed", False))

        if len(results) > 1:
            logger.info("\n" + "=" * 80)
            logger.info("MULTI-STRATEGY VERIFICATION SUMMARY")
            logger.info("=" * 80)
            for sid, ok in results.items():
                logger.info(f"{sid}: {'PASS' if ok else 'FAIL'}")

            try:
                combo_outputs = generate_multi_strategy_artifacts(
                    run_infos=list(run_infos.values()),
                    artifacts_dir=os.path.join(current_dir, "artifacts"),
                    start_date=args.start,
                    end_date=args.end,
                )
                logger.info("\n" + "=" * 80)
                logger.info("MULTI-STRATEGY COMBINED ARTIFACTS")
                logger.info("=" * 80)
                for key, path in combo_outputs.items():
                    logger.info(f"{key}: {path}")
            except Exception as exc:
                logger.exception("Failed to build multi-strategy combined artifacts: %s", exc)



