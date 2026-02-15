"""
SQL state store for sim_run live/replay state.

This module persists and restores:
1) daily nav history
2) daily holdings snapshot
3) trade history
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine

from data_pipeline.config import Config
from execution.sim_trader import TradeRecord
from strategy_manager.cb_portfolio import CBPosition
from strategy_manager.nav_tracker import DailyRecord

logger = logging.getLogger(__name__)


class SQLStateStore:
    """Persist sim state into MySQL tables."""

    TABLES = {
        "replay": {
            "nav": "sim_nav_history",
            "holding": "sim_daily_holdings",
            "trade": "sim_trade_history",
        },
        "live": {
            "nav": "sim_live_nav_history",
            "holding": "sim_live_daily_holdings",
            "trade": "sim_live_trade_history",
        },
    }

    def __init__(self, sql_engine: Optional[Engine] = None, dataset: str = "replay"):
        if dataset not in self.TABLES:
            raise ValueError(f"Unsupported dataset '{dataset}', expected one of {list(self.TABLES)}")
        self.sql_engine = sql_engine or create_engine(Config.CB_DB_DSN)
        self.dataset = dataset
        self.NAV_TABLE = self.TABLES[dataset]["nav"]
        self.HOLDING_TABLE = self.TABLES[dataset]["holding"]
        self.TRADE_TABLE = self.TABLES[dataset]["trade"]

    def ensure_tables_exist(self):
        """Check required sim tables exist."""
        inspector = inspect(self.sql_engine)
        required = [self.NAV_TABLE, self.HOLDING_TABLE, self.TRADE_TABLE]
        missing = [t for t in required if not inspector.has_table(t)]
        if missing:
            raise RuntimeError(
                f"Missing SQL tables for sim state dataset='{self.dataset}': "
                f"{missing}. Run migrations under infra/migrations/."
            )

    def reset_strategy(self, strategy_id: str):
        """Delete all state rows for one strategy."""
        with self.sql_engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {self.TRADE_TABLE} WHERE strategy_id = :strategy_id"),
                {"strategy_id": strategy_id},
            )
            conn.execute(
                text(f"DELETE FROM {self.HOLDING_TABLE} WHERE strategy_id = :strategy_id"),
                {"strategy_id": strategy_id},
            )
            conn.execute(
                text(f"DELETE FROM {self.NAV_TABLE} WHERE strategy_id = :strategy_id"),
                {"strategy_id": strategy_id},
            )

    def load_runtime_state(
        self,
        strategy_id: str,
        initial_capital: float,
    ) -> Dict:
        """Load latest runtime state for strategy."""
        self.ensure_tables_exist()

        with self.sql_engine.connect() as conn:
            nav_rows = conn.execute(
                text(
                    f"""
                    SELECT trade_date, nav, cash, holdings_value, holdings_count, daily_ret, cum_ret, mdd
                    FROM {self.NAV_TABLE}
                    WHERE strategy_id = :strategy_id
                    ORDER BY trade_date
                    """
                ),
                {"strategy_id": strategy_id},
            ).mappings().all()

            peak_nav_row = conn.execute(
                text(
                    f"""
                    SELECT MAX(nav) AS peak_nav
                    FROM {self.NAV_TABLE}
                    WHERE strategy_id = :strategy_id
                    """
                ),
                {"strategy_id": strategy_id},
            ).mappings().first()

            latest_holding_date_row = conn.execute(
                text(
                    f"""
                    SELECT MAX(trade_date) AS trade_date
                    FROM {self.HOLDING_TABLE}
                    WHERE strategy_id = :strategy_id
                    """
                ),
                {"strategy_id": strategy_id},
            ).mappings().first()

            positions: List[CBPosition] = []
            latest_holding_date = (
                latest_holding_date_row.get("trade_date")
                if latest_holding_date_row is not None
                else None
            )
            if latest_holding_date is not None:
                holding_rows = conn.execute(
                    text(
                        f"""
                        SELECT code, name, shares, avg_cost, last_price, entry_date
                        FROM {self.HOLDING_TABLE}
                        WHERE strategy_id = :strategy_id
                          AND trade_date = :trade_date
                        ORDER BY code
                        """
                    ),
                    {"strategy_id": strategy_id, "trade_date": latest_holding_date},
                ).mappings().all()
                for row in holding_rows:
                    positions.append(
                        CBPosition(
                            code=str(row["code"]),
                            name=str(row["name"]),
                            shares=int(row["shares"]),
                            avg_cost=float(row["avg_cost"]),
                            last_price=float(row["last_price"]),
                            entry_date=self._to_date_str(row["entry_date"]) or "",
                        )
                    )

            trade_rows = conn.execute(
                text(
                    f"""
                    SELECT trade_date, trade_time, code, name, side, shares, price, amount
                    FROM {self.TRADE_TABLE}
                    WHERE strategy_id = :strategy_id
                    ORDER BY trade_date, trade_time, id
                    """
                ),
                {"strategy_id": strategy_id},
            ).mappings().all()

        records: List[DailyRecord] = [
            DailyRecord(
                date=self._to_date_str(r["trade_date"]) or "",
                nav=float(r["nav"]),
                cash=float(r["cash"]),
                holdings_value=float(r["holdings_value"]),
                holdings_count=int(r["holdings_count"]),
                daily_ret=float(r["daily_ret"]),
                cum_ret=float(r["cum_ret"]),
                mdd=float(r["mdd"]),
            )
            for r in nav_rows
        ]

        trade_history: List[TradeRecord] = [
            TradeRecord(
                date=self._to_date_str(r["trade_date"]) or "",
                code=str(r["code"]),
                name=str(r["name"]),
                side=str(r["side"]),
                shares=int(r["shares"]),
                price=float(r["price"]),
                amount=float(r["amount"]),
                timestamp=self._to_datetime_str(r["trade_time"]),
            )
            for r in trade_rows
        ]

        if records:
            latest_cash = records[-1].cash
            peak_nav = float(peak_nav_row["peak_nav"]) if peak_nav_row and peak_nav_row["peak_nav"] is not None else max(
                r.nav for r in records
            )
        else:
            latest_cash = float(initial_capital)
            peak_nav = float(initial_capital)

        return {
            "records": records,
            "positions": positions,
            "trade_history": trade_history,
            "cash": latest_cash,
            "peak_nav": peak_nav,
        }

    def save_daily_state(
        self,
        strategy_id: str,
        trade_date: str,
        nav_record: DailyRecord,
        positions: List[CBPosition],
        trade_records: List[TradeRecord],
    ):
        """Persist one day state in a single transaction."""
        self.ensure_tables_exist()

        date_value = self._normalize_date(trade_date)

        try:
            with self.sql_engine.begin() as conn:
                for rec in trade_records:
                    conn.execute(
                        text(
                            f"""
                            INSERT INTO {self.TRADE_TABLE}
                            (strategy_id, trade_date, trade_time, code, name, side, shares, price, amount)
                            VALUES
                            (:strategy_id, :trade_date, :trade_time, :code, :name, :side, :shares, :price, :amount)
                            """
                        ),
                        {
                            "strategy_id": strategy_id,
                            "trade_date": self._normalize_date(rec.date),
                            "trade_time": self._normalize_datetime(rec.timestamp),
                            "code": rec.code,
                            "name": rec.name,
                            "side": str(rec.side),
                            "shares": int(rec.shares),
                            "price": float(rec.price),
                            "amount": float(rec.amount),
                        },
                    )

                conn.execute(
                    text(
                        f"""
                        DELETE FROM {self.HOLDING_TABLE}
                        WHERE strategy_id = :strategy_id
                          AND trade_date = :trade_date
                        """
                    ),
                    {"strategy_id": strategy_id, "trade_date": date_value},
                )

                for pos in positions:
                    conn.execute(
                        text(
                            f"""
                            INSERT INTO {self.HOLDING_TABLE}
                            (strategy_id, trade_date, code, name, shares, avg_cost, last_price, entry_date, market_value, pnl, pnl_pct)
                            VALUES
                            (:strategy_id, :trade_date, :code, :name, :shares, :avg_cost, :last_price, :entry_date, :market_value, :pnl, :pnl_pct)
                            """
                        ),
                        {
                            "strategy_id": strategy_id,
                            "trade_date": date_value,
                            "code": pos.code,
                            "name": pos.name,
                            "shares": int(pos.shares),
                            "avg_cost": float(pos.avg_cost),
                            "last_price": float(pos.last_price),
                            "entry_date": self._normalize_date(pos.entry_date) if pos.entry_date else None,
                            "market_value": float(pos.market_value),
                            "pnl": float(pos.pnl),
                            "pnl_pct": float(pos.pnl_pct),
                        },
                    )

                conn.execute(
                    text(
                        f"""
                        INSERT INTO {self.NAV_TABLE}
                        (strategy_id, trade_date, nav, cash, holdings_value, holdings_count, daily_ret, cum_ret, mdd)
                        VALUES
                        (:strategy_id, :trade_date, :nav, :cash, :holdings_value, :holdings_count, :daily_ret, :cum_ret, :mdd)
                        ON DUPLICATE KEY UPDATE
                          nav = VALUES(nav),
                          cash = VALUES(cash),
                          holdings_value = VALUES(holdings_value),
                          holdings_count = VALUES(holdings_count),
                          daily_ret = VALUES(daily_ret),
                          cum_ret = VALUES(cum_ret),
                          mdd = VALUES(mdd),
                          updated_at = CURRENT_TIMESTAMP
                        """
                    ),
                    {
                        "strategy_id": strategy_id,
                        "trade_date": date_value,
                        "nav": float(nav_record.nav),
                        "cash": float(nav_record.cash),
                        "holdings_value": float(nav_record.holdings_value),
                        "holdings_count": int(nav_record.holdings_count),
                        "daily_ret": float(nav_record.daily_ret),
                        "cum_ret": float(nav_record.cum_ret),
                        "mdd": float(nav_record.mdd),
                    },
                )
        except SQLAlchemyError as e:
            raise RuntimeError(f"Failed to persist SQL state for strategy={strategy_id}, date={trade_date}: {e}") from e

    @staticmethod
    def _normalize_date(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        text_value = str(value)
        if len(text_value) >= 10:
            text_value = text_value[:10]
        return datetime.strptime(text_value, "%Y-%m-%d").date()

    @staticmethod
    def _normalize_datetime(value):
        if isinstance(value, datetime):
            return value
        text_value = str(value)
        if len(text_value) == 10:
            text_value = text_value + "T00:00:00"
        return datetime.fromisoformat(text_value)

    @staticmethod
    def _to_date_str(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        text_value = str(value)
        return text_value[:10]

    @staticmethod
    def _to_datetime_str(value) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%dT%H:%M:%S")
        text_value = str(value)
        if len(text_value) == 10:
            return text_value + "T00:00:00"
        return text_value.replace(" ", "T")
