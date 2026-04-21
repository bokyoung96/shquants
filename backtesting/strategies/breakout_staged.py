from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan

from .base import RegisteredStrategy


@dataclass(slots=True)
class Breakout52WeekStaged(RegisteredStrategy):
    breakout_window: int = 252
    exit_window: int = 20
    pullback_ma_window: int = 10
    pullback_band: float = 0.01

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_ADJ_C,)

    def build_signal(self, market: MarketData) -> pd.DataFrame:
        return market.frames["close"].astype(float)

    def build_plan(self, market: MarketData) -> PositionPlan:
        close = self.build_signal(market)
        if market.universe is not None:
            universe = market.universe.reindex(index=close.index, columns=close.columns)
            universe = universe.astype("boolean").fillna(False).astype(bool)
            close = close.where(universe)

        prior_high = close.rolling(self.breakout_window, min_periods=self.breakout_window).max().shift(1)
        prior_low = close.rolling(self.exit_window, min_periods=self.exit_window).min().shift(1)
        ma10 = close.rolling(self.pullback_ma_window, min_periods=self.pullback_ma_window).mean()

        breakout = close.gt(prior_high).fillna(False)
        exit_signal = close.lt(prior_low).fillna(False)

        bucket_ids = ("entry", "add_1", "add_2")
        states = {
            symbol: {"active": [False, False, False], "peak": None, "pullback_ready": False}
            for symbol in close.columns
        }

        target_rows: list[pd.Series] = []
        ledger_rows: list[dict[str, object]] = []

        for timestamp in close.index:
            for symbol in close.columns:
                price = close.loc[timestamp, symbol]
                if pd.isna(price):
                    states[symbol] = {"active": [False, False, False], "peak": None, "pullback_ready": False}
                    continue

                state = states[symbol]
                active = state["active"]
                peak = state["peak"]
                pullback_ready = state["pullback_ready"]

                if not any(active):
                    if bool(breakout.loc[timestamp, symbol]):
                        active[0] = True
                        peak = float(price)
                        pullback_ready = False
                else:
                    if bool(exit_signal.loc[timestamp, symbol]):
                        for bucket_index in range(len(bucket_ids) - 1, -1, -1):
                            if active[bucket_index]:
                                active[bucket_index] = False
                                break
                        pullback_ready = False
                        if not any(active):
                            peak = None
                    else:
                        current_peak = float(price) if peak is None else float(peak)
                        ma_value = ma10.loc[timestamp, symbol]
                        if not pd.isna(ma_value) and float(price) <= float(ma_value) * (1.0 + self.pullback_band):
                            pullback_ready = True

                        if pullback_ready and not pd.isna(ma_value) and float(price) > float(ma_value):
                            for bucket_index in range(1, len(bucket_ids)):
                                if not active[bucket_index]:
                                    active[bucket_index] = True
                                    break
                            pullback_ready = False
                            peak = float(price)
                        else:
                            peak = max(current_peak, float(price))

                state["peak"] = peak
                state["pullback_ready"] = pullback_ready

            active_symbols = [symbol for symbol, state in states.items() if any(state["active"])]
            full_weight = (1.0 / len(active_symbols)) if active_symbols else 0.0
            weights = pd.Series(0.0, index=close.columns, dtype=float)

            for symbol in active_symbols:
                state = states[symbol]
                for bucket_index, bucket_id in enumerate(bucket_ids):
                    if not state["active"][bucket_index]:
                        continue
                    bucket_weight = full_weight / len(bucket_ids)
                    weights.loc[symbol] += bucket_weight
                    ledger_rows.append(
                        {
                            "date": timestamp,
                            "symbol": symbol,
                            "side": "long",
                            "bucket_id": bucket_id,
                            "stage_index": bucket_index,
                            "target_weight": bucket_weight,
                            "actual_weight": bucket_weight,
                            "target_qty": 0.0,
                            "actual_qty": 0.0,
                            "entry_price": None,
                            "mark_price": None,
                            "bucket_return": 0.0,
                            "state": "active",
                            "event": "staged_breakout",
                            "construction_group": None,
                            "budget_id": bucket_id,
                        }
                    )

            target_rows.append(weights)

        target_weights = pd.DataFrame(target_rows, index=close.index, columns=close.columns).fillna(0.0)
        bucket_ledger = pd.DataFrame.from_records(ledger_rows, columns=BUCKET_LEDGER_COLUMNS)
        return PositionPlan(
            target_weights=target_weights,
            bucket_ledger=bucket_ledger,
            bucket_meta={"policy_name": pd.Series(["breakout_staged"], name="policy_name")},
            validation={},
        )
