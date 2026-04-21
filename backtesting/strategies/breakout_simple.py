from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.policy.pass_through import PassThroughPolicy
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(frozen=True, slots=True)
class Breakout52WeekSignalProducer:
    breakout_window: int = 252
    exit_window: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_ADJ_C,)

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"].astype(float)
        prior_high = close.rolling(self.breakout_window, min_periods=self.breakout_window).max().shift(1)
        prior_low = close.rolling(self.exit_window, min_periods=self.exit_window).min().shift(1)
        entry = close.gt(prior_high).fillna(False)
        exit_mask = close.lt(prior_low).fillna(False)
        return SignalBundle(
            alpha=entry.astype(float),
            context={
                "close": close,
                "entry": entry,
                "exit": exit_mask,
                "tradable": close.notna(),
            },
        )


@dataclass(frozen=True, slots=True)
class _Breakout52WeekConstructionRule:
    def build(self, bundle: SignalBundle) -> ConstructionResult:
        close = bundle.context["close"]
        entry = bundle.context["entry"]
        exit_mask = bundle.context["exit"]
        tradable = bundle.context.get("tradable")
        if not isinstance(tradable, pd.DataFrame):
            tradable = pd.DataFrame(True, index=close.index, columns=close.columns)
        else:
            tradable = tradable.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)

        active = pd.Series(False, index=close.columns, dtype=bool)
        rows: dict[pd.Timestamp, pd.Series] = {}

        for timestamp in close.index:
            tradable_row = tradable.loc[timestamp]
            active = active & tradable_row
            active = active & ~exit_mask.loc[timestamp].fillna(False)
            active = active | (entry.loc[timestamp].fillna(False) & tradable_row)

            weights = pd.Series(0.0, index=close.columns, dtype=float)
            active_count = int(active.sum())
            if active_count > 0:
                weights.loc[active] = 1.0 / active_count
            rows[timestamp] = weights

        base_target_weights = (
            pd.DataFrame.from_dict(rows, orient="index")
            .reindex(index=close.index, columns=close.columns)
            .fillna(0.0)
            .astype(float)
        )
        selection_mask = base_target_weights.ne(0.0)
        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=selection_mask,
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )


@dataclass(slots=True)
class Breakout52WeekSimple(ComposableStrategy):
    breakout_window: int = 252
    exit_window: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = Breakout52WeekSignalProducer(
            breakout_window=self.breakout_window,
            exit_window=self.exit_window,
        )
        self.construction_rule = _Breakout52WeekConstructionRule()
        self.position_policy = PassThroughPolicy()
