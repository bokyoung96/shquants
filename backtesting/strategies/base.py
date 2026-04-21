from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.policy.pass_through import PassThroughPolicy
from backtesting.policy.base import PositionPlan
from backtesting.signals.base import SignalBundle


class RegisteredStrategy(ABC):
    @property
    @abstractmethod
    def datasets(self) -> tuple[DatasetId, ...]:
        raise NotImplementedError

    @abstractmethod
    def build_signal(self, market: MarketData) -> pd.DataFrame:
        raise NotImplementedError

    def target_weights(self, signal: pd.Series) -> pd.Series:
        raise NotImplementedError

    def build_plan(self, market: MarketData) -> PositionPlan:
        signal = self.build_signal(market)
        if market.universe is not None:
            universe = market.universe.reindex(index=signal.index, columns=signal.columns)
            universe = universe.astype("boolean").fillna(False).astype(bool)
            signal = signal.where(universe)

        rows: dict[pd.Timestamp, pd.Series] = {}
        for ts in signal.index:
            rows[ts] = self.target_weights(signal.loc[ts])

        weights = (
            pd.DataFrame.from_dict(rows, orient="index")
            .reindex(index=signal.index, columns=signal.columns)
            .fillna(0.0)
            .astype(float)
        )
        bundle = SignalBundle(alpha=signal, context={"tradable": signal.notna()})
        construction = ConstructionResult(
            base_target_weights=weights,
            selection_mask=weights.ne(0.0),
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )
        return PassThroughPolicy().apply(construction=construction, market=market, bundle=bundle)

    def build_weights(self, market: MarketData) -> pd.DataFrame:
        return self.build_plan(market).target_weights
