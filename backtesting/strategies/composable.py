from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.policy.base import PositionPlan, PositionPolicy
from backtesting.policy.pass_through import PassThroughPolicy
from backtesting.signals.base import SignalBundle

from .base import RegisteredStrategy


class SignalProducer(Protocol):
    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        ...

    def build(self, market: MarketData) -> SignalBundle:
        ...


class ConstructionRule(Protocol):
    def build(self, bundle: SignalBundle) -> ConstructionResult:
        ...


@dataclass(slots=True)
class ComposableStrategy(RegisteredStrategy):
    position_policy: PositionPolicy = field(default_factory=PassThroughPolicy)
    signal_producer: SignalProducer = field(init=False)
    construction_rule: ConstructionRule = field(init=False)

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return self.signal_producer.datasets

    def build_signal(self, market: MarketData) -> pd.DataFrame:
        return self.signal_producer.build(market).alpha

    def build_plan(self, market: MarketData) -> PositionPlan:
        bundle = self.signal_producer.build(market)
        bundle = self._apply_universe(bundle=bundle, market=market)
        construction = self.construction_rule.build(bundle)
        return self.position_policy.apply(
            construction=construction,
            market=market,
            bundle=bundle,
        )

    def _apply_universe(self, bundle: SignalBundle, market: MarketData) -> SignalBundle:
        if market.universe is None:
            return bundle

        universe = market.universe.reindex(index=bundle.alpha.index, columns=bundle.alpha.columns)
        universe = universe.astype("boolean").fillna(False).astype(bool)
        context = dict(bundle.context)
        tradable = context.get("tradable")
        if isinstance(tradable, pd.DataFrame):
            tradable = tradable.reindex(index=bundle.alpha.index, columns=bundle.alpha.columns)
            tradable = tradable.astype("boolean").fillna(False).astype(bool)
            context["tradable"] = tradable & universe
        else:
            context["tradable"] = universe

        return SignalBundle(
            alpha=bundle.alpha.where(universe),
            context=context,
            meta=dict(bundle.meta),
        )
