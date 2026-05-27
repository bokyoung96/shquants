from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


class MfbtFactor(Protocol):
    name: str

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        ...

    def build(self, market: MarketData) -> pd.DataFrame:
        ...


@dataclass(slots=True)
class PriceMomentumFactor:
    high_lookback: int = 252
    threshold: float = 0.8
    name: str = "price_momentum"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_ADJ_C,)

    def build(self, market: MarketData) -> pd.DataFrame:
        close = market.frames["close"]
        high = close.rolling(self.high_lookback, min_periods=self.high_lookback).max()
        ratio = close.divide(high)
        return ratio.gt(self.threshold).astype(float)


@dataclass(slots=True)
class Mfbt(ComposableStrategy):
    top_n: int = 20
    high_lookback: int = 252
    price_momentum_threshold: float = 0.8

    def __post_init__(self) -> None:
        self.signal_producer = _MfbtSignal(
            factors=(
                PriceMomentumFactor(
                    high_lookback=self.high_lookback,
                    threshold=self.price_momentum_threshold,
                ),
            ),
        )
        self.construction_rule = _MfbtConstruction(top_n=self.top_n)


@dataclass(slots=True)
class _MfbtSignal:
    factors: tuple[MfbtFactor, ...]

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        datasets: list[DatasetId] = []
        for factor in self.factors:
            for dataset in factor.datasets:
                if dataset not in datasets:
                    datasets.append(dataset)
        return tuple(datasets)

    def build(self, market: MarketData) -> SignalBundle:
        factor_frames = {factor.name: factor.build(market) for factor in self.factors}
        price_momentum = factor_frames["price_momentum"]
        return SignalBundle(
            alpha=price_momentum,
            context={"tradable": price_momentum.eq(1.0)},
            meta=factor_frames,
        )


@dataclass(slots=True)
class _MfbtConstruction:
    top_n: int = 20

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        eligible = alpha.gt(0.0)
        ranks = alpha.where(eligible).rank(axis=1, ascending=False, method="first", na_option="bottom")
        selection_mask = ranks.le(self.top_n) & eligible
        selected_count = selection_mask.sum(axis=1).clip(upper=self.top_n)
        denominator = selected_count.astype(float).where(selected_count.ne(0), float("nan"))
        base_target_weights = selection_mask.astype(float).div(denominator, axis=0).fillna(0.0).astype(float)
        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=selection_mask,
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )
