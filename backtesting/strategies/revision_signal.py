from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class RevisionSignal(ComposableStrategy):
    lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = _RevisionSignalProducer(lookback=self.lookback)
        self.construction_rule = _SignalEqualWeight()


@dataclass(slots=True)
class _RevisionSignalProducer:
    lookback: int = 20

    _MARKET_TREND_LOOKBACK = 120
    _MARKET_TREND_MIN_PERIODS = 60

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_OP_NFQ1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        eps = market.frames["eps_fwd_q1"].reindex_like(close).ffill()
        op = market.frames["op_fwd_q1"].reindex_like(close).ffill()
        benchmark = self._benchmark(market.frames["benchmark"]).reindex(close.index).ffill()

        eps_revision = eps.pct_change(self.lookback, fill_method=None)
        op_revision = op.pct_change(self.lookback, fill_method=None)
        revision_passes = eps_revision.gt(0.0) & op_revision.gt(0.0)

        benchmark_average = benchmark.rolling(
            self._MARKET_TREND_LOOKBACK,
            min_periods=self._MARKET_TREND_MIN_PERIODS,
        ).mean()
        market_risk_on = benchmark.gt(benchmark_average)
        selected = revision_passes & market_risk_on.reindex(close.index).fillna(False).to_numpy()[:, None]

        alpha = (0.5 * eps_revision.rank(axis=1, pct=True)) + (0.5 * op_revision.rank(axis=1, pct=True))
        alpha = alpha.where(selected)
        return SignalBundle(
            alpha=alpha,
            context={"tradable": selected & close.notna()},
            meta={
                "market_trend_lookback": self._MARKET_TREND_LOOKBACK,
                "market_trend_min_periods": self._MARKET_TREND_MIN_PERIODS,
            },
        )

    @staticmethod
    def _benchmark(frame: pd.DataFrame) -> pd.Series:
        if "IKS200" in frame.columns:
            return frame["IKS200"]
        return frame.iloc[:, 0]


@dataclass(slots=True)
class _SignalEqualWeight:
    def build(self, bundle: SignalBundle) -> ConstructionResult:
        selected = bundle.alpha.notna()
        tradable = bundle.context.get("tradable")
        if isinstance(tradable, pd.DataFrame):
            tradable = tradable.reindex(index=selected.index, columns=selected.columns).fillna(False).astype(bool)
            selected = selected & tradable

        count = selected.sum(axis=1).astype(float)
        denominator = count.where(count.ne(0.0), float("nan"))
        weights = selected.astype(float).div(denominator, axis=0).fillna(0.0).astype(float)
        return ConstructionResult(
            base_target_weights=weights,
            selection_mask=selected,
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )
