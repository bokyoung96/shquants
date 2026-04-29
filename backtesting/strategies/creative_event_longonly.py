from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.long_only import LongOnlyTopN
from backtesting.policy.base import PositionPlan
from backtesting.signals.creative_event_asymmetry import CreativeEventAsymmetrySignalProducer

from .base import RegisteredStrategy


@dataclass(slots=True)
class CreativeEventLongOnlyTopN(RegisteredStrategy):
    top_n: int = 8
    revision_threshold: float = 0.04
    flow_lookback: int = 20
    support_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = CreativeEventAsymmetrySignalProducer(
            revision_threshold=self.revision_threshold,
            flow_lookback=self.flow_lookback,
            support_lookback=self.support_lookback,
        )
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return self.signal_producer.datasets

    def build_signal(self, market) -> pd.DataFrame:
        return self.signal_producer.build(market).alpha

    def target_weights(self, signal: pd.Series) -> pd.Series:
        raise NotImplementedError

    def build_plan(self, market) -> PositionPlan:
        bundle = self.signal_producer.build(market)
        if market.universe is not None:
            universe = market.universe.reindex(index=bundle.alpha.index, columns=bundle.alpha.columns).fillna(False).astype(bool)
            bundle = type(bundle)(
                alpha=bundle.alpha.where(universe),
                context={k: v.reindex(index=bundle.alpha.index, columns=bundle.alpha.columns).fillna(False).astype(bool) if getattr(v, 'dtypes', None) is not None and str(v.dtypes.iloc[0]) in ('bool','boolean') else v for k, v in bundle.context.items()},
                meta=bundle.meta,
            )
        construction = self.construction_rule.build(bundle)
        weights = construction.base_target_weights.fillna(0.0).astype(float)
        eligible = bundle.context['eligible_entry'].reindex(index=weights.index, columns=weights.columns).fillna(False).astype(bool)
        weights = weights.where(eligible, 0.0)
        return PositionPlan(target_weights=weights, bucket_ledger=pd.DataFrame(), bucket_meta={}, validation={})
