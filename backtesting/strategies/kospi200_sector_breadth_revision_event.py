from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.policy.base import PositionPlan
from backtesting.signals.kospi200_sector_breadth_revision_event import (
    Kospi200SectorBreadthRevisionEventSignalProducer,
)

from .base import RegisteredStrategy


@dataclass(slots=True)
class Kospi200SectorBreadthRevisionEventEqualWeight(RegisteredStrategy):
    def __post_init__(self) -> None:
        self.signal_producer = Kospi200SectorBreadthRevisionEventSignalProducer()

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
            tradable = bundle.context["tradable"].reindex_like(universe).fillna(False).astype(bool) & universe
            eligible_entry = bundle.context["eligible_entry"].reindex_like(universe).fillna(False).astype(bool) & universe
            eligible_exit = bundle.context["eligible_exit"].reindex_like(universe).fillna(False).astype(bool) | ~universe
        else:
            tradable = bundle.context["tradable"].fillna(False).astype(bool)
            eligible_entry = bundle.context["eligible_entry"].fillna(False).astype(bool)
            eligible_exit = bundle.context["eligible_exit"].fillna(False).astype(bool)

        symbols = list(bundle.alpha.columns)
        active = pd.Series(False, index=symbols, dtype=bool)
        rows: dict[pd.Timestamp, pd.Series] = {}

        for date in bundle.alpha.index:
            active = active & ~eligible_exit.loc[date]
            entries_today = eligible_entry.loc[date] & tradable.loc[date] & ~active
            active = active | entries_today

            row = pd.Series(0.0, index=symbols, dtype=float)
            active_count = int(active.sum())
            if active_count > 0:
                row.loc[active] = 1.0 / active_count
            rows[date] = row

        weights = pd.DataFrame.from_dict(rows, orient="index").reindex(index=bundle.alpha.index, columns=symbols).fillna(0.0)
        return PositionPlan(target_weights=weights, bucket_ledger=pd.DataFrame(), bucket_meta={}, validation={})
