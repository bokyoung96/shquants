from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.policy.base import PositionPlan
from backtesting.signals.revision_gapup_hold_compression import RevisionGapupHoldCompressionSignalProducer

from .base import RegisteredStrategy


@dataclass(slots=True)
class RevisionGapupHoldCompressionEqualWeight(RegisteredStrategy):
    revision_lookback: int = 20
    compression_lookback: int = 20
    trend_lookback: int = 60
    exit_lookback: int = 10
    max_hold_days: int = 7
    op_revision_threshold: float = 0.03
    eps_revision_threshold: float = 0.02
    gap_threshold: float = 0.02

    def __post_init__(self) -> None:
        self.signal_producer = RevisionGapupHoldCompressionSignalProducer(
            revision_lookback=self.revision_lookback,
            compression_lookback=self.compression_lookback,
            trend_lookback=self.trend_lookback,
            exit_lookback=self.exit_lookback,
            max_hold_days=self.max_hold_days,
            op_revision_threshold=self.op_revision_threshold,
            eps_revision_threshold=self.eps_revision_threshold,
            gap_threshold=self.gap_threshold,
        )

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
            tradable = bundle.context['tradable'].reindex_like(universe).fillna(False).astype(bool) & universe
            eligible_entry = bundle.context['eligible_entry'].reindex_like(universe).fillna(False).astype(bool) & universe
            eligible_exit = bundle.context['eligible_exit'].reindex_like(universe).fillna(False).astype(bool) | ~universe
        else:
            tradable = bundle.context['tradable'].fillna(False).astype(bool)
            eligible_entry = bundle.context['eligible_entry'].fillna(False).astype(bool)
            eligible_exit = bundle.context['eligible_exit'].fillna(False).astype(bool)

        max_hold_days = int(bundle.context.get('max_hold_days', self.max_hold_days))
        symbols = list(bundle.alpha.columns)
        active = pd.Series(False, index=symbols, dtype=bool)
        ages = pd.Series(0, index=symbols, dtype=int)
        rows: dict[pd.Timestamp, pd.Series] = {}

        for date in bundle.alpha.index:
            exits_today = eligible_exit.loc[date] | ages.ge(max_hold_days)
            active = active & ~exits_today
            entries_today = eligible_entry.loc[date] & tradable.loc[date] & ~active
            active = active | entries_today
            ages = ages.where(~active, ages + 1)
            ages = ages.where(active, 0)

            row = pd.Series(0.0, index=symbols, dtype=float)
            active_count = int(active.sum())
            if active_count > 0:
                row.loc[active] = 1.0 / active_count
            rows[date] = row

        weights = pd.DataFrame.from_dict(rows, orient='index').reindex(index=bundle.alpha.index, columns=symbols).fillna(0.0)
        return PositionPlan(target_weights=weights, bucket_ledger=pd.DataFrame(), bucket_meta={}, validation={})
