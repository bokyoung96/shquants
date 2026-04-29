from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.policy.base import PositionPlan
from backtesting.signals.kospi200_bs_opf_band import Kospi200BsOpfBandSignalProducer

from .base import RegisteredStrategy


@dataclass(slots=True)
class _Kospi200BsOpfBandBase(RegisteredStrategy):
    variant: str = 'balanced'

    def __post_init__(self) -> None:
        self.signal_producer = Kospi200BsOpfBandSignalProducer()

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return self.signal_producer.datasets

    def build_signal(self, market) -> pd.DataFrame:
        return self.signal_producer.build(market).alpha

    def target_weights(self, signal: pd.Series) -> pd.Series:
        raise NotImplementedError

    def build_plan(self, market) -> PositionPlan:
        bundle = self.signal_producer.build(market)
        tradable = bundle.context['tradable'].fillna(False).astype(bool)
        band_state = bundle.meta['band_state']
        exposure = band_state.apply(lambda row: self._map_exposure(row['bs_band'], row['ofs_band']), axis=1).fillna(0.0)

        counts = tradable.sum(axis=1).replace(0, pd.NA)
        per_name = exposure.divide(counts).fillna(0.0)
        weights = tradable.mul(per_name, axis=0).astype(float)

        plan = PositionPlan(target_weights=weights, bucket_ledger=pd.DataFrame(), bucket_meta={}, validation={})
        plan.validation['variant'] = self.variant
        return plan

    def _map_exposure(self, bs_band: str | float, ofs_band: str | float) -> float:
        if pd.isna(bs_band) or pd.isna(ofs_band):
            return 0.0
        return float(self._matrix().get((str(bs_band), str(ofs_band)), 0.0))

    def _matrix(self) -> dict[tuple[str, str], float]:
        if self.variant == 'bs_only':
            return {
                ('L', 'L'): 0.0, ('L', 'M'): 0.0, ('L', 'H'): 0.0,
                ('M', 'L'): 0.25, ('M', 'M'): 0.25, ('M', 'H'): 0.25,
                ('H', 'L'): 1.0, ('H', 'M'): 1.0, ('H', 'H'): 1.0,
            }
        if self.variant == 'strict':
            return {
                ('L', 'L'): 0.0, ('L', 'M'): 0.0, ('L', 'H'): 0.0,
                ('M', 'L'): 0.0, ('M', 'M'): 0.0, ('M', 'H'): 0.25,
                ('H', 'L'): 0.0, ('H', 'M'): 0.5, ('H', 'H'): 1.0,
            }
        if self.variant == 'aggressive':
            return {
                ('L', 'L'): 0.0, ('L', 'M'): 0.0, ('L', 'H'): 0.25,
                ('M', 'L'): 0.0, ('M', 'M'): 0.5, ('M', 'H'): 0.75,
                ('H', 'L'): 0.25, ('H', 'M'): 0.75, ('H', 'H'): 1.0,
            }
        return {
            ('L', 'L'): 0.0, ('L', 'M'): 0.0, ('L', 'H'): 0.25,
            ('M', 'L'): 0.0, ('M', 'M'): 0.25, ('M', 'H'): 0.5,
            ('H', 'L'): 0.25, ('H', 'M'): 0.5, ('H', 'H'): 1.0,
        }


@dataclass(slots=True)
class Kospi200BsOnlyBand(_Kospi200BsOpfBandBase):
    variant: str = 'bs_only'


@dataclass(slots=True)
class Kospi200BsOpfBandBalanced(_Kospi200BsOpfBandBase):
    variant: str = 'balanced'


@dataclass(slots=True)
class Kospi200BsOpfBandStrict(_Kospi200BsOpfBandBase):
    variant: str = 'strict'


@dataclass(slots=True)
class Kospi200BsOpfBandAggressive(_Kospi200BsOpfBandBase):
    variant: str = 'aggressive'
