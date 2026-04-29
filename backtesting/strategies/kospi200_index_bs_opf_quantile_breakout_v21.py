from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.policy.base import PositionPlan
from backtesting.signals.kospi200_index_bs_opf_quantile_breakout import (
    Kospi200IndexBsOpfQuantileBreakoutSignalProducer,
)

from .base import RegisteredStrategy


@dataclass(slots=True)
class _BaseV21(RegisteredStrategy):
    variant: str = 'hysteresis'

    def __post_init__(self) -> None:
        self.signal_producer = Kospi200IndexBsOpfQuantileBreakoutSignalProducer()

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return self.signal_producer.datasets

    def build_signal(self, market) -> pd.DataFrame:
        return self.signal_producer.build(market).alpha

    def target_weights(self, signal: pd.Series) -> pd.Series:
        raise NotImplementedError

    def build_plan(self, market) -> PositionPlan:
        bundle = self.signal_producer.build(market)
        meta = bundle.meta
        tradable = bundle.context['tradable'].fillna(False).astype(bool)
        idx = bundle.alpha.index
        bs = meta['bs'].reindex(idx).ffill()
        ofs = meta['ofs'].reindex(idx).ffill()
        bs_u = meta['bs_upper'].reindex(idx).ffill()
        bs_l = meta['bs_lower'].reindex(idx).ffill()
        ofs_u = meta['ofs_upper'].reindex(idx).ffill()
        ofs_l = meta['ofs_lower'].reindex(idx).ffill()

        exposure = pd.Series(0.0, index=idx, dtype=float)
        prev = 0.0
        for dt in idx:
            vals = [bs.loc[dt], ofs.loc[dt], bs_u.loc[dt], bs_l.loc[dt], ofs_u.loc[dt], ofs_l.loc[dt]]
            if any(pd.isna(v) for v in vals):
                exposure.loc[dt] = prev
                continue
            bs_v, ofs_v, bs_u_v, bs_l_v, ofs_u_v, ofs_l_v = vals

            if self.variant == 'hysteresis':
                if bs_v > bs_u_v and ofs_v > ofs_u_v:
                    tgt = 1.0
                elif bs_v > bs_u_v and ofs_v > ofs_l_v:
                    tgt = 0.75
                elif prev >= 0.75 and bs_v > (bs_u_v + bs_l_v) / 2 and ofs_v > ofs_l_v:
                    tgt = prev
                elif bs_v < bs_l_v or ofs_v < ofs_l_v:
                    tgt = 0.0
                else:
                    tgt = 0.25 if prev == 0 else prev
                if abs(tgt - prev) < 0.02:
                    tgt = prev
            else:  # smoothing-sensitive
                if bs_v > bs_u_v and ofs_v > ofs_u_v:
                    raw = 1.0
                elif bs_v > bs_u_v and ofs_v > ofs_l_v:
                    raw = 0.75
                elif bs_v > bs_l_v and ofs_v > ofs_u_v:
                    raw = 0.5
                elif bs_v < bs_l_v or ofs_v < ofs_l_v:
                    raw = 0.0
                else:
                    raw = 0.25
                tgt = 0.7 * prev + 0.3 * raw
                if abs(tgt - prev) < 0.02:
                    tgt = prev

            exposure.loc[dt] = tgt
            prev = tgt

        weights = pd.DataFrame({'IKS200': exposure}, index=idx).where(tradable, 0.0).fillna(0.0)
        plan = PositionPlan(target_weights=weights, bucket_ledger=pd.DataFrame(), bucket_meta={}, validation={})
        plan.validation['variant'] = self.variant
        return plan


@dataclass(slots=True)
class Kospi200IndexBsOpfQuantileBreakoutV21Hysteresis(_BaseV21):
    variant: str = 'hysteresis'


@dataclass(slots=True)
class Kospi200IndexBsOpfQuantileBreakoutV21Smoothing(_BaseV21):
    variant: str = 'smoothing'
