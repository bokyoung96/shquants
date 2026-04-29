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
class Kospi200IndexBsOpfQuantileBreakout(RegisteredStrategy):
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
            bs_v = bs.loc[dt]
            ofs_v = ofs.loc[dt]
            bs_u_v = bs_u.loc[dt]
            bs_l_v = bs_l.loc[dt]
            ofs_u_v = ofs_u.loc[dt]
            ofs_l_v = ofs_l.loc[dt]
            if pd.isna(bs_v) or pd.isna(ofs_v) or pd.isna(bs_u_v) or pd.isna(bs_l_v) or pd.isna(ofs_u_v) or pd.isna(ofs_l_v):
                exposure.loc[dt] = prev
                continue

            if bs_v > bs_u_v and ofs_v > ofs_u_v:
                prev = 1.0
            elif bs_v > bs_u_v and ofs_v > ofs_l_v:
                prev = 0.75
            elif bs_v > bs_l_v and ofs_v > ofs_u_v:
                prev = max(prev, 0.5)
            elif bs_v < bs_l_v or ofs_v < ofs_l_v:
                prev = 0.0
            else:
                prev = min(prev, 0.25) if prev > 0 else 0.25
            exposure.loc[dt] = prev

        weights = pd.DataFrame({'IKS200': exposure}, index=idx).where(tradable, 0.0).fillna(0.0)
        return PositionPlan(target_weights=weights, bucket_ledger=pd.DataFrame(), bucket_meta={}, validation={})
