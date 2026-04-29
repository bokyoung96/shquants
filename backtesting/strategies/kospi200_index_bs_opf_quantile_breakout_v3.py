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
class Kospi200IndexBsOpfQuantileBreakoutV3(RegisteredStrategy):
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
        benchmark = bundle.context['benchmark_close']['IKS200'].reindex(bundle.alpha.index).ffill()
        idx = bundle.alpha.index
        bs = meta['bs'].reindex(idx).ffill()
        ofs = meta['ofs'].reindex(idx).ffill()
        bs_u = meta['bs_upper'].reindex(idx).ffill()
        bs_l = meta['bs_lower'].reindex(idx).ffill()
        ofs_u = meta['ofs_upper'].reindex(idx).ffill()
        ofs_l = meta['ofs_lower'].reindex(idx).ffill()

        dd = benchmark / benchmark.cummax() - 1.0
        target = pd.Series(0.0, index=idx, dtype=float)
        applied = pd.Series(0.0, index=idx, dtype=float)
        prev_target = 0.0
        prev_applied = 0.0

        for dt in idx:
            bs_v = bs.loc[dt]
            ofs_v = ofs.loc[dt]
            bs_u_v = bs_u.loc[dt]
            bs_l_v = bs_l.loc[dt]
            ofs_u_v = ofs_u.loc[dt]
            ofs_l_v = ofs_l.loc[dt]
            dd_v = dd.loc[dt]
            if pd.isna(bs_v) or pd.isna(ofs_v) or pd.isna(bs_u_v) or pd.isna(bs_l_v) or pd.isna(ofs_u_v) or pd.isna(ofs_l_v):
                target.loc[dt] = prev_target
                applied.loc[dt] = prev_applied
                continue

            # hysteresis-like target generation
            if bs_v > bs_u_v and ofs_v > ofs_u_v:
                raw = 1.0
            elif bs_v > bs_u_v and ofs_v > ofs_l_v:
                raw = 0.75
            elif bs_v > bs_l_v and ofs_v > ofs_u_v:
                raw = max(prev_target, 0.5)
            elif bs_v < bs_l_v or ofs_v < ofs_l_v:
                raw = 0.0
            else:
                raw = prev_target if prev_target > 0 else 0.25

            # drawdown control on cap
            if dd_v <= -0.20:
                cap = 0.0
            elif dd_v <= -0.15:
                cap = 0.25
            elif dd_v <= -0.10:
                cap = 0.50
            else:
                cap = 1.0

            tgt = min(raw, cap)

            # minimum change filter: ignore smaller than 0.02
            if abs(tgt - prev_target) < 0.02:
                tgt = prev_target

            # applied smoothing
            app = 0.5 * prev_applied + 0.5 * tgt

            target.loc[dt] = tgt
            applied.loc[dt] = app
            prev_target = tgt
            prev_applied = app

        weights = pd.DataFrame({'IKS200': applied}, index=idx).where(tradable, 0.0).fillna(0.0)
        plan = PositionPlan(target_weights=weights, bucket_ledger=pd.DataFrame(), bucket_meta={}, validation={})
        plan.validation['v3_fixed_spec'] = {
            'drawdown_caps': {'-10%': 0.50, '-15%': 0.25, '-20%': 0.0},
            'min_change': 0.02,
            'smoothing_lambda': 0.5,
        }
        return plan
