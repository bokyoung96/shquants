from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class Kospi200IndexBsOpfQuantileBreakoutSignalProducer:
    high_lookback: int = 252
    bs_smoothing: int = 5
    op_revision_lookback: int = 20
    ofs_smoothing: int = 20
    band_lookback: int = 252
    upper_q: float = 0.7
    lower_q: float = 0.3
    revision_up_threshold: float = 0.0
    revision_down_threshold: float = 0.0
    benchmark_code: str = 'IKS200'

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFY1,
            DatasetId.QW_K200_YN,
            DatasetId.QW_BM,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames['close']
        op_fwd_q1 = market.frames['op_fwd_q1']
        op_fwd_y1 = market.frames['op_fwd']
        benchmark = market.frames['benchmark'][self.benchmark_code].astype(float)

        universe = market.universe if market.universe is not None else market.frames['k200_yn'].fillna(0).astype(bool)
        universe = universe.reindex_like(close).fillna(False).astype(bool)

        rolling_high = close.where(universe).rolling(self.high_lookback, min_periods=126).max()
        rolling_low = close.where(universe).rolling(self.high_lookback, min_periods=126).min()
        new_high = close.ge(rolling_high) & universe & rolling_high.notna()
        new_low = close.le(rolling_low) & universe & rolling_low.notna()
        universe_count = universe.sum(axis=1).replace(0, float('nan'))
        nh_ratio = new_high.sum(axis=1).divide(universe_count).astype(float)
        nl_ratio = new_low.sum(axis=1).divide(universe_count).astype(float)
        bs_raw = nh_ratio - nl_ratio
        bs = bs_raw.rolling(self.bs_smoothing, min_periods=max(2, self.bs_smoothing // 2)).mean()

        op_q1_revision = op_fwd_q1.pct_change(self.op_revision_lookback, fill_method=None)
        op_y1_revision = op_fwd_y1.pct_change(self.op_revision_lookback, fill_method=None)
        revision = 0.6 * op_q1_revision + 0.4 * op_y1_revision
        consensus_valid = revision.notna() & universe
        valid_count = consensus_valid.sum(axis=1).replace(0, float('nan'))
        op_up = (revision > self.revision_up_threshold) & consensus_valid
        op_down = (revision < -self.revision_down_threshold) & consensus_valid
        op_up_ratio = op_up.sum(axis=1).divide(valid_count).astype(float)
        op_down_ratio = op_down.sum(axis=1).divide(valid_count).astype(float)
        ofs_raw = op_up_ratio - op_down_ratio
        ofs = ofs_raw.rolling(self.ofs_smoothing, min_periods=max(2, self.ofs_smoothing // 2)).mean()

        bs_upper = bs.rolling(self.band_lookback, min_periods=max(60, self.band_lookback // 2)).quantile(self.upper_q)
        bs_lower = bs.rolling(self.band_lookback, min_periods=max(60, self.band_lookback // 2)).quantile(self.lower_q)
        ofs_upper = ofs.rolling(self.band_lookback, min_periods=max(60, self.band_lookback // 2)).quantile(self.upper_q)
        ofs_lower = ofs.rolling(self.band_lookback, min_periods=max(60, self.band_lookback // 2)).quantile(self.lower_q)

        alpha = pd.DataFrame({'IKS200': benchmark}, index=benchmark.index)
        return SignalBundle(
            alpha=alpha,
            context={
                'tradable': pd.DataFrame({'IKS200': benchmark.notna()}, index=benchmark.index),
                'benchmark_close': pd.DataFrame({'IKS200': benchmark}, index=benchmark.index),
            },
            meta={
                'nh_ratio': nh_ratio,
                'nl_ratio': nl_ratio,
                'bs_raw': bs_raw,
                'bs': bs,
                'op_up_ratio': op_up_ratio,
                'op_down_ratio': op_down_ratio,
                'ofs_raw': ofs_raw,
                'ofs': ofs,
                'bs_upper': bs_upper,
                'bs_lower': bs_lower,
                'ofs_upper': ofs_upper,
                'ofs_lower': ofs_lower,
            },
        )
