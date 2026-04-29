from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class CreativeEventAsymmetrySignalProducer:
    revision_threshold: float = 0.04
    flow_lookback: int = 20
    support_lookback: int = 20

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_RETAIL,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFY1,
            DatasetId.QW_EPS_NFY1,
            DatasetId.QW_SHA_OUT,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames['close']
        volume = market.frames['volume']
        foreign_flow = market.frames['foreign_flow']
        inst_flow = market.frames['inst_flow']
        retail_flow = market.frames['retail_flow']
        op_fwd_q1 = market.frames['op_fwd_q1']
        op_fwd = market.frames['op_fwd']
        eps_fwd = market.frames['eps_fwd']
        shares_out = market.frames['shares_out']

        adv = (close.abs() * volume).rolling(self.flow_lookback).mean()
        foreign_intensity = foreign_flow.rolling(self.flow_lookback).sum().divide(adv)
        inst_intensity = inst_flow.rolling(15).sum().divide(adv)
        retail_intensity = retail_flow.rolling(10).sum().divide(adv)

        q1_revision = op_fwd_q1.pct_change(fill_method=None)
        fy1_revision = op_fwd.pct_change(fill_method=None)
        eps_revision = eps_fwd.pct_change(fill_method=None)
        revision_raw = 0.45 * q1_revision + 0.35 * fy1_revision + 0.20 * eps_revision

        share_change = shares_out.pct_change(fill_method=None)
        price_support = close.pct_change(self.support_lookback, fill_method=None)
        short_rebound = close.pct_change(5, fill_method=None)

        event_now = (
            revision_raw.gt(self.revision_threshold)
            & foreign_intensity.gt(0.0)
            & inst_intensity.gt(0.0)
            & retail_intensity.lt(0.0)
            & price_support.gt(-0.03)
            & close.notna()
        )
        buyback_overlay = share_change.le(-0.005)

        alpha_raw = (
            0.45 * revision_raw
            + 0.20 * foreign_intensity
            + 0.20 * inst_intensity
            - 0.10 * retail_intensity
            + 0.05 * short_rebound
            + 0.10 * (-share_change.clip(upper=0.0))
        ).replace([float('inf'), float('-inf')], pd.NA)
        alpha = self._cross_sectional_zscore(alpha_raw)

        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_add_1 = event_now & event_prev & foreign_intensity.gt(foreign_intensity.rolling(5).mean())
        eligible_add_2 = eligible_add_1 & short_rebound.gt(0.0) & buyback_overlay.fillna(False)
        eligible_exit = (~event_now) | alpha.isna() | price_support.lt(-0.08)
        tradable = close.notna() & alpha.notna()

        return SignalBundle(
            alpha=alpha.where(event_now & tradable),
            context={
                'close': close,
                'tradable': tradable,
                'eligible_entry': eligible_entry.fillna(False),
                'eligible_add_1': eligible_add_1.fillna(False),
                'eligible_add_2': eligible_add_2.fillna(False),
                'eligible_exit': eligible_exit.fillna(False),
            },
            meta={
                'revision_raw': revision_raw,
                'foreign_intensity': foreign_intensity,
                'inst_intensity': inst_intensity,
                'retail_intensity': retail_intensity,
                'share_change': share_change,
                'price_support': price_support,
            },
        )

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)
