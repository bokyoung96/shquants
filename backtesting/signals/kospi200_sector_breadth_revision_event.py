from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class Kospi200SectorBreadthRevisionEventSignalProducer:
    sector_momentum_lookback: int = 126
    stock_momentum_lookback: int = 63
    high_lookback: int = 252
    op_revision_lookback: int = 21
    near_high_ratio: float = 0.98
    sector_top_ratio: float = 0.20
    entry_score_threshold: float = 0.75

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFY1,
            DatasetId.QW_WICS_SEC_BIG,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        op_fwd_q1 = market.frames["op_fwd_q1"]
        op_fwd_y1 = market.frames["op_fwd"]
        sector = market.frames["sector_big"]

        stock_momentum = close.pct_change(self.stock_momentum_lookback, fill_method=None)
        sector_momentum = self._sector_aggregate(stock_momentum, sector, close.columns)
        new_high_ratio = close.divide(close.rolling(self.high_lookback, min_periods=126).max())
        near_high = new_high_ratio.ge(self.near_high_ratio)
        sector_breadth = self._sector_aggregate(near_high.astype(float), sector, close.columns)

        op_q1_revision = op_fwd_q1.pct_change(self.op_revision_lookback, fill_method=None)
        op_y1_revision = op_fwd_y1.pct_change(self.op_revision_lookback, fill_method=None)
        revision_level = 0.6 * op_q1_revision + 0.4 * op_y1_revision
        revision_accel = revision_level - revision_level.shift(self.op_revision_lookback)

        sector_momentum_score = self._cross_sectional_zscore(sector_momentum)
        sector_breadth_score = self._cross_sectional_zscore(sector_breadth)
        stock_score = self._cross_sectional_zscore(stock_momentum)
        revision_score = self._cross_sectional_zscore(revision_level)
        revision_accel_score = self._cross_sectional_zscore(revision_accel)
        new_high_score = self._cross_sectional_zscore(new_high_ratio)

        sector_composite = 0.55 * sector_momentum_score + 0.45 * sector_breadth_score
        alpha = (
            0.35 * sector_composite
            + 0.30 * revision_score
            + 0.20 * revision_accel_score
            + 0.10 * new_high_score
            + 0.05 * stock_score
        )
        tradable = close.notna() & alpha.notna() & sector.notna()
        sector_gate = self._top_ratio_mask(sector_composite, self.sector_top_ratio)

        signal_on = (
            tradable
            & sector_gate
            & near_high
            & revision_level.gt(0.02)
            & revision_accel.gt(0.0)
            & stock_momentum.gt(0.0)
            & sector_breadth.gt(0.18)
            & alpha.gt(self.entry_score_threshold)
        )
        signal_off = ~signal_on | ~tradable
        event_prev = signal_on.shift(1).fillna(False)
        eligible_entry = signal_on & ~event_prev

        return SignalBundle(
            alpha=alpha.where(signal_on),
            context={
                "close": close,
                "tradable": tradable,
                "eligible_entry": eligible_entry.fillna(False),
                "eligible_exit": signal_off.fillna(False),
                "sector_gate": sector_gate.fillna(False),
                "signal_on": signal_on.fillna(False),
            },
            meta={
                "stock_momentum": stock_momentum,
                "sector_momentum": sector_momentum,
                "sector_breadth": sector_breadth,
                "revision_level": revision_level,
                "revision_accel": revision_accel,
                "new_high_ratio": new_high_ratio,
                "sector_composite": sector_composite,
            },
        )

    def _sector_aggregate(self, values: pd.DataFrame, sector: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
        rows: dict[pd.Timestamp, pd.Series] = {}
        for date in values.index:
            sector_row = sector.loc[date]
            value_row = values.loc[date]
            valid = pd.DataFrame({"sector": sector_row, "value": value_row}).dropna()
            if valid.empty:
                rows[date] = pd.Series(index=columns, dtype=float)
                continue
            sector_mean = valid.groupby("sector", sort=False)["value"].mean()
            rows[date] = sector_row.map(sector_mean).reindex(columns)
        return pd.DataFrame.from_dict(rows, orient="index").reindex(index=values.index, columns=columns)

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).replace(0.0, pd.NA)
        return frame.sub(mean, axis=0).div(std, axis=0)

    @staticmethod
    def _top_ratio_mask(frame: pd.DataFrame, top_ratio: float) -> pd.DataFrame:
        ratio = min(max(float(top_ratio), 0.0), 1.0)
        rows: dict[pd.Timestamp, pd.Series] = {}
        for date in frame.index:
            row = frame.loc[date].dropna().sort_values(ascending=False)
            if row.empty:
                rows[date] = pd.Series(False, index=frame.columns, dtype=bool)
                continue
            keep = max(1, int(len(row) * ratio + 0.999999))
            allowed = row.index[:keep]
            rows[date] = pd.Series(frame.columns.isin(allowed), index=frame.columns, dtype=bool)
        return pd.DataFrame.from_dict(rows, orient="index").reindex(index=frame.index, columns=frame.columns).fillna(False)
