from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData

from .base import SignalBundle


@dataclass(frozen=True, slots=True)
class Kospi200SectorMomentumOpEventSignalProducer:
    sector_momentum_lookback: int = 126
    stock_momentum_lookback: int = 63
    high_lookback: int = 252
    op_revision_lookback: int = 21
    sector_top_ratio: float = 0.35
    entry_score_threshold: float = 0.25
    exit_score_threshold: float = -0.5
    new_high_entry_ratio: float = 0.97
    new_high_exit_ratio: float = 0.92

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

        sector_momentum = self._sector_momentum(close=close, sector=sector)
        sector_score = self._cross_sectional_zscore(sector_momentum)
        stock_momentum = close.pct_change(self.stock_momentum_lookback, fill_method=None)
        stock_score = self._cross_sectional_zscore(stock_momentum)
        new_high_ratio = close.divide(close.rolling(self.high_lookback, min_periods=126).max())

        op_q1_revision = op_fwd_q1.pct_change(self.op_revision_lookback, fill_method=None)
        op_y1_revision = op_fwd_y1.pct_change(self.op_revision_lookback, fill_method=None)
        revision_raw = 0.6 * op_q1_revision + 0.4 * op_y1_revision
        revision_score = self._cross_sectional_zscore(revision_raw)
        new_high_score = self._cross_sectional_zscore(new_high_ratio)

        alpha = 0.55 * sector_score + 0.30 * revision_score + 0.15 * new_high_score + 0.10 * stock_score
        tradable = close.notna() & alpha.notna() & sector.notna()

        sector_gate = self._top_ratio_mask(sector_score, self.sector_top_ratio)
        event_now = (
            tradable
            & sector_gate
            & new_high_ratio.ge(self.new_high_entry_ratio)
            & revision_raw.gt(0.0)
            & alpha.gt(self.entry_score_threshold)
        )
        event_prev = event_now.shift(1).fillna(False)
        eligible_entry = event_now & ~event_prev
        eligible_add_1 = event_now & event_prev & revision_raw.gt(revision_raw.shift(1))
        eligible_add_2 = event_now & event_prev & stock_momentum.gt(0.0)
        eligible_exit = (
            alpha.lt(self.exit_score_threshold)
            | new_high_ratio.lt(self.new_high_exit_ratio)
            | revision_raw.le(0.0)
            | ~sector_gate
            | ~tradable
        )

        return SignalBundle(
            alpha=alpha.where(event_now),
            context={
                "close": close,
                "tradable": tradable,
                "eligible_entry": eligible_entry.fillna(False),
                "eligible_add_1": eligible_add_1.fillna(False),
                "eligible_add_2": eligible_add_2.fillna(False),
                "eligible_exit": eligible_exit.fillna(False),
                "sector_gate": sector_gate.fillna(False),
            },
            meta={
                "sector_momentum": sector_momentum,
                "sector_score": sector_score,
                "stock_momentum": stock_momentum,
                "revision_raw": revision_raw,
                "revision_score": revision_score,
                "new_high_ratio": new_high_ratio,
                "new_high_score": new_high_score,
            },
        )

    def _sector_momentum(self, *, close: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
        stock_momentum = close.pct_change(self.sector_momentum_lookback, fill_method=None)
        rows: dict[pd.Timestamp, pd.Series] = {}
        for date in stock_momentum.index:
            sector_row = sector.loc[date]
            momentum_row = stock_momentum.loc[date]
            valid = pd.DataFrame({"sector": sector_row, "momentum": momentum_row}).dropna()
            if valid.empty:
                rows[date] = pd.Series(index=close.columns, dtype=float)
                continue
            sector_mean = valid.groupby("sector", sort=False)["momentum"].mean()
            rows[date] = sector_row.map(sector_mean).reindex(close.columns)
        return pd.DataFrame.from_dict(rows, orient="index").reindex(index=close.index, columns=close.columns)

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
