from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


class MfbtFactor(Protocol):
    name: str

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        ...

    def build(self, market: MarketData) -> pd.DataFrame:
        ...


def _month_end_observations(frame: pd.DataFrame) -> pd.DataFrame:
    periods = frame.index.to_period("M")
    return frame.loc[~periods.duplicated(keep="last")]


def _universe_mask(market: MarketData, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    if market.universe is None:
        return pd.DataFrame(True, index=index, columns=columns)
    universe = market.universe.reindex(index=index, columns=columns)
    return universe.astype("boolean").fillna(False).astype(bool)


def _aligned_frame(market: MarketData, key: str) -> pd.DataFrame:
    close = market.frames["close"]
    return market.frames[key].reindex(index=close.index, columns=close.columns).astype(float)


def _monthly_output(
    template: pd.DataFrame,
    monthly_score: pd.DataFrame,
    market: MarketData,
    *,
    fill_missing: float | None = None,
) -> pd.DataFrame:
    values = monthly_score
    if fill_missing is not None:
        values = values.fillna(fill_missing).astype(float)
    score = pd.DataFrame(float("nan"), index=template.index, columns=template.columns, dtype=float)
    score.loc[values.index, values.columns] = values
    return score.where(_universe_mask(market, score.index, score.columns))


def _score_row(row: pd.Series, quantile_count: int) -> pd.Series:
    valid = row.dropna()
    score = pd.Series(float("nan"), index=row.index, dtype=float)
    if valid.empty:
        return score

    ranks = valid.rank(method="average", ascending=True)
    buckets = ((ranks - 1.0) * quantile_count / len(valid)).astype(int)
    score.loc[valid.index] = buckets.clip(lower=0, upper=quantile_count - 1).astype(float)
    return score


def _score_frame(frame: pd.DataFrame, market: MarketData, quantile_count: int) -> pd.DataFrame:
    universe = _universe_mask(market, frame.index, frame.columns)
    score = frame.where(universe).apply(_score_row, axis=1, quantile_count=quantile_count)
    return score.where(universe)


def _quarter_lagged_financials(frame: pd.DataFrame, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    monthly = _month_end_observations(frame)
    source_monthly = monthly.loc[monthly.index.month.isin((3, 5, 8, 11))]
    if source_monthly.empty:
        return pd.DataFrame(float("nan"), index=signal_dates, columns=frame.columns, dtype=float)

    source_monthly = source_monthly.copy()
    source_monthly.index = source_monthly.index.to_period("M")
    source_monthly = source_monthly.loc[~source_monthly.index.duplicated(keep="last")]

    source_periods = pd.PeriodIndex([_value_source_period(period) for period in signal_dates.to_period("M")], freq="M")
    available = source_monthly.reindex(source_periods)
    available.index = signal_dates
    return available.astype(float)


def _value_source_period(signal_period: pd.Period) -> pd.Period:
    if signal_period.month in (4, 5):
        return pd.Period(year=signal_period.year, month=3, freq="M")
    if signal_period.month in (6, 7, 8):
        return pd.Period(year=signal_period.year, month=5, freq="M")
    if signal_period.month in (9, 10, 11):
        return pd.Period(year=signal_period.year, month=8, freq="M")
    if signal_period.month == 12:
        return pd.Period(year=signal_period.year, month=11, freq="M")
    return pd.Period(year=signal_period.year - 1, month=11, freq="M")


@dataclass(slots=True)
class PriceMomentumFactor:
    # Signal: pass names trading above 80% of their 252-day close high.
    high_lookback: int = 252
    threshold: float = 0.8
    name: str = "price_momentum"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_ADJ_C,)

    def build(self, market: MarketData) -> pd.DataFrame:
        close = market.frames["close"]
        high = close.rolling(self.high_lookback, min_periods=self.high_lookback).max()
        ratio = close.divide(high)
        monthly_ratio = _month_end_observations(ratio)
        monthly_score = monthly_ratio.gt(self.threshold).astype(float).where(monthly_ratio.notna())
        return _monthly_output(close, monthly_score, market)


@dataclass(slots=True)
class EarningsMomentumFactor:
    # Signal: score monthly 12MF operating-profit estimate growth into 0-4 cross-sectional buckets.
    low_op_threshold: float = 100_000_000_000.0
    extreme_growth_threshold: float = 0.50
    quantile_count: int = 5
    name: str = "earnings_momentum"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_OP_FWD_12M,)

    def build(self, market: MarketData) -> pd.DataFrame:
        close = market.frames["close"]
        op_fwd_12m = _aligned_frame(market, "op_fwd_12m")

        monthly_op = _month_end_observations(op_fwd_12m)
        previous = monthly_op.shift(1)
        denominator = previous.abs().mask(previous.abs().eq(0.0))
        growth = monthly_op.sub(previous).divide(denominator)
        growth = growth.where(monthly_op.notna() & previous.notna())

        low_op_extreme = monthly_op.lt(self.low_op_threshold) & growth.gt(self.extreme_growth_threshold)
        growth = growth.mask(low_op_extreme, 0.0)

        monthly_score = _score_frame(growth, market, self.quantile_count)
        return _monthly_output(close, monthly_score, market)


@dataclass(slots=True)
class DividendYieldFactor:
    # Signal: score monthly dividend yield into 0-4 buckets, then add a 3-year year-end cash-dividend growth bonus.
    quantile_count: int = 5
    name: str = "dividend_yield"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_ADJ_C, DatasetId.QW_DPS_TTM, DatasetId.QW_DIVIDEND_CASH_TTM)

    def build(self, market: MarketData) -> pd.DataFrame:
        close = market.frames["close"].astype(float)
        dps_ttm = _aligned_frame(market, "dps_ttm")
        dividend_cash_ttm = _aligned_frame(market, "dividend_cash_ttm")

        monthly_close = _month_end_observations(close)
        monthly_dps = _month_end_observations(dps_ttm).reindex(index=monthly_close.index, columns=monthly_close.columns)
        dividend_yield = monthly_dps.divide(monthly_close.where(monthly_close.gt(0.0)))

        base_score = _score_frame(dividend_yield, market, self.quantile_count)
        monthly_score = base_score.add(
            self._three_year_increase_bonus(dividend_cash_ttm, base_score.index, base_score.columns),
            fill_value=0.0,
        )
        monthly_score = monthly_score.where(base_score.notna())
        monthly_score = monthly_score.where(_universe_mask(market, monthly_score.index, monthly_score.columns))

        return _monthly_output(close, monthly_score, market)

    @staticmethod
    def _three_year_increase_bonus(
        dividend_cash: pd.DataFrame,
        signal_dates: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> pd.DataFrame:
        years = pd.Index(dividend_cash.index.year)
        year_end_cash = dividend_cash.loc[~years.duplicated(keep="last")].copy()
        year_end_cash.index = year_end_cash.index.year
        bonus = pd.DataFrame(0.0, index=signal_dates, columns=columns, dtype=float)

        for signal_date in signal_dates:
            completed_year = signal_date.year if signal_date.month == 12 else signal_date.year - 1
            years = [completed_year - 2, completed_year - 1, completed_year]
            if any(year not in year_end_cash.index for year in years):
                continue
            first = year_end_cash.loc[years[0], columns]
            second = year_end_cash.loc[years[1], columns]
            third = year_end_cash.loc[years[2], columns]
            increased = first.notna() & second.notna() & third.notna() & first.lt(second) & second.lt(third)
            bonus.loc[signal_date, increased.index] = increased.astype(float)
        return bonus


@dataclass(slots=True)
class RetailFlowFactor:
    # Signal: score sectors by average 252-day cumulative retail net selling, then assign scores to K200 members.
    lookback: int = 252
    quantile_count: int = 5
    name: str = "retail_flow"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_RETAIL, DatasetId.QW_WI_SEC_26)

    def build(self, market: MarketData) -> pd.DataFrame:
        close = market.frames["close"]
        retail_flow = _aligned_frame(market, "retail_flow")
        sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()

        rolling_flow = retail_flow.rolling(self.lookback, min_periods=self.lookback).sum()
        monthly_flow = _month_end_observations(rolling_flow)
        monthly_sector = _month_end_observations(sector).reindex(index=monthly_flow.index, columns=monthly_flow.columns)
        monthly_universe = _universe_mask(market, monthly_flow.index, monthly_flow.columns)

        monthly_score = pd.DataFrame(float("nan"), index=monthly_flow.index, columns=monthly_flow.columns, dtype=float)
        for date in monthly_flow.index:
            flows = monthly_flow.loc[date]
            sectors = monthly_sector.loc[date]
            universe = monthly_universe.loc[date]
            valid = universe & flows.notna() & sectors.notna()
            if not valid.any():
                continue

            sector_flow = flows.loc[valid].groupby(sectors.loc[valid]).mean()
            sector_score = _score_row(-sector_flow, self.quantile_count)
            member_scores = sectors.map(sector_score).astype(float)
            monthly_score.loc[date] = member_scores.where(universe)

        return _monthly_output(close, monthly_score, market)


@dataclass(slots=True)
class ValueFactor:
    # Signal: score FCF/TEV using lagged quarterly financials and month-end market cap.
    quantile_count: int = 5
    name: str = "value"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_MKTCAP,
            DatasetId.QW_FCF,
            DatasetId.QW_INT_BEARING_LIAB_NFQ0,
            DatasetId.QW_QUICK_ASSETS_NFQ0,
        )

    def build(self, market: MarketData) -> pd.DataFrame:
        close = market.frames["close"]
        market_cap = _aligned_frame(market, "market_cap")
        free_cash_flow = _aligned_frame(market, "free_cash_flow")
        interest_bearing_liability = _aligned_frame(market, "interest_bearing_liability")
        quick_asset = _aligned_frame(market, "quick_asset")

        monthly_market_cap = _month_end_observations(market_cap)
        signal_dates = pd.DatetimeIndex(monthly_market_cap.index)
        lagged_fcf = _quarter_lagged_financials(free_cash_flow, signal_dates)
        lagged_debt = _quarter_lagged_financials(interest_bearing_liability, signal_dates)
        lagged_quick_asset = _quarter_lagged_financials(quick_asset, signal_dates)

        tev = monthly_market_cap.add(lagged_debt).sub(lagged_quick_asset)
        required = monthly_market_cap.notna() & lagged_fcf.notna() & lagged_debt.notna() & lagged_quick_asset.notna()
        value_metric = lagged_fcf.divide(tev)
        value_metric = value_metric.where(required)
        value_metric = value_metric.mask(required & tev.le(0.0), float("-inf"))

        monthly_score = _score_frame(value_metric, market, self.quantile_count)
        monthly_score = monthly_score.where(value_metric.notna())
        monthly_score = monthly_score.mask(required & tev.le(0.0), 0.0)

        return _monthly_output(close, monthly_score, market)


@dataclass(slots=True)
class Mfbt(ComposableStrategy):
    top_n: int = 20
    high_lookback: int = 252
    price_momentum_threshold: float = 0.8

    def __post_init__(self) -> None:
        self.signal_producer = _MfbtSignal(
            factors=(
                PriceMomentumFactor(
                    high_lookback=self.high_lookback,
                    threshold=self.price_momentum_threshold,
                ),
                EarningsMomentumFactor(),
                DividendYieldFactor(),
                RetailFlowFactor(),
                ValueFactor(),
            ),
        )
        self.construction_rule = _MfbtConstruction(top_n=self.top_n)


@dataclass(slots=True)
class _MfbtSignal:
    factors: tuple[MfbtFactor, ...]

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        datasets: list[DatasetId] = []
        for factor in self.factors:
            for dataset in factor.datasets:
                if dataset not in datasets:
                    datasets.append(dataset)
        return tuple(datasets)

    def build(self, market: MarketData) -> SignalBundle:
        factor_frames = {factor.name: factor.build(market) for factor in self.factors}
        common_dates = set.intersection(*(set(frame.dropna(how="all").index) for frame in factor_frames.values()))
        factor_frames = {
            name: frame.where(pd.Series(frame.index.isin(common_dates), index=frame.index), axis=0)
            for name, frame in factor_frames.items()
        }
        price_momentum = factor_frames["price_momentum"]
        return SignalBundle(
            alpha=price_momentum,
            context={"tradable": price_momentum.eq(1.0)},
            meta=factor_frames,
        )


@dataclass(slots=True)
class _MfbtConstruction:
    top_n: int = 20

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        eligible = alpha.gt(0.0)
        ranks = alpha.where(eligible).rank(axis=1, ascending=False, method="first", na_option="bottom")
        selection_mask = ranks.le(self.top_n) & eligible
        selected_count = selection_mask.sum(axis=1).clip(upper=self.top_n)
        denominator = selected_count.astype(float).where(selected_count.ne(0), float("nan"))
        base_target_weights = selection_mask.astype(float).div(denominator, axis=0).fillna(0.0).astype(float)
        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=selection_mask,
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )
