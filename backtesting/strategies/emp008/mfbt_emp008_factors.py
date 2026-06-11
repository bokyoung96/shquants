from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.data import MarketData

from .mfbt_emp008_data import MfbtEmp008Config


def month_end_observations(frame: pd.DataFrame) -> pd.DataFrame:
    periods = frame.index.to_period("M")
    return frame.loc[~periods.duplicated(keep="last")]


def align_like_close(market: MarketData, key: str) -> pd.DataFrame:
    close = market.frames["close"]
    return market.frames[key].reindex(index=close.index, columns=close.columns)


def build_raw_mfbt_factors(market: MarketData, config: MfbtEmp008Config) -> dict[str, pd.DataFrame]:
    close = market.frames["close"].astype(float)
    return {
        "price_momentum": _price_momentum(close),
        "earnings_momentum": _earnings_momentum(market, config),
        "dividend_yield": _dividend_yield(market),
        "retail_flow": _retail_flow(market, config),
        "value": _value(market),
        "ln_market_cap": _ln_market_cap(market),
    }


def _monthly_output(template: pd.DataFrame, monthly_values: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(float("nan"), index=template.index, columns=template.columns, dtype=float)
    output.loc[monthly_values.index, monthly_values.columns] = monthly_values
    return output


def _price_momentum(close: pd.DataFrame) -> pd.DataFrame:
    trailing_high = close.rolling(252, min_periods=252).max()
    ratio = close.divide(trailing_high.where(trailing_high.gt(0.0)))
    monthly_ratio = month_end_observations(ratio)
    return _monthly_output(close, monthly_ratio)


def _earnings_momentum(market: MarketData, config: MfbtEmp008Config) -> pd.DataFrame:
    close = market.frames["close"].astype(float)
    op = align_like_close(market, "op_fwd_12m").astype(float)

    monthly_op = month_end_observations(op)
    previous_op = monthly_op.shift(1)
    denominator = previous_op.abs().mask(previous_op.abs().eq(0.0))
    growth = monthly_op.sub(previous_op).divide(denominator)
    growth = growth.where(monthly_op.notna() & previous_op.notna() & denominator.notna())

    low_op_extreme = monthly_op.lt(config.low_op_threshold) & growth.gt(config.extreme_growth_threshold)
    growth = growth.mask(low_op_extreme, 0.0)

    return _monthly_output(close, growth)


def _dividend_yield(market: MarketData) -> pd.DataFrame:
    close = market.frames["close"].astype(float)
    dps_ttm = align_like_close(market, "dps_ttm").astype(float)

    monthly_close = month_end_observations(close)
    monthly_dps = month_end_observations(dps_ttm).reindex(index=monthly_close.index, columns=monthly_close.columns)
    dividend_yield = monthly_dps.divide(monthly_close.where(monthly_close.gt(0.0)))

    return _monthly_output(close, dividend_yield)


def _retail_flow(market: MarketData, config: MfbtEmp008Config) -> pd.DataFrame:
    close = market.frames["close"].astype(float)
    retail_flow = align_like_close(market, "retail_flow").astype(float)
    sector = align_like_close(market, "sector_big").ffill()

    rolling_flow = retail_flow.rolling(
        config.retail_flow_lookback_days,
        min_periods=config.retail_flow_lookback_days,
    ).sum()
    monthly_flow = month_end_observations(rolling_flow)
    monthly_sector = month_end_observations(sector).reindex(index=monthly_flow.index, columns=monthly_flow.columns)
    monthly_metric = _sector_relative_retail_flow(monthly_flow, monthly_sector)

    return _monthly_output(close, monthly_metric)


def _sector_relative_retail_flow(monthly_flow: pd.DataFrame, monthly_sector: pd.DataFrame) -> pd.DataFrame:
    monthly_metric = pd.DataFrame(float("nan"), index=monthly_flow.index, columns=monthly_flow.columns, dtype=float)
    for date in monthly_flow.index:
        flows = monthly_flow.loc[date]
        sectors = monthly_sector.loc[date]
        valid = flows.notna() & sectors.notna()
        if not valid.any():
            continue

        sector_average = flows.loc[valid].groupby(sectors.loc[valid]).transform("mean")
        monthly_metric.loc[date, valid] = -(flows.loc[valid] - sector_average).astype(float)
    return monthly_metric


def _value(market: MarketData) -> pd.DataFrame:
    close = market.frames["close"].astype(float)
    market_cap = align_like_close(market, "market_cap").astype(float)
    free_cash_flow = align_like_close(market, "free_cash_flow").astype(float)
    debt = align_like_close(market, "interest_bearing_liability").astype(float)
    quick_asset = align_like_close(market, "quick_asset").astype(float)

    monthly_market_cap = month_end_observations(market_cap)
    monthly_fcf = month_end_observations(free_cash_flow).reindex(
        index=monthly_market_cap.index,
        columns=monthly_market_cap.columns,
    )
    monthly_debt = month_end_observations(debt).reindex(index=monthly_market_cap.index, columns=monthly_market_cap.columns)
    monthly_quick_asset = month_end_observations(quick_asset).reindex(
        index=monthly_market_cap.index,
        columns=monthly_market_cap.columns,
    )

    tev = monthly_market_cap.add(monthly_debt).sub(monthly_quick_asset)
    required = (
        monthly_market_cap.notna()
        & monthly_fcf.notna()
        & monthly_debt.notna()
        & monthly_quick_asset.notna()
        & tev.gt(0.0)
    )
    value = monthly_fcf.divide(tev.where(tev.gt(0.0))).where(required)

    return _monthly_output(close, value)


def _ln_market_cap(market: MarketData) -> pd.DataFrame:
    close = market.frames["close"].astype(float)
    market_cap = align_like_close(market, "market_cap").astype(float)
    monthly_market_cap = month_end_observations(market_cap)
    ln_market_cap = np.log(monthly_market_cap.where(monthly_market_cap.gt(0.0)))
    return _monthly_output(close, ln_market_cap)


__all__ = ["align_like_close", "build_raw_mfbt_factors", "month_end_observations"]
