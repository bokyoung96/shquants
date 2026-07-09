from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class BreakoutAtrConfig:
    range_end_hhmm: str = "0920"
    exit_hhmm: str = "1455"
    range_buffer_bps: float = 8.0
    atr_stop_multiplier: float = 1.0
    min_holding_days: int = 1
    round_trip_cost_bps: float = 35.0


@dataclass(frozen=True, slots=True)
class BreakoutAtrResult:
    config: BreakoutAtrConfig
    entries: pd.DataFrame
    trades: pd.DataFrame


class FiftyTwoWeekBreakoutAtrStrategy:
    def __init__(self, config: BreakoutAtrConfig | None = None) -> None:
        self.config = config or BreakoutAtrConfig()

    def run(self, intraday: pd.DataFrame, daily: pd.DataFrame) -> BreakoutAtrResult:
        return run_breakout_atr_strategy(intraday, daily, self.config)


def confirmed_breakout_entries(frame: pd.DataFrame, config: BreakoutAtrConfig = BreakoutAtrConfig()) -> pd.DataFrame:
    if frame.empty:
        return _empty_entries()
    required = {
        "ticker",
        "date",
        "ts",
        "hhmm",
        "close",
        "previous_intraday_close",
        "next_ts",
        "next_open",
        "atr",
        "prior_52w_close_high",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing entry columns: {sorted(missing)}")

    working = frame.sort_values(["ticker", "date", "ts"]).reset_index(drop=True).copy()
    grouped = working.groupby(["ticker", "date"], sort=False)
    working["confirmation_close"] = grouped["close"].shift(-1)
    working["confirmed_entry_time"] = grouped["next_ts"].shift(-1)
    working["confirmed_entry_open"] = grouped["next_open"].shift(-1)
    previous_close = working["previous_intraday_close"]
    prior_high = working["prior_52w_close_high"]
    breakout_bps = (working["close"].astype(float) / prior_high.astype(float) - 1.0) * 10_000.0

    mask = (
        working["confirmed_entry_open"].notna()
        & working["confirmed_entry_time"].notna()
        & working["atr"].notna()
        & prior_high.notna()
        & _strictly_above(working["close"], prior_high)
        & (previous_close.isna() | previous_close.le(prior_high))
        & working["hhmm"].astype(str).gt(config.range_end_hhmm)
        & working["hhmm"].astype(str).lt(config.exit_hhmm)
        & breakout_bps.ge(config.range_buffer_bps)
        & _strictly_above(working["confirmation_close"], prior_high)
    )
    entries = pd.DataFrame(
        {
            "ticker": working.loc[mask, "ticker"].astype(str),
            "date": pd.to_datetime(working.loc[mask, "date"]).dt.normalize(),
            "signal_time": pd.to_datetime(working.loc[mask, "ts"]),
            "entry_time": pd.to_datetime(working.loc[mask, "confirmed_entry_time"]),
            "entry_price": working.loc[mask, "confirmed_entry_open"].astype(float),
            "atr": working.loc[mask, "atr"].astype(float),
            "prior_52w_close_high": working.loc[mask, "prior_52w_close_high"].astype(float),
            "signal_score": breakout_bps.loc[mask].clip(lower=0.0).divide(10.0).astype(float),
        }
    )
    if entries.empty:
        return _empty_entries()
    return entries.sort_values(["ticker", "date", "signal_time"]).groupby(["ticker", "date"], sort=True).head(1).reset_index(drop=True)


def simulate_atr_continuation(
    entries: pd.DataFrame,
    daily: pd.DataFrame,
    config: BreakoutAtrConfig = BreakoutAtrConfig(),
) -> pd.DataFrame:
    if entries.empty:
        return _empty_trades()
    required = {"ticker", "date", "signal_time", "entry_time", "entry_price", "atr", "signal_score"}
    missing = required.difference(entries.columns)
    if missing:
        raise ValueError(f"missing entry columns: {sorted(missing)}")
    daily_required = {"ticker", "date", "close", "daily_low", "prior_52w_close_high"}
    daily_missing = daily_required.difference(daily.columns)
    if daily_missing:
        raise ValueError(f"missing daily columns: {sorted(daily_missing)}")

    rows: list[dict[str, object]] = []
    daily_groups = {
        str(ticker): group.assign(date=pd.to_datetime(group["date"]).dt.normalize()).sort_values("date").reset_index(drop=True)
        for ticker, group in daily.groupby("ticker", sort=True)
    }
    for ticker, ticker_entries in entries.groupby("ticker", sort=True):
        available_date = pd.Timestamp.min
        ticker_daily = daily_groups.get(str(ticker))
        if ticker_daily is None:
            continue
        for _, entry in ticker_entries.sort_values("signal_time").iterrows():
            entry_date = pd.Timestamp(entry["date"]).normalize()
            if entry_date <= available_date:
                continue
            trade = _continuation_trade(entry, ticker_daily, config)
            if trade is None:
                continue
            rows.append(trade)
            available_date = pd.Timestamp(trade["exit_time"]).normalize()
    if not rows:
        return _empty_trades()
    return pd.DataFrame(rows).sort_values(["entry_time", "ticker"]).reset_index(drop=True)


def run_breakout_atr_strategy(
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    config: BreakoutAtrConfig = BreakoutAtrConfig(),
) -> BreakoutAtrResult:
    entries = confirmed_breakout_entries(intraday, config)
    trades = simulate_atr_continuation(entries, daily, config)
    return BreakoutAtrResult(config=config, entries=entries, trades=trades)


def _continuation_trade(entry: pd.Series, daily: pd.DataFrame, config: BreakoutAtrConfig) -> dict[str, object] | None:
    entry_date = pd.Timestamp(entry["date"]).normalize()
    entry_price = float(entry["entry_price"])
    stop_price = entry_price - float(entry["atr"]) * config.atr_stop_multiplier
    holding_days = (pd.to_datetime(daily["date"]) - entry_date).dt.days
    exits = daily.loc[
        holding_days.ge(config.min_holding_days)
        & (daily["daily_low"].astype(float).le(stop_price) | daily["close"].astype(float).le(daily["prior_52w_close_high"].astype(float)))
    ]
    if exits.empty:
        valid = daily.loc[holding_days.ge(config.min_holding_days)]
        if valid.empty:
            return None
        exit_row = valid.iloc[-1]
        exit_reason = "end_of_data"
    else:
        exit_row = exits.iloc[0]
        exit_reason = "atr_stop" if float(exit_row["daily_low"]) <= stop_price else "new_high_lost"

    exit_price = stop_price if exit_reason == "atr_stop" else float(exit_row["close"])
    gross = exit_price / entry_price - 1.0
    exit_time = pd.Timestamp(exit_row["date"]).normalize() + pd.Timedelta(hours=15, minutes=30)
    return {
        "ticker": str(entry["ticker"]),
        "side": "long",
        "signal_time": pd.Timestamp(entry["signal_time"]),
        "entry_time": pd.Timestamp(entry["entry_time"]),
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "signal_score": float(entry["signal_score"]),
        "gross_return": gross,
        "net_return": gross - config.round_trip_cost_bps / 10_000.0,
        "exit_reason": exit_reason,
    }


def _strictly_above(value: pd.Series, benchmark: pd.Series) -> pd.Series:
    return value.astype(float).gt(benchmark.astype(float) + benchmark.astype(float).abs().mul(1e-12))


def _empty_entries() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "date",
            "signal_time",
            "entry_time",
            "entry_price",
            "atr",
            "prior_52w_close_high",
            "signal_score",
        ]
    )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "side",
            "signal_time",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "signal_score",
            "gross_return",
            "net_return",
            "exit_reason",
        ]
    )


__all__ = (
    "BreakoutAtrConfig",
    "BreakoutAtrResult",
    "FiftyTwoWeekBreakoutAtrStrategy",
    "confirmed_breakout_entries",
    "run_breakout_atr_strategy",
    "simulate_atr_continuation",
)
