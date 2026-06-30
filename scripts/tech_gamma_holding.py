from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

from scripts.tech_gamma_costs import net_return_after_costs
from scripts.tech_gamma_schemes import get_scheme

if TYPE_CHECKING:
    from scripts.run_tech_gamma_long_only import TechGammaConfig


@dataclass(frozen=True, slots=True)
class HoldingPosition:
    ticker: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    signal_score: float
    entry_date: pd.Timestamp
    atr_stop_price: float


class HoldingTrade(TypedDict):
    ticker: str
    side: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    signal_score: float
    gross_return: float
    net_return: float
    exit_reason: str


def simulate_continuation_holding(frame: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    rows: list[HoldingTrade] = []
    scheme = get_scheme(config.scheme)
    for ticker, group in frame.sort_values("ts").groupby("ticker", sort=True):
        position: HoldingPosition | None = None
        for date, day in group.groupby("date", sort=True):
            daily = _daily_row(day)
            if position is not None:
                exit_reason = _holding_exit_reason(position, daily, config)
                if exit_reason is not None:
                    rows.append(_trade(position, daily, exit_reason))
                    position = None
                continue
            candidates = day[day.apply(lambda row: scheme.intraday_entry_ok(row, config), axis=1)]
            if candidates.empty:
                continue
            signal = candidates.iloc[0]
            position = HoldingPosition(
                ticker=str(ticker),
                signal_time=pd.Timestamp(signal["ts"]),
                entry_time=pd.Timestamp(signal["next_ts"]),
                entry_price=float(signal["next_open"]),
                signal_score=float(signal["signal_score"]),
                entry_date=pd.Timestamp(date),
                atr_stop_price=float(signal["next_open"]) - float(signal["atr"]) * config.atr_stop_multiplier,
            )
        if position is not None:
            rows.append(_trade(position, _daily_row(group.iloc[-1:]), "end_of_data"))
    return pd.DataFrame(rows)


def _daily_row(day: pd.DataFrame) -> pd.Series:
    last = day.iloc[-1].copy()
    last["daily_low"] = float(day["low"].min())
    return last


def _holding_exit_reason(position: HoldingPosition, row: pd.Series, config: TechGammaConfig) -> str | None:
    holding_days = (pd.Timestamp(row["date"]) - position.entry_date).days
    if holding_days < config.min_holding_days:
        return None
    if float(row["daily_low"]) <= position.atr_stop_price:
        return "atr_stop"
    if float(row["close"]) <= float(row["prior_52w_close_high"]):
        return "new_high_lost"
    return None


def _trade(position: HoldingPosition, row: pd.Series, exit_reason: str) -> HoldingTrade:
    exit_price = position.atr_stop_price if exit_reason == "atr_stop" else float(row["close"])
    gross = exit_price / position.entry_price - 1.0
    return {
        "ticker": position.ticker,
        "side": "long",
        "signal_time": position.signal_time,
        "entry_time": position.entry_time,
        "exit_time": pd.Timestamp(row["ts"]),
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "signal_score": position.signal_score,
        "gross_return": gross,
        "net_return": net_return_after_costs(gross),
        "exit_reason": exit_reason,
    }
