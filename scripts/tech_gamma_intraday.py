from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict, assert_never

import pandas as pd

from scripts.tech_gamma_schemes import get_scheme

if TYPE_CHECKING:
    from scripts.run_tech_gamma_long_only import TechGammaConfig


ROUND_TRIP_BPS = 3.0


class TradeSide(StrEnum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True, slots=True)
class IntradayPosition:
    ticker: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    score: float
    peak_price: float
    trough_price: float
    atr_stop_price: float | None


class IntradayTrade(TypedDict):
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


def simulate_intraday(frame: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    rows: list[IntradayTrade] = []
    current: IntradayPosition | None = None
    scheme = get_scheme(config.scheme)
    for _, row in frame.sort_values(["ts", "signal_score"], ascending=[True, False]).iterrows():
        if current is not None:
            if str(row["ticker"]) != current.ticker:
                continue
            exit_reason = _exit_reason(row, current, config)
            current = _mark_position(current, row)
            if exit_reason is not None:
                rows.append(_trade(current, row, config.side, exit_reason))
                current = None
            continue
        if not scheme.intraday_entry_ok(row, config):
            continue
        entry_price = float(row["next_open"])
        atr_stop_price = _atr_stop_price(entry_price, row, config)
        current = IntradayPosition(
            ticker=str(row["ticker"]),
            signal_time=pd.Timestamp(row["ts"]),
            entry_time=pd.Timestamp(row["next_ts"]),
            entry_price=entry_price,
            score=float(row["signal_score"]),
            peak_price=entry_price,
            trough_price=entry_price,
            atr_stop_price=atr_stop_price,
        )
    return pd.DataFrame(rows)


def _mark_position(current: IntradayPosition, row: pd.Series) -> IntradayPosition:
    return IntradayPosition(
        ticker=current.ticker,
        signal_time=current.signal_time,
        entry_time=current.entry_time,
        entry_price=current.entry_price,
        score=current.score,
        peak_price=max(current.peak_price, float(row["high"])),
        trough_price=min(current.trough_price, float(row["low"])),
        atr_stop_price=current.atr_stop_price,
    )


def _exit_reason(row: pd.Series, current: IntradayPosition, config: TechGammaConfig) -> str | None:
    match config.side:
        case TradeSide.LONG:
            return _long_exit_reason(row, current, config)
        case TradeSide.SHORT:
            return _short_exit_reason(row, current, config)
        case unreachable:
            assert_never(unreachable)


def _long_exit_reason(row: pd.Series, current: IntradayPosition, config: TechGammaConfig) -> str | None:
    if current.atr_stop_price is not None and float(row["low"]) <= current.atr_stop_price:
        return "atr_stop"
    if float(row["close"]) / current.entry_price - 1.0 <= -config.stop_bps / 10_000.0:
        return "stop_loss"
    if float(row["close"]) / current.peak_price - 1.0 <= -config.trailing_bps / 10_000.0:
        return "trailing_stop"
    if float(row["close"]) < float(row["vwap"]):
        return "vwap_failure"
    if str(row["hhmm"]) >= config.exit_hhmm:
        return "time_exit"
    return None


def _short_exit_reason(row: pd.Series, current: IntradayPosition, config: TechGammaConfig) -> str | None:
    if float(row["close"]) / current.entry_price - 1.0 >= config.stop_bps / 10_000.0:
        return "stop_loss"
    if float(row["close"]) / current.trough_price - 1.0 >= config.trailing_bps / 10_000.0:
        return "trailing_stop"
    if float(row["close"]) > float(row["vwap"]):
        return "vwap_reclaim"
    if str(row["hhmm"]) >= config.exit_hhmm:
        return "time_exit"
    return None


def _trade(current: IntradayPosition, row: pd.Series, side: TradeSide, exit_reason: str) -> IntradayTrade:
    entry_price = current.entry_price
    exit_price = _exit_price(row, current, exit_reason)
    match side:
        case TradeSide.LONG:
            gross = exit_price / entry_price - 1.0
        case TradeSide.SHORT:
            gross = entry_price / exit_price - 1.0
        case unreachable:
            assert_never(unreachable)
    return {
        "ticker": current.ticker,
        "side": side.value,
        "signal_time": current.signal_time,
        "entry_time": current.entry_time,
        "exit_time": pd.Timestamp(row["ts"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "signal_score": current.score,
        "gross_return": gross,
        "net_return": gross - ROUND_TRIP_BPS / 10_000.0,
        "exit_reason": exit_reason,
    }


def _atr_stop_price(entry_price: float, row: pd.Series, config: TechGammaConfig) -> float | None:
    if config.side != TradeSide.LONG or pd.isna(row["atr"]):
        return None
    return entry_price - float(row["atr"]) * config.atr_stop_multiplier


def _exit_price(row: pd.Series, current: IntradayPosition, exit_reason: str) -> float:
    if exit_reason == "atr_stop" and current.atr_stop_price is not None:
        return current.atr_stop_price
    return float(row["close"])
