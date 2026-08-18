"""Causal KOSPI200 double-bottom event study and staged-buy backtest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


OHLC = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class DoubleBottomConfig:
    pivot_window: int = 2
    min_pivot_gap: int = 10
    max_pivot_gap: int = 60
    low_tolerance: float = 0.03
    min_rebound: float = 0.10
    holding_period: int = 60
    event_horizons: tuple[int, ...] = (20, 60, 120)
    staged_weights: tuple[float, ...] = (0.50, 0.25, 0.25)
    staged_offsets: tuple[int, ...] = (0, 5, 10)
    cost_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.pivot_window < 1:
            raise ValueError("pivot_window must be positive")
        if self.min_pivot_gap < 1 or self.max_pivot_gap < self.min_pivot_gap:
            raise ValueError("pivot gap bounds are invalid")
        if len(self.staged_weights) != len(self.staged_offsets):
            raise ValueError("staged weights and offsets must have equal lengths")
        if not np.isclose(sum(self.staged_weights), 1.0):
            raise ValueError("staged weights must sum to one")
        if self.staged_offsets[0] != 0 or any(
            a >= b for a, b in zip(self.staged_offsets, self.staged_offsets[1:])
        ):
            raise ValueError("staged offsets must start at zero and be increasing")


def aggregate_daily_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars into Seoul-local daily OHLC bars."""
    required = {"ts", *OHLC}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    result = frame.copy()
    timestamps = pd.to_datetime(result["ts"], utc=True)
    result["date"] = (
        timestamps.dt.tz_convert("Asia/Seoul").dt.normalize().dt.tz_localize(None)
    )
    result = result.sort_values("ts")
    daily = result.groupby("date", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    return daily.dropna(subset=OHLC).astype(float)


def load_daily_ohlc(path: Path) -> pd.DataFrame:
    return aggregate_daily_ohlc(pd.read_parquet(path, columns=["ts", *OHLC]))


def find_pivot_lows(daily: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Return pivot flags and the first date on which each pivot is knowable."""
    if len(daily) < 2 * window + 1:
        return pd.DataFrame(
            index=daily.index, data={"is_pivot_low": False, "confirmed_at": pd.NaT}
        )
    lows = daily["low"].astype(float)
    rolling_min = lows.rolling(2 * window + 1, center=True).min()
    is_pivot = lows.eq(rolling_min).fillna(False)
    confirmed = pd.Series(pd.NaT, index=daily.index, dtype="datetime64[ns]")
    positions = np.flatnonzero(is_pivot.to_numpy())
    valid = positions[(positions + window) < len(daily)]
    confirmed.iloc[valid] = daily.index.to_numpy()[valid + window]
    return pd.DataFrame(
        {"is_pivot_low": is_pivot.astype(bool), "confirmed_at": confirmed},
        index=daily.index,
    )


def detect_double_bottoms(
    daily: pd.DataFrame, config: DoubleBottomConfig
) -> pd.DataFrame:
    """Find confirmed double bottoms and the first causal neckline breakout."""
    daily = daily.sort_index().reset_index(names="date")
    pivots = find_pivot_lows(daily.set_index("date"), config.pivot_window)
    pivot_positions = np.flatnonzero(pivots["is_pivot_low"].to_numpy())
    records: list[dict[str, object]] = []
    for left_pos in pivot_positions:
        for right_pos in pivot_positions:
            gap = right_pos - left_pos
            if gap < config.min_pivot_gap:
                continue
            if gap > config.max_pivot_gap:
                break
            left_low = float(daily.loc[left_pos, "low"])
            right_low = float(daily.loc[right_pos, "low"])
            if abs(right_low / left_low - 1.0) > config.low_tolerance:
                continue
            between = daily.iloc[left_pos + 1 : right_pos]
            neckline = float(between["high"].max())
            if neckline / left_low - 1.0 < config.min_rebound:
                continue
            confirmed_pos = right_pos + config.pivot_window
            breakout_mask = (np.arange(len(daily)) > confirmed_pos) & (
                daily["close"].to_numpy() > neckline
            )
            breakout_positions = np.flatnonzero(breakout_mask)
            if len(breakout_positions) == 0:
                continue
            breakout_pos = int(breakout_positions[0])
            breakout_date = daily.loc[breakout_pos, "date"]
            entry_pos = breakout_pos + 1
            if entry_pos >= len(daily):
                continue
            records.append(
                {
                    "first_pivot_date": daily.loc[left_pos, "date"],
                    "second_pivot_date": daily.loc[right_pos, "date"],
                    "first_pivot_low": left_low,
                    "second_pivot_low": right_low,
                    "neckline": neckline,
                    "rebound_pct": neckline / left_low - 1.0,
                    "breakout_date": breakout_date,
                    "breakout_pos": breakout_pos,
                    "entry_date": daily.loc[entry_pos, "date"],
                    "entry_pos": entry_pos,
                    "entry_price": float(daily.loc[entry_pos, "open"]),
                    "exit_pos": min(
                        entry_pos + config.holding_period - 1, len(daily) - 1
                    ),
                }
            )
            break
    if not records:
        return pd.DataFrame(
            columns=[
                "first_pivot_date",
                "second_pivot_date",
                "breakout_date",
                "entry_date",
                "entry_pos",
                "exit_pos",
            ]
        )
    return (
        pd.DataFrame(records)
        .drop_duplicates(subset=["breakout_date"])
        .sort_values("entry_date")
        .reset_index(drop=True)
    )


def build_fills(
    daily: pd.DataFrame,
    entry_pos: int,
    weights: Sequence[float],
    offsets: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for weight, offset in zip(weights, offsets):
        pos = entry_pos + int(offset)
        if pos >= len(daily):
            continue
        rows.append(
            {
                "fill_pos": pos,
                "fill_date": daily.index[pos],
                "weight": float(weight),
                "price": float(daily.iloc[pos]["open"]),
            }
        )
    return pd.DataFrame(rows, columns=["fill_pos", "fill_date", "weight", "price"])


def select_non_overlapping_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    selected = []
    next_available = -1
    for _, signal in signals.sort_values("entry_pos").iterrows():
        if int(signal["entry_pos"]) > next_available:
            selected.append(signal)
            next_available = int(signal["exit_pos"])
    return pd.DataFrame(selected).reset_index(drop=True)


def _event_return(
    daily: pd.DataFrame,
    entry_pos: int,
    horizon: int,
    weights: Sequence[float],
    offsets: Sequence[int],
    cost: float,
) -> float:
    exit_pos = min(entry_pos + horizon - 1, len(daily) - 1)
    fills = build_fills(daily, entry_pos, weights, offsets)
    if fills.empty:
        return 0.0
    terminal = float(daily.iloc[exit_pos]["close"])
    wealth = sum(
        float(row.weight) * (1.0 - cost) * terminal / float(row.price)
        for row in fills.itertuples()
    )
    return wealth * (1.0 - cost) - 1.0


def run_event_study(
    daily: pd.DataFrame, signals: pd.DataFrame, config: DoubleBottomConfig
) -> pd.DataFrame:
    rows = []
    cost = config.cost_bps / 10_000.0
    schemes = {
        "lump_sum": ((1.0,), (0,)),
        "staged_50_25_25": (config.staged_weights, config.staged_offsets),
    }
    for signal in signals.itertuples():
        for scheme, (weights, offsets) in schemes.items():
            for horizon in config.event_horizons:
                rows.append(
                    {
                        "entry_date": signal.entry_date,
                        "scheme": scheme,
                        "horizon_sessions": horizon,
                        "strategy_return": _event_return(
                            daily, signal.entry_pos, horizon, weights, offsets, cost
                        ),
                        "benchmark_return": float(
                            daily.iloc[
                                min(signal.entry_pos + horizon - 1, len(daily) - 1)
                            ]["close"]
                            / signal.entry_price
                            - 1.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_portfolio(
    daily: pd.DataFrame, signals: pd.DataFrame, config: DoubleBottomConfig, scheme: str
) -> tuple[pd.Series, list[float]]:
    signals = select_non_overlapping_signals(signals)
    if scheme == "lump_sum":
        weights, offsets = (1.0,), (0,)
    elif scheme == "staged_50_25_25":
        weights, offsets = config.staged_weights, config.staged_offsets
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    cost = config.cost_bps / 10_000.0
    cash, units = 1.0, 0.0
    active: dict[str, object] | None = None
    equity_values: list[float] = []
    trade_returns: list[float] = []
    for pos, (date, bar) in enumerate(daily.iterrows()):
        if active is None:
            candidates = signals[signals["entry_pos"] == pos]
            if not candidates.empty:
                active = {
                    "base": cash,
                    "fills": build_fills(daily, pos, weights, offsets),
                    "exit_pos": min(pos + config.holding_period - 1, len(daily) - 1),
                    "start": cash,
                }
        if active is not None:
            base = float(active["base"])
            fills = active["fills"]
            for fill in fills[fills["fill_pos"] == pos].itertuples():
                amount = base * float(fill.weight)
                cash -= amount * (1.0 + cost)
                units += amount / float(fill.price)
            mark = cash + units * float(bar["close"])
            if pos == int(active["exit_pos"]):
                cash += units * float(bar["close"]) * (1.0 - cost)
                units = 0.0
                trade_returns.append(cash / float(active["start"]) - 1.0)
                active = None
                mark = cash
            equity_values.append(mark)
        else:
            equity_values.append(cash)
    return pd.Series(equity_values, index=daily.index, name=scheme), trade_returns


def summarize_equity(
    equity: pd.Series, trade_returns: Iterable[float] = ()
) -> dict[str, float | int]:
    equity = equity.astype(float)
    trade_returns = list(trade_returns)
    returns = equity.pct_change().fillna(0.0)
    drawdown = equity / equity.cummax() - 1.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if returns.std(ddof=1)
        else 0.0,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float(np.mean([r > 0 for r in trade_returns]))
        if trade_returns
        else 0.0,
        "trade_count": len(trade_returns),
    }
