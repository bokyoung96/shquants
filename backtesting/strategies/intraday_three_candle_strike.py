from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    fast_smma: int = 21
    medium_smma: int = 50
    slow_smma: int = 200
    ema_period: int = 8
    cross_lookback: int = 10
    slope_lookback: int = 60
    slope_threshold: float = 0.00001
    atr_period: int = 14
    atr_buffer_fraction: float = 0.05
    rr: float = 3.0
    round_trip_cost_bps: float = 2.0
    use_filters: bool = True
    use_cross_reversal: bool = True
    trailing: bool = False


@dataclass(frozen=True, slots=True)
class BacktestResult:
    config: BacktestConfig
    trades: pd.DataFrame
    summary: dict[str, float | int | str | None]
    signals: pd.DataFrame


def smma(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be positive")
    return series.astype(float).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def detect_three_candle_strike(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)

    prev3_down = close.shift(3).lt(open_.shift(3)) & close.shift(2).lt(open_.shift(2)) & close.shift(1).lt(open_.shift(1))
    prev3_up = close.shift(3).gt(open_.shift(3)) & close.shift(2).gt(open_.shift(2)) & close.shift(1).gt(open_.shift(1))

    prev_body_low = pd.concat([open_.shift(i).combine(close.shift(i), min) for i in (1, 2, 3)], axis=1).min(axis=1)
    prev_body_high = pd.concat([open_.shift(i).combine(close.shift(i), max) for i in (1, 2, 3)], axis=1).max(axis=1)

    bullish_engulf = close.gt(open_) & open_.le(prev_body_low) & close.ge(prev_body_high)
    bearish_engulf = close.lt(open_) & open_.ge(prev_body_high) & close.le(prev_body_low)

    signal = pd.Series(0, index=df.index, dtype="int8")
    signal.loc[prev3_down & bullish_engulf] = 1
    signal.loc[prev3_up & bearish_engulf] = -1
    return signal


def compute_indicators(
    df: pd.DataFrame,
    *,
    fast_smma: int = 21,
    medium_smma: int = 50,
    slow_smma: int = 200,
    ema_period: int = 8,
    cross_lookback: int = 10,
    slope_lookback: int = 60,
    slope_threshold: float = 0.00001,
    atr_period: int = 14,
) -> pd.DataFrame:
    out = df.sort_values(["trade_date_kst", "hhmm_kst", "ts"]).reset_index(drop=True).copy()
    out["strike_signal"] = detect_three_candle_strike(out)
    out["smma_fast"] = smma(out["close"], fast_smma)
    out["smma_medium"] = smma(out["close"], medium_smma)
    out["smma_slow"] = smma(out["close"], slow_smma)
    out["ema_fast"] = out["close"].astype(float).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()

    out["allow_long"] = out["smma_fast"].gt(out["smma_medium"]) & out["smma_fast"].diff().gt(0.0)
    out["allow_short"] = out["smma_fast"].lt(out["smma_medium"]) & out["smma_fast"].diff().lt(0.0)

    ema_above_smma = out["ema_fast"].gt(out["smma_fast"])
    was_above_smma = ema_above_smma.shift(1, fill_value=False)
    long_cross = ema_above_smma & ~was_above_smma
    short_cross = ~ema_above_smma & was_above_smma
    out["recent_long_cross"] = long_cross.rolling(cross_lookback + 1, min_periods=1).max().astype(bool)
    out["recent_short_cross"] = short_cross.rolling(cross_lookback + 1, min_periods=1).max().astype(bool)

    out["linreg_slope"] = _rolling_normalized_slope(out["close"], slope_lookback)
    out["is_trending"] = out["linreg_slope"].abs().gt(slope_threshold)
    out["daily_atr"] = _previous_daily_atr(out, atr_period)
    return out


def run_backtest(df: pd.DataFrame, config: BacktestConfig = BacktestConfig()) -> BacktestResult:
    signals = compute_indicators(
        df,
        fast_smma=config.fast_smma,
        medium_smma=config.medium_smma,
        slow_smma=config.slow_smma,
        ema_period=config.ema_period,
        cross_lookback=config.cross_lookback,
        slope_lookback=config.slope_lookback,
        slope_threshold=config.slope_threshold,
        atr_period=config.atr_period,
    )
    return run_backtest_from_signals(signals, config)


def run_backtest_from_signals(
    signals: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    signals = signals.copy()
    signals["is_trending"] = signals["linreg_slope"].abs().gt(config.slope_threshold)
    trades: list[dict[str, object]] = []
    i = 0
    n = len(signals)
    while i < n - 1:
        signal = int(signals.loc[i, "strike_signal"])
        if signal == 0 or not _entry_allowed(signals.loc[i], signal, config):
            i += 1
            continue

        entry_idx = i + 1
        if signals.loc[entry_idx, "trade_date_kst"] != signals.loc[i, "trade_date_kst"]:
            i += 1
            continue

        trade = _simulate_trade(signals, signal, signal_idx=i, entry_idx=entry_idx, config=config)
        if trade is None:
            i += 1
            continue
        trades.append(trade)
        i = int(trade["exit_idx"]) + 1

    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades_df, signals, config)
    return BacktestResult(config=config, trades=trades_df, summary=summary, signals=signals)


def summarize_trades(
    trades: pd.DataFrame,
    signals: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, float | int | str | None]:
    if trades.empty:
        return {
            "start": str(signals["trade_date_kst"].min()) if not signals.empty else None,
            "end": str(signals["trade_date_kst"].max()) if not signals.empty else None,
            "bars": int(len(signals)),
            "trades": 0,
            "slope_threshold": config.slope_threshold,
            "trailing": str(config.trailing),
            "total_net_bps": 0.0,
            "avg_net_bps": np.nan,
            "hit_rate": np.nan,
            "avg_r": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_bps": 0.0,
        }

    wins = trades[trades["net_bps"] > 0.0]
    losses = trades[trades["net_bps"] <= 0.0]
    total_win = float(wins["net_bps"].sum())
    total_loss = float(losses["net_bps"].sum())
    cumulative = trades["net_bps"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    return {
        "start": str(signals["trade_date_kst"].min()),
        "end": str(signals["trade_date_kst"].max()),
        "bars": int(len(signals)),
        "trades": int(len(trades)),
        "long_trades": int((trades["side"] == "long").sum()),
        "short_trades": int((trades["side"] == "short").sum()),
        "slope_threshold": float(config.slope_threshold),
        "trailing": str(config.trailing),
        "total_net_bps": float(trades["net_bps"].sum()),
        "avg_net_bps": float(trades["net_bps"].mean()),
        "median_net_bps": float(trades["net_bps"].median()),
        "hit_rate": float((trades["net_bps"] > 0.0).mean()),
        "avg_r": float(trades["r_multiple"].mean()),
        "profit_factor": float(total_win / abs(total_loss)) if total_loss < 0.0 else np.inf,
        "max_drawdown_bps": float(drawdown.min()),
    }


def _entry_allowed(row: pd.Series, signal: int, config: BacktestConfig) -> bool:
    if not config.use_filters:
        return True
    if not bool(row["is_trending"]):
        return False
    if signal > 0:
        return bool(row["allow_long"]) or (config.use_cross_reversal and bool(row["recent_long_cross"]))
    return bool(row["allow_short"]) or (config.use_cross_reversal and bool(row["recent_short_cross"]))


def _simulate_trade(
    df: pd.DataFrame,
    signal: int,
    *,
    signal_idx: int,
    entry_idx: int,
    config: BacktestConfig,
) -> dict[str, object] | None:
    side = "long" if signal > 0 else "short"
    entry = float(df.loc[entry_idx, "open"])
    trade_date = df.loc[entry_idx, "trade_date_kst"]
    lookback = df.iloc[max(0, signal_idx - 3) : signal_idx + 1]
    atr_buffer = _safe_float(df.loc[signal_idx, "daily_atr"], default=0.0) * config.atr_buffer_fraction
    if signal > 0:
        stop = float(lookback["low"].min()) - atr_buffer
        risk = entry - stop
        target = entry + config.rr * risk
    else:
        stop = float(lookback["high"].max()) + atr_buffer
        risk = stop - entry
        target = entry - config.rr * risk
    if not np.isfinite(risk) or risk <= 0.0:
        return None

    initial_stop = stop
    target_touched = False
    exit_idx = entry_idx
    exit_price = entry
    exit_reason = "end_of_day"

    j = entry_idx
    while j < len(df) and df.loc[j, "trade_date_kst"] == trade_date:
        row = df.loc[j]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if signal > 0:
            if low <= stop:
                exit_price = stop
                exit_reason = "trail_stop" if target_touched else "stop"
                exit_idx = j
                break
            if high >= target:
                if not config.trailing:
                    exit_price = target
                    exit_reason = "target"
                    exit_idx = j
                    break
                target_touched = True
                stop = max(stop, entry + risk)
            if target_touched:
                trail = _safe_float(row["smma_fast"], default=np.nan)
                if np.isfinite(trail):
                    stop = max(stop, trail - atr_buffer)
        else:
            if high >= stop:
                exit_price = stop
                exit_reason = "trail_stop" if target_touched else "stop"
                exit_idx = j
                break
            if low <= target:
                if not config.trailing:
                    exit_price = target
                    exit_reason = "target"
                    exit_idx = j
                    break
                target_touched = True
                stop = min(stop, entry - risk)
            if target_touched:
                trail = _safe_float(row["smma_fast"], default=np.nan)
                if np.isfinite(trail):
                    stop = min(stop, trail + atr_buffer)

        exit_idx = j
        exit_price = close
        j += 1

    gross_bps = signal * (exit_price / entry - 1.0) * 10_000.0
    net_bps = gross_bps - config.round_trip_cost_bps
    r_multiple = signal * (exit_price - entry) / risk
    return {
        "signal_idx": int(signal_idx),
        "entry_idx": int(entry_idx),
        "exit_idx": int(exit_idx),
        "side": side,
        "signal_ts": df.loc[signal_idx, "ts"],
        "entry_ts": df.loc[entry_idx, "ts"],
        "exit_ts": df.loc[exit_idx, "ts"],
        "trade_date": trade_date,
        "entry_price": entry,
        "initial_stop": initial_stop,
        "exit_price": exit_price,
        "target_price": target,
        "exit_reason": exit_reason,
        "gross_bps": gross_bps,
        "net_bps": net_bps,
        "r_multiple": r_multiple,
    }


def _previous_daily_atr(df: pd.DataFrame, period: int) -> pd.Series:
    daily = df.groupby("trade_date_kst", sort=True).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean().shift(1)
    return df["trade_date_kst"].map(atr).astype(float)


def _rolling_normalized_slope(series: pd.Series, lookback: int) -> pd.Series:
    if lookback <= 1:
        raise ValueError("slope_lookback must be greater than 1")
    x = np.arange(lookback, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any() or values[-1] == 0.0:
            return np.nan
        y = values.astype(float) - float(np.mean(values))
        return float(np.dot(x, y) / denom / values[-1])

    return series.astype(float).rolling(lookback, min_periods=lookback).apply(slope, raw=True)


def _safe_float(value: object, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


__all__ = (
    "BacktestConfig",
    "BacktestResult",
    "compute_indicators",
    "detect_three_candle_strike",
    "run_backtest",
    "run_backtest_from_signals",
    "smma",
    "summarize_trades",
)
