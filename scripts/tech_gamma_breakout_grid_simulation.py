from __future__ import annotations

import pandas as pd

from scripts.run_tech_gamma_long_only import TechGammaConfig, summarize
from scripts.tech_gamma_costs import net_return_after_costs


def rank_grid_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["trade_viable"] = ranked["trades"].ge(30).astype(int)
    drawdown = ranked["max_drawdown"].abs().where(ranked["max_drawdown"].abs().gt(0.0), pd.NA)
    ranked["robust_avg_bps"] = ranked[["avg_net_bps", "early_avg_net_bps", "late_avg_net_bps"]].min(axis=1)
    ranked["robust_score"] = ranked["robust_avg_bps"].divide(drawdown).fillna(0.0)
    ranked = ranked.sort_values(
        ["trade_viable", "robust_score", "robust_avg_bps", "hit_rate", "trades"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "selection_rank", range(1, len(ranked) + 1))
    return ranked


def simulate_grid_continuation(base_candidates: pd.DataFrame, daily: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    candidates = entry_candidates(base_candidates, config)
    rows: list[dict[str, int | float | str | pd.Timestamp]] = []
    daily_groups = {str(ticker): group.reset_index(drop=True) for ticker, group in daily.groupby("ticker", sort=True)}
    for ticker, ticker_candidates in candidates.groupby("ticker", sort=True):
        available_date = pd.Timestamp.min
        ticker_daily = daily_groups[str(ticker)]
        for _, signal in ticker_candidates.iterrows():
            signal_date = pd.Timestamp(signal["date"])
            if signal_date <= available_date:
                continue
            trade = continuation_trade(signal, ticker_daily, config)
            if trade is None:
                continue
            rows.append(trade)
            available_date = pd.Timestamp(trade["exit_time"]).normalize()
    return pd.DataFrame(rows)


def base_entry_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    factor_ok = frame["factor_filter_ok"] if "factor_filter_ok" in frame.columns else True
    positivity_ok = frame["positivity_filter_ok"] if "positivity_filter_ok" in frame.columns else True
    previous_close = frame["previous_intraday_close"]
    mask = (
        frame["next_open"].notna()
        & frame["signal_score"].notna()
        & frame["prior_52w_close_high"].notna()
        & frame["atr"].notna()
        & frame["close"].gt(frame["prior_52w_close_high"])
        & (previous_close.isna() | previous_close.le(frame["prior_52w_close_high"]))
        & positivity_ok
        & factor_ok
    )
    columns = ["ticker", "date", "ts", "hhmm", "next_ts", "next_open", "atr", "signal_score", "breakout_52w_bps"]
    return frame.loc[mask, columns].sort_values(["ticker", "date", "ts"])


def entry_candidates(base_candidates: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    mask = (
        base_candidates["hhmm"].gt(config.range_end_hhmm)
        & base_candidates["hhmm"].lt(config.exit_hhmm)
        & base_candidates["breakout_52w_bps"].ge(config.range_buffer_bps)
    )
    columns = ["ticker", "date", "ts", "next_ts", "next_open", "atr", "signal_score"]
    return base_candidates.loc[mask, columns].groupby(["ticker", "date"], sort=True).head(1)


def daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.sort_values("ts").groupby(["ticker", "date"], sort=True).agg(
        ts=("ts", "last"),
        close=("close", "last"),
        daily_low=("low", "min"),
        prior_52w_close_high=("prior_52w_close_high", "last"),
    )
    return daily.reset_index()


def continuation_trade(
    signal: pd.Series,
    daily: pd.DataFrame,
    config: TechGammaConfig,
) -> dict[str, int | float | str | pd.Timestamp] | None:
    entry_date = pd.Timestamp(signal["date"])
    entry_price = float(signal["next_open"])
    stop_price = entry_price - float(signal["atr"]) * config.atr_stop_multiplier
    holding_days = (pd.to_datetime(daily["date"]) - entry_date).dt.days
    exits = daily.loc[
        holding_days.ge(config.min_holding_days)
        & (daily["daily_low"].le(stop_price) | daily["close"].le(daily["prior_52w_close_high"]))
    ]
    if exits.empty:
        if int(holding_days.iloc[-1]) < config.min_holding_days:
            return None
        exit_row = daily.iloc[-1]
        exit_reason = "end_of_data"
    else:
        exit_row = exits.iloc[0]
        exit_reason = "atr_stop" if float(exit_row["daily_low"]) <= stop_price else "new_high_lost"
    exit_price = stop_price if exit_reason == "atr_stop" else float(exit_row["close"])
    gross = exit_price / entry_price - 1.0
    return {
        "ticker": str(signal["ticker"]),
        "side": "long",
        "signal_time": pd.Timestamp(signal["ts"]),
        "entry_time": pd.Timestamp(signal["next_ts"]),
        "exit_time": pd.Timestamp(exit_row["ts"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "signal_score": float(signal["signal_score"]),
        "gross_return": gross,
        "net_return": net_return_after_costs(gross),
        "exit_reason": exit_reason,
    }


def strategy_summary(
    strategy: str,
    config: TechGammaConfig,
    trades: pd.DataFrame,
) -> dict[str, int | float | str]:
    row = summarize(trades, pd.DataFrame()).loc[0].to_dict()
    returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    return {
        "strategy": strategy,
        "trades": int(row["trades"]),
        "net_return_sum": float(row["net_return_sum"]),
        "avg_net_bps": float(row["avg_net_bps"]),
        "hit_rate": float(row["hit_rate"]),
        "early_avg_net_bps": period_avg_bps(returns, "early"),
        "late_avg_net_bps": period_avg_bps(returns, "late"),
        "max_drawdown": trade_equity_mdd(returns),
        **config_columns(config),
    }


def period_avg_bps(returns: pd.Series, part: str) -> float:
    if returns.empty:
        return 0.0
    midpoint = len(returns) // 2
    window = returns.iloc[:midpoint] if part == "early" else returns.iloc[midpoint:]
    return float(window.mean() * 10_000.0) if not window.empty else 0.0


def trade_equity_mdd(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float(equity.div(equity.cummax()).sub(1.0).min())


def config_columns(config: TechGammaConfig) -> dict[str, int | float | str]:
    return {
        "positivity_lookback_days": config.positivity_lookback_days,
        "positivity_benchmark": config.positivity_benchmark,
        "positivity_margin": config.positivity_margin,
        "factor_filter": config.factor_filter,
        "factor_lookback_days": config.factor_lookback_days,
        "holding_mode": config.holding_mode,
        "min_holding_days": config.min_holding_days,
        "atr_stop_multiplier": config.atr_stop_multiplier,
        "range_buffer_bps": config.range_buffer_bps,
        "range_end_hhmm": config.range_end_hhmm,
        "exit_hhmm": config.exit_hhmm,
        "stop_bps": config.stop_bps,
        "trailing_bps": config.trailing_bps,
    }
