from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from backtesting.strategies.positivity import positivity_score


EntryMode = Literal["near_high", "breakout"]


@dataclass(frozen=True, slots=True)
class EventQueueConfig:
    max_positions: int = 5
    positivity_lookback: int = 60
    min_periods: int | None = None
    high_lookback: int = 252
    atr_lookback: int = 20
    atr_multiplier: float = 2.5
    relative_signal_groups: int = 3
    entry_high_ratio: float = 0.95
    exit_high_ratio: float = 0.90
    replacement_margin: float = 0.25
    entry_mode: EntryMode = "near_high"
    exit_rank_group_count: int | None = None


@dataclass(frozen=True, slots=True)
class PositivityEventQueueResult:
    weights: pd.DataFrame
    trades: pd.DataFrame
    entry_events: pd.DataFrame
    score: pd.DataFrame
    entry_signal: pd.DataFrame


def true_range_atr(
    *,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    lookback: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    periods = lookback if min_periods is None else int(min_periods)
    if periods <= 0:
        raise ValueError("min_periods must be positive")
    if periods > lookback:
        raise ValueError("min_periods must be less than or equal to lookback")

    prices = close.astype(float)
    highs = high.reindex(index=prices.index, columns=prices.columns).astype(float)
    lows = low.reindex(index=prices.index, columns=prices.columns).astype(float)
    prior_close = prices.shift(1)
    true_range = highs.sub(lows)
    for component in (highs.sub(prior_close).abs(), lows.sub(prior_close).abs()):
        true_range = true_range.where(true_range.ge(component) | component.isna(), component)
    return true_range.rolling(window=lookback, min_periods=periods).mean()


def build_positivity_event_queue_strategy(
    *,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    membership: pd.DataFrame,
    entry_filter: pd.DataFrame | None = None,
    score_bonus: pd.DataFrame | None = None,
    config: EventQueueConfig | None = None,
) -> PositivityEventQueueResult:
    cfg = EventQueueConfig() if config is None else config
    _validate_config(cfg)

    prices = close.astype(float)
    highs = high.reindex(index=prices.index, columns=prices.columns).astype(float)
    lows = low.reindex(index=prices.index, columns=prices.columns).astype(float)
    members = membership.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    external_entry = _aligned_entry_filter(entry_filter=entry_filter, index=prices.index, columns=prices.columns)
    bonus = _aligned_bonus(score_bonus=score_bonus, index=prices.index, columns=prices.columns)

    returns = prices.pct_change(fill_method=None)
    pos = positivity_score(returns, lookback=cfg.positivity_lookback, min_periods=cfg.min_periods).where(members)
    pos_rank = pos.rank(axis=1, pct=True)
    prior_high = prices.shift(1).rolling(window=cfg.high_lookback, min_periods=cfg.high_lookback).max()
    high_ratio = prices.div(prior_high)
    atr = true_range_atr(high=highs, low=lows, close=prices, lookback=cfg.atr_lookback)

    rank_cut = 1.0 - 1.0 / cfg.relative_signal_groups
    price_event = high_ratio.ge(cfg.entry_high_ratio)
    if cfg.entry_mode == "breakout":
        price_event = high_ratio.ge(1.0)
    entry_signal = members & external_entry & pos_rank.ge(rank_cut) & price_event & atr.notna() & prices.notna()
    score = pos_rank.add(high_ratio.rank(axis=1, pct=True), fill_value=0.0).add(bonus, fill_value=0.0).where(members)

    held = pd.Series(False, index=prices.columns)
    held_mode = pd.Series("", index=prices.columns, dtype=object)
    entry_date = pd.Series(pd.NaT, index=prices.columns, dtype="datetime64[ns]")
    entry_price = pd.Series(float("nan"), index=prices.columns, dtype=float)
    entry_stop = pd.Series(float("nan"), index=prices.columns, dtype=float)
    active_score = pd.Series(float("nan"), index=prices.columns, dtype=float)

    rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []
    entry_events: list[dict[str, object]] = []

    for ts in prices.index:
        current_price = prices.loc[ts]
        current_score = score.loc[ts]
        exit_mask, exit_reasons = _exit_mask(
            held=held,
            current_price=current_price,
            entry_stop=entry_stop,
            high_ratio=high_ratio.loc[ts],
            pos_rank=pos_rank.loc[ts],
            config=cfg,
        )
        for symbol in exit_mask.index[exit_mask]:
            _append_trade(
                trades=trades,
                symbol=symbol,
                mode=held_mode.loc[symbol],
                entry_date=entry_date.loc[symbol],
                exit_date=ts,
                entry_price=entry_price.loc[symbol],
                exit_price=current_price.loc[symbol],
                exit_reason=exit_reasons.loc[symbol],
                entry_score=active_score.loc[symbol],
                exit_score=current_score.loc[symbol],
            )
        _clear_positions(
            held=held,
            held_mode=held_mode,
            entry_date=entry_date,
            entry_price=entry_price,
            entry_stop=entry_stop,
            active_score=active_score,
            mask=exit_mask,
        )

        ranked_candidates = current_score.where(entry_signal.loc[ts] & ~held).dropna().sort_values(ascending=False)
        for symbol, candidate_score in ranked_candidates.items():
            if int(held.sum()) < cfg.max_positions:
                _enter_position(
                    symbol=symbol,
                    ts=ts,
                    mode=cfg.entry_mode,
                    current_price=current_price,
                    current_atr=atr.loc[ts],
                    candidate_score=float(candidate_score),
                    high_ratio=high_ratio.loc[ts],
                    pos_rank=pos_rank.loc[ts],
                    held=held,
                    held_mode=held_mode,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    entry_stop=entry_stop,
                    active_score=active_score,
                    entry_events=entry_events,
                    atr_multiplier=cfg.atr_multiplier,
                )
                continue

            weakest_symbol = active_score.where(held).idxmin()
            weakest_score = float(active_score.loc[weakest_symbol])
            if float(candidate_score) <= weakest_score + cfg.replacement_margin:
                continue
            _append_trade(
                trades=trades,
                symbol=weakest_symbol,
                mode=held_mode.loc[weakest_symbol],
                entry_date=entry_date.loc[weakest_symbol],
                exit_date=ts,
                entry_price=entry_price.loc[weakest_symbol],
                exit_price=current_price.loc[weakest_symbol],
                exit_reason="replacement",
                entry_score=active_score.loc[weakest_symbol],
                exit_score=current_score.loc[weakest_symbol],
            )
            replacement_mask = pd.Series(False, index=held.index)
            replacement_mask.loc[weakest_symbol] = True
            _clear_positions(
                held=held,
                held_mode=held_mode,
                entry_date=entry_date,
                entry_price=entry_price,
                entry_stop=entry_stop,
                active_score=active_score,
                mask=replacement_mask,
            )
            _enter_position(
                symbol=symbol,
                ts=ts,
                mode=cfg.entry_mode,
                current_price=current_price,
                current_atr=atr.loc[ts],
                candidate_score=float(candidate_score),
                high_ratio=high_ratio.loc[ts],
                pos_rank=pos_rank.loc[ts],
                held=held,
                held_mode=held_mode,
                entry_date=entry_date,
                entry_price=entry_price,
                entry_stop=entry_stop,
                active_score=active_score,
                entry_events=entry_events,
                atr_multiplier=cfg.atr_multiplier,
            )

        rows.append(held.copy())

    weights = _equal_weight_from_mask(pd.DataFrame(rows, index=prices.index, columns=prices.columns))
    trades_frame = pd.DataFrame(
        trades,
        columns=[
            "symbol",
            "mode",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "return",
            "exit_reason",
            "entry_score",
            "exit_score",
        ],
    )
    events_frame = pd.DataFrame(
        entry_events,
        columns=[
            "date",
            "symbol",
            "mode",
            "entry_price",
            "stop_price",
            "atr",
            "score",
            "positivity_rank",
            "high_ratio",
        ],
    )
    return PositivityEventQueueResult(
        weights=weights,
        trades=trades_frame,
        entry_events=events_frame,
        score=score,
        entry_signal=entry_signal,
    )


def _validate_config(config: EventQueueConfig) -> None:
    if config.max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if config.positivity_lookback <= 0:
        raise ValueError("positivity_lookback must be positive")
    if config.high_lookback <= 0:
        raise ValueError("high_lookback must be positive")
    if config.atr_lookback <= 0:
        raise ValueError("atr_lookback must be positive")
    if config.atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be positive")
    if config.relative_signal_groups <= 0:
        raise ValueError("relative_signal_groups must be positive")
    if config.entry_high_ratio <= 0:
        raise ValueError("entry_high_ratio must be positive")
    if config.exit_high_ratio <= 0:
        raise ValueError("exit_high_ratio must be positive")
    if config.entry_mode not in {"near_high", "breakout"}:
        raise ValueError("entry_mode must be 'near_high' or 'breakout'")
    if config.exit_rank_group_count is not None and config.exit_rank_group_count <= 0:
        raise ValueError("exit_rank_group_count must be positive")


def _aligned_bonus(
    *,
    score_bonus: pd.DataFrame | None,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    if score_bonus is None:
        return pd.DataFrame(0.0, index=index, columns=columns)
    return score_bonus.reindex(index=index, columns=columns).fillna(0.0).astype(float)


def _aligned_entry_filter(
    *,
    entry_filter: pd.DataFrame | None,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    if entry_filter is None:
        return pd.DataFrame(True, index=index, columns=columns)
    return entry_filter.reindex(index=index, columns=columns).fillna(False).astype(bool)


def _exit_mask(
    *,
    held: pd.Series,
    current_price: pd.Series,
    entry_stop: pd.Series,
    high_ratio: pd.Series,
    pos_rank: pd.Series,
    config: EventQueueConfig,
) -> tuple[pd.Series, pd.Series]:
    atr_stop = held & current_price.lt(entry_stop)
    high_failure = held & high_ratio.lt(config.exit_high_ratio)
    rank_failure = pd.Series(False, index=held.index)
    if config.exit_rank_group_count is not None:
        rank_cut = 1.0 - 1.0 / config.exit_rank_group_count
        rank_failure = held & pos_rank.lt(rank_cut)
    exit_mask = atr_stop | high_failure | rank_failure
    reasons = pd.Series("", index=held.index, dtype=object)
    reasons.loc[rank_failure] = "positivity_rank_exit"
    reasons.loc[high_failure] = "near_high_failure"
    reasons.loc[atr_stop] = "atr_stop"
    return exit_mask, reasons


def _append_trade(
    *,
    trades: list[dict[str, object]],
    symbol: object,
    mode: object,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    exit_reason: object,
    entry_score: float,
    exit_score: float,
) -> None:
    trades.append(
        {
            "symbol": str(symbol),
            "mode": mode,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "return": float(exit_price / entry_price - 1.0),
            "exit_reason": exit_reason,
            "entry_score": float(entry_score),
            "exit_score": float(exit_score) if pd.notna(exit_score) else float("nan"),
        }
    )


def _clear_positions(
    *,
    held: pd.Series,
    held_mode: pd.Series,
    entry_date: pd.Series,
    entry_price: pd.Series,
    entry_stop: pd.Series,
    active_score: pd.Series,
    mask: pd.Series,
) -> None:
    held.loc[mask] = False
    held_mode.loc[mask] = ""
    entry_date.loc[mask] = pd.NaT
    entry_price.loc[mask] = float("nan")
    entry_stop.loc[mask] = float("nan")
    active_score.loc[mask] = float("nan")


def _enter_position(
    *,
    symbol: object,
    ts: pd.Timestamp,
    mode: str,
    current_price: pd.Series,
    current_atr: pd.Series,
    candidate_score: float,
    high_ratio: pd.Series,
    pos_rank: pd.Series,
    held: pd.Series,
    held_mode: pd.Series,
    entry_date: pd.Series,
    entry_price: pd.Series,
    entry_stop: pd.Series,
    active_score: pd.Series,
    entry_events: list[dict[str, object]],
    atr_multiplier: float,
) -> None:
    price = float(current_price.loc[symbol])
    atr = float(current_atr.loc[symbol])
    stop_price = price - atr_multiplier * atr
    held.loc[symbol] = True
    held_mode.loc[symbol] = mode
    entry_date.loc[symbol] = ts
    entry_price.loc[symbol] = price
    entry_stop.loc[symbol] = stop_price
    active_score.loc[symbol] = candidate_score
    entry_events.append(
        {
            "date": ts,
            "symbol": str(symbol),
            "mode": mode,
            "entry_price": price,
            "stop_price": stop_price,
            "atr": atr,
            "score": candidate_score,
            "positivity_rank": float(pos_rank.loc[symbol]) if pd.notna(pos_rank.loc[symbol]) else float("nan"),
            "high_ratio": float(high_ratio.loc[symbol]) if pd.notna(high_ratio.loc[symbol]) else float("nan"),
        }
    )


def _equal_weight_from_mask(mask: pd.DataFrame) -> pd.DataFrame:
    clean = mask.fillna(False).astype(bool)
    weights = clean.astype(float)
    counts = clean.sum(axis=1).astype(float)
    return weights.div(counts.where(counts.gt(0.0)), axis=0).fillna(0.0)
