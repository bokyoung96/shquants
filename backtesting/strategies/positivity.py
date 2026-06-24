from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from backtesting.analytics.factor import quantile_returns


@dataclass(frozen=True, slots=True)
class SignalBandStrategyResult:
    weights: pd.DataFrame
    trades: pd.DataFrame


@dataclass(frozen=True, slots=True)
class SectorBreakoutStrategyResult:
    weights: pd.DataFrame
    trades: pd.DataFrame
    sector_state: pd.DataFrame
    market_state: pd.DataFrame
    entry_candidates: pd.DataFrame


def positivity_score(returns: pd.DataFrame, *, lookback: int, min_periods: int | None = None) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    periods = lookback if min_periods is None else int(min_periods)
    if periods <= 0:
        raise ValueError("min_periods must be positive")
    if periods > lookback:
        raise ValueError("min_periods must be less than or equal to lookback")

    clean = returns.astype(float)
    non_negative = clean.ge(0.0).where(clean.notna())
    return non_negative.rolling(window=lookback, min_periods=periods).mean()


def build_sector_neutral_positivity_long_short_weights(
    *,
    score: pd.DataFrame,
    membership: pd.DataFrame,
    sector: pd.DataFrame,
    max_sectors: int = 5,
    pairs_per_sector: int = 1,
) -> pd.DataFrame:
    if max_sectors <= 0:
        raise ValueError("max_sectors must be positive")
    if pairs_per_sector <= 0:
        raise ValueError("pairs_per_sector must be positive")

    scores = score.astype(float)
    members = membership.reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool)
    sectors = sector.reindex(index=scores.index, columns=scores.columns)
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for ts in scores.index:
        day_score = scores.loc[ts].where(members.loc[ts])
        day_sector = sectors.loc[ts]
        sector_candidates: list[tuple[float, object, pd.Index, pd.Index]] = []
        for sector_name in _unique_groups(day_sector):
            sector_members = day_sector.eq(sector_name) & day_score.notna()
            sector_score = day_score.loc[sector_members].sort_values(ascending=False)
            if len(sector_score) < pairs_per_sector * 2:
                continue
            long_symbols = sector_score.head(pairs_per_sector).index
            short_symbols = sector_score.tail(pairs_per_sector).index
            dispersion = float(sector_score.loc[long_symbols].mean() - sector_score.loc[short_symbols].mean())
            sector_candidates.append((dispersion, sector_name, long_symbols, short_symbols))

        selected = sorted(sector_candidates, key=lambda item: item[0], reverse=True)[:max_sectors]
        if not selected:
            continue
        sector_gross = 1.0 / len(selected)
        for _, _, long_symbols, short_symbols in selected:
            weights.loc[ts, long_symbols] = 0.5 * sector_gross / len(long_symbols)
            weights.loc[ts, short_symbols] = -0.5 * sector_gross / len(short_symbols)

    return weights


def build_positivity_new_high_long_only_strategy(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    sector: pd.DataFrame,
    max_positions: int = 6,
    max_positions_per_sector: int | None = 1,
    positivity_lookback: int = 60,
    min_periods: int | None = None,
    breakout_lookback: int = 120,
    stop_lookback: int = 20,
    relative_signal_groups: int = 3,
    breakout_basis: str = "absolute",
) -> SignalBandStrategyResult:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if max_positions_per_sector is not None and max_positions_per_sector <= 0:
        raise ValueError("max_positions_per_sector must be positive")
    if breakout_lookback <= 0:
        raise ValueError("breakout_lookback must be positive")
    if stop_lookback <= 0:
        raise ValueError("stop_lookback must be positive")
    if relative_signal_groups <= 0:
        raise ValueError("relative_signal_groups must be positive")
    if breakout_basis not in {"absolute", "sector_relative"}:
        raise ValueError("breakout_basis must be 'absolute' or 'sector_relative'")

    prices = close.astype(float)
    members = membership.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    sectors = sector.reindex(index=prices.index, columns=prices.columns)
    stock_returns = prices.pct_change(fill_method=None)
    stock_pos = positivity_score(stock_returns, lookback=positivity_lookback, min_periods=min_periods).where(members)
    positivity_leader = _top_group_members(
        values=stock_pos,
        groups=sectors,
        mask=members,
        group_count=relative_signal_groups,
    )
    breakout_source = prices
    if breakout_basis == "sector_relative":
        sector_returns = _equal_average_by_group(values=stock_returns, groups=sectors, mask=members)
        sector_index = sector_returns.fillna(0.0).add(1.0).cumprod()
        breakout_source = prices.div(_map_group_values_to_columns(sector_index, sectors))
    prior_high = breakout_source.shift(1).rolling(window=breakout_lookback, min_periods=breakout_lookback).max()
    stop_line = prices.shift(1).rolling(window=stop_lookback, min_periods=stop_lookback).min()
    breakout = breakout_source.gt(prior_high)
    entry_signal = positivity_leader & breakout & members
    entry_review_dates = _last_dates_by_week(prices.index)

    held = pd.Series(False, index=prices.columns)
    held_mode = pd.Series("", index=prices.columns, dtype=object)
    entry_date = pd.Series(pd.NaT, index=prices.columns, dtype="datetime64[ns]")
    entry_price = pd.Series(float("nan"), index=prices.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []

    for ts in prices.index:
        current_price = prices.loc[ts]
        stop_break = held & current_price.lt(stop_line.loc[ts])
        for symbol in stop_break.index[stop_break]:
            trades.append(
                {
                    "symbol": str(symbol),
                    "mode": held_mode.loc[symbol],
                    "entry_date": entry_date.loc[symbol],
                    "exit_date": ts,
                    "entry_price": float(entry_price.loc[symbol]),
                    "exit_price": float(current_price.loc[symbol]),
                    "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                    "exit_reason": "stop",
                }
            )
            held_mode.loc[symbol] = ""
        held = held & ~stop_break

        if ts in entry_review_dates:
            open_slots = max_positions - int(held.sum())
            if open_slots > 0:
                eligible = (
                    entry_signal.loc[ts]
                    & ~held
                    & stop_line.loc[ts].notna()
                    & current_price.notna()
                )
                entry_score = (
                    stock_pos.loc[ts].rank(pct=True)
                    + breakout_source.loc[ts].div(prior_high.loc[ts]).rank(pct=True)
                ).where(eligible)
                ranked = entry_score.dropna().sort_values(ascending=False)
                selected_symbols: list[object] = []
                sector_counts = sectors.loc[ts, held.index[held]].value_counts(dropna=False).to_dict()
                for symbol in ranked.index:
                    sector_name = sectors.loc[ts, symbol]
                    if max_positions_per_sector is not None:
                        current_count = int(sector_counts.get(sector_name, 0))
                        if current_count >= max_positions_per_sector:
                            continue
                        sector_counts[sector_name] = current_count + 1
                    selected_symbols.append(symbol)
                    if len(selected_symbols) >= open_slots:
                        break
                for symbol in selected_symbols:
                    held.loc[symbol] = True
                    held_mode.loc[symbol] = "new_high_breakout"
                    entry_date.loc[symbol] = ts
                    entry_price.loc[symbol] = float(current_price.loc[symbol])
        rows.append(held.copy())

    weights = _equal_weight_from_mask(pd.DataFrame(rows, index=prices.index, columns=prices.columns))
    trades_frame = pd.DataFrame(
        trades,
        columns=["symbol", "mode", "entry_date", "exit_date", "entry_price", "exit_price", "return", "exit_reason"],
    )
    return SignalBandStrategyResult(weights=weights, trades=trades_frame)


def build_positivity_pullback_reclaim_strategy(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    sector: pd.DataFrame,
    max_positions: int = 5,
    max_positions_per_sector: int | None = 1,
    positivity_lookback: int = 60,
    min_periods: int | None = None,
    high_lookback: int = 252,
    reclaim_lookback: int = 20,
    pullback_low_lookback: int = 20,
    relative_signal_groups: int = 3,
) -> SignalBandStrategyResult:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if max_positions_per_sector is not None and max_positions_per_sector <= 0:
        raise ValueError("max_positions_per_sector must be positive")
    if high_lookback <= 0:
        raise ValueError("high_lookback must be positive")
    if reclaim_lookback <= 0:
        raise ValueError("reclaim_lookback must be positive")
    if pullback_low_lookback <= 0:
        raise ValueError("pullback_low_lookback must be positive")
    if relative_signal_groups <= 0:
        raise ValueError("relative_signal_groups must be positive")

    prices = close.astype(float)
    members = membership.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    sectors = sector.reindex(index=prices.index, columns=prices.columns)
    stock_returns = prices.pct_change(fill_method=None)
    stock_pos = positivity_score(stock_returns, lookback=positivity_lookback, min_periods=min_periods).where(members)
    positivity_leader = _top_group_members(
        values=stock_pos,
        groups=sectors,
        mask=members,
        group_count=relative_signal_groups,
    )
    prior_high = prices.shift(1).rolling(window=high_lookback, min_periods=high_lookback).max()
    reclaim_high = prices.shift(1).rolling(window=reclaim_lookback, min_periods=reclaim_lookback).max()
    pullback_low = prices.shift(1).rolling(window=pullback_low_lookback, min_periods=pullback_low_lookback).min()
    pulled_back = prices.shift(1).lt(prior_high)
    reclaim = prices.gt(reclaim_high) & prices.lt(prior_high)
    entry_signal = positivity_leader & pulled_back & reclaim & members

    held = pd.Series(False, index=prices.columns)
    held_mode = pd.Series("", index=prices.columns, dtype=object)
    entry_date = pd.Series(pd.NaT, index=prices.columns, dtype="datetime64[ns]")
    entry_price = pd.Series(float("nan"), index=prices.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []

    for ts in prices.index:
        current_price = prices.loc[ts]
        stop_break = held & current_price.lt(pullback_low.loc[ts])
        for symbol in stop_break.index[stop_break]:
            trades.append(
                {
                    "symbol": str(symbol),
                    "mode": held_mode.loc[symbol],
                    "entry_date": entry_date.loc[symbol],
                    "exit_date": ts,
                    "entry_price": float(entry_price.loc[symbol]),
                    "exit_price": float(current_price.loc[symbol]),
                    "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                    "exit_reason": "pullback_low",
                }
            )
            held_mode.loc[symbol] = ""
        held = held & ~stop_break

        open_slots = max_positions - int(held.sum())
        if open_slots > 0:
            eligible = (
                entry_signal.loc[ts]
                & ~held
                & pullback_low.loc[ts].notna()
                & current_price.notna()
            )
            entry_score = (
                stock_pos.loc[ts].rank(pct=True)
                + current_price.div(reclaim_high.loc[ts]).rank(pct=True)
            ).where(eligible)
            ranked = entry_score.dropna().sort_values(ascending=False)
            selected_symbols: list[object] = []
            sector_counts = sectors.loc[ts, held.index[held]].value_counts(dropna=False).to_dict()
            for symbol in ranked.index:
                sector_name = sectors.loc[ts, symbol]
                if max_positions_per_sector is not None:
                    current_count = int(sector_counts.get(sector_name, 0))
                    if current_count >= max_positions_per_sector:
                        continue
                    sector_counts[sector_name] = current_count + 1
                selected_symbols.append(symbol)
                if len(selected_symbols) >= open_slots:
                    break
            for symbol in selected_symbols:
                held.loc[symbol] = True
                held_mode.loc[symbol] = "pullback_reclaim"
                entry_date.loc[symbol] = ts
                entry_price.loc[symbol] = float(current_price.loc[symbol])
        rows.append(held.copy())

    weights = _equal_weight_from_mask(pd.DataFrame(rows, index=prices.index, columns=prices.columns))
    trades_frame = pd.DataFrame(
        trades,
        columns=["symbol", "mode", "entry_date", "exit_date", "entry_price", "exit_price", "return", "exit_reason"],
    )
    return SignalBandStrategyResult(weights=weights, trades=trades_frame)


def build_positivity_stable_sleeve_strategy(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    sector: pd.DataFrame,
    max_positions: int = 10,
    max_positions_per_sector: int | None = 2,
    short_lookback: int = 60,
    mid_lookback: int = 120,
    long_lookback: int = 252,
    min_periods: int | None = None,
    entry_group_count: int = 3,
    hold_group_count: int = 2,
) -> SignalBandStrategyResult:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if max_positions_per_sector is not None and max_positions_per_sector <= 0:
        raise ValueError("max_positions_per_sector must be positive")
    if entry_group_count <= 0:
        raise ValueError("entry_group_count must be positive")
    if hold_group_count <= 0:
        raise ValueError("hold_group_count must be positive")

    prices = close.astype(float)
    members = membership.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    sectors = sector.reindex(index=prices.index, columns=prices.columns)
    stock_returns = prices.pct_change(fill_method=None)
    short_pos = positivity_score(stock_returns, lookback=short_lookback, min_periods=min_periods).where(members)
    mid_pos = positivity_score(stock_returns, lookback=mid_lookback, min_periods=min_periods).where(members)
    long_pos = positivity_score(stock_returns, lookback=long_lookback, min_periods=min_periods).where(members)

    short_entry = _top_group_members(values=short_pos, groups=sectors, mask=members, group_count=entry_group_count)
    mid_entry = _top_group_members(values=mid_pos, groups=sectors, mask=members, group_count=entry_group_count)
    long_entry = _top_group_members(values=long_pos, groups=sectors, mask=members, group_count=hold_group_count)
    entry_ok = short_entry & mid_entry & long_entry & members

    short_hold = _top_group_members(values=short_pos, groups=sectors, mask=members, group_count=hold_group_count)
    mid_hold = _top_group_members(values=mid_pos, groups=sectors, mask=members, group_count=hold_group_count)
    hold_ok = short_hold & mid_hold & members

    composite_score = (
        short_pos.rank(axis=1, pct=True)
        + mid_pos.rank(axis=1, pct=True)
        + long_pos.rank(axis=1, pct=True)
    )
    rebalance_dates = _last_dates_by_month(prices.index)

    held = pd.Series(False, index=prices.columns)
    held_mode = pd.Series("", index=prices.columns, dtype=object)
    entry_date = pd.Series(pd.NaT, index=prices.columns, dtype="datetime64[ns]")
    entry_price = pd.Series(float("nan"), index=prices.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []

    for ts in prices.index:
        current_price = prices.loc[ts]
        if ts in rebalance_dates:
            exit_mask = held & ~hold_ok.loc[ts]
            for symbol in exit_mask.index[exit_mask]:
                trades.append(
                    {
                        "symbol": str(symbol),
                        "mode": held_mode.loc[symbol],
                        "entry_date": entry_date.loc[symbol],
                        "exit_date": ts,
                        "entry_price": float(entry_price.loc[symbol]),
                        "exit_price": float(current_price.loc[symbol]),
                        "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                        "exit_reason": "rank_exit",
                    }
                )
                held_mode.loc[symbol] = ""
            held = held & ~exit_mask

            open_slots = max_positions - int(held.sum())
            if open_slots > 0:
                eligible = entry_ok.loc[ts] & ~held & current_price.notna()
                ranked = composite_score.loc[ts].where(eligible).dropna().sort_values(ascending=False)
                selected_symbols: list[object] = []
                sector_counts = sectors.loc[ts, held.index[held]].value_counts(dropna=False).to_dict()
                for symbol in ranked.index:
                    sector_name = sectors.loc[ts, symbol]
                    if max_positions_per_sector is not None:
                        current_count = int(sector_counts.get(sector_name, 0))
                        if current_count >= max_positions_per_sector:
                            continue
                        sector_counts[sector_name] = current_count + 1
                    selected_symbols.append(symbol)
                    if len(selected_symbols) >= open_slots:
                        break
                for symbol in selected_symbols:
                    held.loc[symbol] = True
                    held_mode.loc[symbol] = "stable_sleeve"
                    entry_date.loc[symbol] = ts
                    entry_price.loc[symbol] = float(current_price.loc[symbol])
        rows.append(held.copy())

    weights = _equal_weight_from_mask(pd.DataFrame(rows, index=prices.index, columns=prices.columns))
    trades_frame = pd.DataFrame(
        trades,
        columns=["symbol", "mode", "entry_date", "exit_date", "entry_price", "exit_price", "return", "exit_reason"],
    )
    return SignalBandStrategyResult(weights=weights, trades=trades_frame)


def return_momentum_score(close: pd.DataFrame, *, lookback: int) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    prices = close.astype(float)
    return prices.div(prices.shift(lookback)).sub(1.0)


def build_positivity_quintile_returns(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    lookback: int = 252,
    q: int = 5,
    min_periods: int | None = None,
) -> pd.DataFrame:
    if q <= 0:
        raise ValueError("q must be positive")

    prices = close.astype(float)
    members = membership.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    returns = prices.pct_change(fill_method=None)
    score = positivity_score(returns, lookback=lookback, min_periods=min_periods).where(members)
    next_returns = returns.shift(-1).where(members)
    return quantile_returns(score, next_returns, q=q).dropna(how="all")


def build_positivity_quintile_weights(
    *,
    score: pd.DataFrame,
    membership: pd.DataFrame,
    q: int = 5,
) -> dict[str, pd.DataFrame]:
    if q <= 0:
        raise ValueError("q must be positive")

    scores = score.astype(float)
    members = membership.reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool)
    out = {
        f"q{i}": pd.DataFrame(0.0, index=scores.index, columns=scores.columns, dtype=float)
        for i in range(1, q + 1)
    }

    for ts in scores.index:
        row = scores.loc[ts].where(members.loc[ts])
        valid = row.dropna()
        if valid.empty:
            continue
        if len(valid) == 1 or valid.nunique(dropna=True) == 1:
            out["q1"].loc[ts, valid.index] = 1.0 / len(valid)
            continue
        try:
            bins = pd.qcut(valid, q=min(q, len(valid)), labels=False, duplicates="drop")
        except ValueError:
            bins = pd.Series(0, index=valid.index)
        bins = bins.dropna().astype(int)
        for bucket, names in bins.groupby(bins).groups.items():
            key = f"q{int(bucket) + 1}"
            if key in out and len(names) > 0:
                out[key].loc[ts, list(names)] = 1.0 / len(names)

    return out


def build_positivity_buckets(
    *,
    score: pd.DataFrame,
    membership: pd.DataFrame,
    q: int = 5,
) -> pd.DataFrame:
    if q <= 0:
        raise ValueError("q must be positive")

    scores = score.astype(float)
    members = membership.reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool)
    out = pd.DataFrame(pd.NA, index=scores.index, columns=scores.columns, dtype="Float64")

    for ts in scores.index:
        row = scores.loc[ts].where(members.loc[ts])
        valid = row.dropna()
        if valid.empty:
            continue
        if len(valid) == 1 or valid.nunique(dropna=True) == 1:
            out.loc[ts, valid.index] = 1
            continue
        try:
            bins = pd.qcut(valid, q=min(q, len(valid)), labels=False, duplicates="drop")
        except ValueError:
            bins = pd.Series(0, index=valid.index)
        bins = bins.dropna().astype(int).add(1)
        out.loc[ts, bins.index] = bins

    return out


def flow_positivity_score(flow: pd.DataFrame, *, lookback: int, min_periods: int | None = None) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    periods = lookback if min_periods is None else int(min_periods)
    if periods <= 0:
        raise ValueError("min_periods must be positive")
    if periods > lookback:
        raise ValueError("min_periods must be less than or equal to lookback")

    clean = flow.astype(float)
    positive = clean.gt(0.0).where(clean.notna())
    return positive.rolling(window=lookback, min_periods=periods).mean()


def build_sponsorship_group_weights(
    *,
    q5_mask: pd.DataFrame,
    foreign_flow: pd.DataFrame,
    institution_flow: pd.DataFrame,
    retail_flow: pd.DataFrame,
    lookback: int = 60,
    long_lookback: int = 120,
    threshold: float = 0.6,
) -> dict[str, pd.DataFrame]:
    foreign = foreign_flow.reindex(index=q5_mask.index, columns=q5_mask.columns).astype(float)
    institution = institution_flow.reindex(index=q5_mask.index, columns=q5_mask.columns).astype(float)
    retail = retail_flow.reindex(index=q5_mask.index, columns=q5_mask.columns).astype(float)
    q5 = q5_mask.fillna(False).astype(bool)

    foreign_persistent = (
        flow_positivity_score(foreign, lookback=lookback).ge(threshold)
        & foreign.rolling(window=long_lookback, min_periods=long_lookback).sum().gt(0.0)
    )
    institution_persistent = (
        flow_positivity_score(institution, lookback=lookback).ge(threshold)
        & institution.rolling(window=long_lookback, min_periods=long_lookback).sum().gt(0.0)
    )
    institutional_buy = foreign.rolling(window=lookback, min_periods=lookback).sum().add(
        institution.rolling(window=lookback, min_periods=lookback).sum(),
        fill_value=0.0,
    )
    retail_sell = retail.rolling(window=lookback, min_periods=lookback).sum().lt(0.0)

    masks = {
        "foreign_persistent": q5 & foreign_persistent,
        "institution_persistent": q5 & institution_persistent,
        "dual_sponsorship": q5 & foreign_persistent & institution_persistent,
        "retail_supply_absorption": q5 & institutional_buy.gt(0.0) & retail_sell,
        "no_persistent_sponsorship": q5 & ~(foreign_persistent | institution_persistent),
    }
    return {name: _equal_weight_from_mask(mask) for name, mask in masks.items()}


def build_reacceleration_entry_weights(
    *,
    buckets: pd.DataFrame,
    sponsorship: pd.DataFrame,
    prior_lookback: int = 63,
) -> pd.DataFrame:
    if prior_lookback <= 0:
        raise ValueError("prior_lookback must be positive")

    bucket = buckets.astype("Float64")
    sponsored = sponsorship.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    was_q5_recently = bucket.eq(5).shift(1).rolling(window=prior_lookback, min_periods=1).max().fillna(0.0).astype(bool)
    paused_in_q4 = bucket.shift(1).eq(4)
    reentered_q5 = bucket.eq(5)
    entry = reentered_q5 & paused_in_q4 & was_q5_recently & sponsored
    return _equal_weight_from_mask(entry)


def build_band_holding_weights(
    *,
    buckets: pd.DataFrame,
    no_sponsor: pd.DataFrame,
    retail_supply: pd.DataFrame,
    dual_sponsorship: pd.DataFrame,
    prior_lookback: int = 63,
) -> pd.DataFrame:
    if prior_lookback <= 0:
        raise ValueError("prior_lookback must be positive")

    bucket = buckets.astype("Float64")
    no_sponsor = no_sponsor.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    retail_supply = retail_supply.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    dual_sponsorship = dual_sponsorship.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    was_q5_recently = bucket.eq(5).shift(1).rolling(window=prior_lookback, min_periods=1).max().fillna(0.0).astype(bool)
    entry = bucket.eq(5) & bucket.shift(1).eq(4) & was_q5_recently & ~dual_sponsorship & (no_sponsor | retail_supply)

    held = pd.Series(False, index=bucket.columns)
    rows: list[pd.Series] = []
    for ts in bucket.index:
        in_band = bucket.loc[ts].isin([4, 5]).fillna(False).astype(bool)
        held = held & in_band
        held = held | entry.loc[ts].astype(bool)
        rows.append(held.copy())

    held_frame = pd.DataFrame(rows, index=bucket.index, columns=bucket.columns)
    return _equal_weight_from_mask(held_frame)


def build_signal_band_strategy(
    *,
    buckets: pd.DataFrame,
    close: pd.DataFrame,
    no_sponsor: pd.DataFrame,
    retail_supply: pd.DataFrame,
    dual_sponsorship: pd.DataFrame,
    consensus_ok: pd.DataFrame,
    prior_lookback: int = 63,
    stop_lookback: int = 20,
    breakout_lookback: int = 20,
) -> SignalBandStrategyResult:
    if prior_lookback <= 0:
        raise ValueError("prior_lookback must be positive")
    if stop_lookback <= 0:
        raise ValueError("stop_lookback must be positive")
    if breakout_lookback <= 0:
        raise ValueError("breakout_lookback must be positive")

    bucket = buckets.astype("Float64")
    prices = close.reindex(index=bucket.index, columns=bucket.columns).astype(float)
    no_sponsor = no_sponsor.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    retail_supply = retail_supply.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    dual_sponsorship = dual_sponsorship.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    consensus_ok = consensus_ok.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)

    was_q5_recently = bucket.eq(5).shift(1).rolling(window=prior_lookback, min_periods=1).max().fillna(0.0).astype(bool)
    entry = (
        bucket.eq(5)
        & bucket.shift(1).eq(4)
        & was_q5_recently
        & ~dual_sponsorship
        & (no_sponsor | retail_supply)
        & consensus_ok
    )
    stop_line = prices.shift(1).rolling(window=stop_lookback, min_periods=stop_lookback).min()
    breakout_line = prices.shift(1).rolling(window=breakout_lookback, min_periods=breakout_lookback).max()

    held = pd.Series(False, index=bucket.columns)
    entry_date = pd.Series(pd.NaT, index=bucket.columns, dtype="datetime64[ns]")
    entry_price = pd.Series(float("nan"), index=bucket.columns, dtype=float)
    entry_stop = pd.Series(float("nan"), index=bucket.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []

    for ts in bucket.index:
        current_bucket = bucket.loc[ts]
        current_price = prices.loc[ts]
        band_break = held & ~current_bucket.isin([4, 5]).fillna(False).astype(bool)
        stop_break = held & current_price.lt(entry_stop)
        exit_mask = band_break | stop_break
        for symbol in exit_mask.index[exit_mask]:
            reason = "stop" if bool(stop_break.loc[symbol]) else "band_exit"
            trades.append(
                {
                    "symbol": str(symbol),
                    "entry_date": entry_date.loc[symbol],
                    "exit_date": ts,
                    "entry_price": float(entry_price.loc[symbol]),
                    "exit_price": float(current_price.loc[symbol]),
                    "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                    "exit_reason": reason,
                }
            )
        held = held & ~exit_mask

        breakout = current_price.gt(breakout_line.loc[ts])
        eligible_entry = (
            entry.loc[ts].astype(bool)
            & ~held
            & current_price.notna()
            & stop_line.loc[ts].notna()
            & breakout_line.loc[ts].notna()
            & breakout
        )
        for symbol in eligible_entry.index[eligible_entry]:
            entry_date.loc[symbol] = ts
            entry_price.loc[symbol] = float(current_price.loc[symbol])
            entry_stop.loc[symbol] = float(stop_line.loc[ts, symbol])
        held = held | eligible_entry
        rows.append(held.copy())

    weights = _equal_weight_from_mask(pd.DataFrame(rows, index=bucket.index, columns=bucket.columns))
    trades_frame = pd.DataFrame(
        trades,
        columns=["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return", "exit_reason"],
    )
    return SignalBandStrategyResult(weights=weights, trades=trades_frame)


def build_pure_signal_tilt_strategy(
    *,
    buckets: pd.DataFrame,
    close: pd.DataFrame,
    signal_score: pd.DataFrame,
    no_sponsor: pd.DataFrame,
    retail_supply: pd.DataFrame,
    dual_sponsorship: pd.DataFrame,
    consensus_ok: pd.DataFrame,
    max_positions: int = 15,
    prior_lookback: int = 63,
    stop_lookback: int = 20,
    breakout_lookback: int = 20,
) -> SignalBandStrategyResult:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if prior_lookback <= 0:
        raise ValueError("prior_lookback must be positive")
    if stop_lookback <= 0:
        raise ValueError("stop_lookback must be positive")
    if breakout_lookback <= 0:
        raise ValueError("breakout_lookback must be positive")

    bucket = buckets.astype("Float64")
    prices = close.reindex(index=bucket.index, columns=bucket.columns).astype(float)
    score = signal_score.reindex(index=bucket.index, columns=bucket.columns).astype(float)
    no_sponsor = no_sponsor.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    retail_supply = retail_supply.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    dual_sponsorship = dual_sponsorship.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)
    consensus_ok = consensus_ok.reindex(index=bucket.index, columns=bucket.columns).fillna(False).astype(bool)

    was_q5_recently = bucket.eq(5).shift(1).rolling(window=prior_lookback, min_periods=1).max().fillna(0.0).astype(bool)
    entry = (
        bucket.eq(5)
        & bucket.shift(1).eq(4)
        & was_q5_recently
        & ~dual_sponsorship
        & (no_sponsor | retail_supply)
        & consensus_ok
    )
    stop_line = prices.shift(1).rolling(window=stop_lookback, min_periods=stop_lookback).min()
    breakout_line = prices.shift(1).rolling(window=breakout_lookback, min_periods=breakout_lookback).max()

    held = pd.Series(False, index=bucket.columns)
    entry_date = pd.Series(pd.NaT, index=bucket.columns, dtype="datetime64[ns]")
    entry_price = pd.Series(float("nan"), index=bucket.columns, dtype=float)
    entry_stop = pd.Series(float("nan"), index=bucket.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []

    for ts in bucket.index:
        current_bucket = bucket.loc[ts]
        current_price = prices.loc[ts]
        band_break = held & ~current_bucket.isin([4, 5]).fillna(False).astype(bool)
        stop_break = held & current_price.lt(entry_stop)
        exit_mask = band_break | stop_break
        for symbol in exit_mask.index[exit_mask]:
            reason = "stop" if bool(stop_break.loc[symbol]) else "band_exit"
            trades.append(
                {
                    "symbol": str(symbol),
                    "entry_date": entry_date.loc[symbol],
                    "exit_date": ts,
                    "entry_price": float(entry_price.loc[symbol]),
                    "exit_price": float(current_price.loc[symbol]),
                    "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                    "exit_reason": reason,
                }
            )
        held = held & ~exit_mask

        open_slots = max_positions - int(held.sum())
        selected_entry = pd.Series(False, index=bucket.columns)
        if open_slots > 0:
            breakout = current_price.gt(breakout_line.loc[ts])
            eligible_entry = (
                entry.loc[ts].astype(bool)
                & ~held
                & current_price.notna()
                & stop_line.loc[ts].notna()
                & breakout_line.loc[ts].notna()
                & breakout
            )
            ranked = score.loc[ts].where(eligible_entry).dropna().sort_values(ascending=False)
            selected_entry.loc[ranked.head(open_slots).index] = True
            for symbol in selected_entry.index[selected_entry]:
                entry_date.loc[symbol] = ts
                entry_price.loc[symbol] = float(current_price.loc[symbol])
                entry_stop.loc[symbol] = float(stop_line.loc[ts, symbol])
        held = held | selected_entry
        rows.append(held.copy())

    weights = _equal_weight_from_mask(pd.DataFrame(rows, index=bucket.index, columns=bucket.columns))
    trades_frame = pd.DataFrame(
        trades,
        columns=["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return", "exit_reason"],
    )
    return SignalBandStrategyResult(weights=weights, trades=trades_frame)


def build_sector_positivity_state(
    *,
    score: pd.DataFrame,
    membership: pd.DataFrame,
    benchmark_weight: pd.DataFrame,
    sector: pd.DataFrame,
    slope_lookback: int = 20,
) -> pd.DataFrame:
    if slope_lookback <= 0:
        raise ValueError("slope_lookback must be positive")

    scores = score.astype(float)
    members = membership.reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool)
    sectors = sector.reindex(index=scores.index, columns=scores.columns)
    weights = benchmark_weight.shift(1).reindex(index=scores.index, columns=scores.columns).astype(float).where(members)

    weighted = _weighted_average_by_group(values=scores, groups=sectors, weights=weights, mask=members)
    equal = _equal_average_by_group(values=scores, groups=sectors, mask=members)
    weighted_slope = weighted.sub(weighted.shift(slope_lookback))
    equal_slope = equal.sub(equal.shift(slope_lookback))

    records: list[dict[str, object]] = []
    for ts in scores.index:
        sectors_for_date = sorted(set(weighted.columns.dropna()).union(equal.columns.dropna()))
        for sector_name in sectors_for_date:
            records.append(
                {
                    "date": ts,
                    "sector": sector_name,
                    "sector_weighted_pos": weighted.loc[ts, sector_name] if sector_name in weighted.columns else float("nan"),
                    "sector_equal_pos": equal.loc[ts, sector_name] if sector_name in equal.columns else float("nan"),
                    "sector_weighted_pos_slope": weighted_slope.loc[ts, sector_name]
                    if sector_name in weighted_slope.columns
                    else float("nan"),
                    "sector_equal_pos_slope": equal_slope.loc[ts, sector_name]
                    if sector_name in equal_slope.columns
                    else float("nan"),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "sector_weighted_pos",
                "sector_equal_pos",
                "sector_weighted_pos_slope",
                "sector_equal_pos_slope",
            ]
        ).rename_axis(index=["date", "sector"])
    return pd.DataFrame.from_records(records).set_index(["date", "sector"]).sort_index()


def build_sector_positivity_breakout_strategy(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    benchmark_weight: pd.DataFrame,
    sector: pd.DataFrame,
    foreign_flow: pd.DataFrame,
    institution_flow: pd.DataFrame,
    retail_flow: pd.DataFrame,
    consensus_ok: pd.DataFrame,
    max_positions: int = 10,
    positivity_lookback: int = 60,
    min_periods: int | None = None,
    sector_slope_lookback: int = 20,
    breakout_lookback: int = 60,
    stop_lookback: int = 60,
    flow_lookback: int = 60,
    flow_long_lookback: int = 120,
    flow_threshold: float = 0.6,
    residual_threshold: float = 0.55,
    leadership_gap: float = 0.08,
) -> SectorBreakoutStrategyResult:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if positivity_lookback <= 0:
        raise ValueError("positivity_lookback must be positive")
    if sector_slope_lookback <= 0:
        raise ValueError("sector_slope_lookback must be positive")
    if breakout_lookback <= 0:
        raise ValueError("breakout_lookback must be positive")
    if stop_lookback <= 0:
        raise ValueError("stop_lookback must be positive")

    prices = close.astype(float)
    members = membership.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    sectors = sector.reindex(index=prices.index, columns=prices.columns)
    bm_weights = benchmark_weight.shift(1).reindex(index=prices.index, columns=prices.columns).astype(float).where(members)
    stock_returns = prices.pct_change(fill_method=None)
    stock_pos = positivity_score(stock_returns, lookback=positivity_lookback, min_periods=min_periods).where(members)
    sector_returns = _weighted_average_by_group(
        values=stock_returns,
        groups=sectors,
        weights=bm_weights,
        mask=members,
    )
    mapped_sector_returns = _map_group_values_to_columns(sector_returns, sectors)
    residual_returns = stock_returns.sub(mapped_sector_returns)
    residual_pos = positivity_score(residual_returns, lookback=positivity_lookback, min_periods=min_periods).where(members)

    sector_state = build_sector_positivity_state(
        score=stock_pos,
        membership=members,
        benchmark_weight=benchmark_weight,
        sector=sectors,
        slope_lookback=sector_slope_lookback,
    )
    sector_weighted = sector_state["sector_weighted_pos"].unstack("sector")
    sector_equal = sector_state["sector_equal_pos"].unstack("sector")
    sector_weighted_slope = sector_state["sector_weighted_pos_slope"].unstack("sector")
    sector_equal_slope = sector_state["sector_equal_pos_slope"].unstack("sector")
    sector_weighted_mapped = _map_group_values_to_columns(sector_weighted, sectors)
    sector_equal_mapped = _map_group_values_to_columns(sector_equal, sectors)
    sector_weighted_slope_mapped = _map_group_values_to_columns(sector_weighted_slope, sectors)
    sector_equal_slope_mapped = _map_group_values_to_columns(sector_equal_slope, sectors)
    market_state = _build_market_positivity_state(score=stock_pos, membership=members, benchmark_weight=benchmark_weight)

    flow_masks = _build_flow_confirmation_masks(
        foreign_flow=foreign_flow.reindex(index=prices.index, columns=prices.columns),
        institution_flow=institution_flow.reindex(index=prices.index, columns=prices.columns),
        retail_flow=retail_flow.reindex(index=prices.index, columns=prices.columns),
        lookback=flow_lookback,
        long_lookback=flow_long_lookback,
        threshold=flow_threshold,
    )
    flow_ok = (~flow_masks["dual_sponsorship"]) & (
        flow_masks["no_persistent_sponsorship"] | flow_masks["retail_supply_absorption"]
    )
    consensus = consensus_ok.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    prior_high = prices.shift(1).rolling(window=breakout_lookback, min_periods=breakout_lookback).max()
    stop_line = prices.shift(1).rolling(window=stop_lookback, min_periods=stop_lookback).min()
    review_dates = _last_dates_by_week(prices.index)

    sector_expansion = (
        sector_weighted_slope_mapped.gt(0.0)
        & sector_equal_slope_mapped.gt(0.0)
        & residual_pos.ge(residual_threshold)
    )
    leadership = (
        stock_pos.sub(sector_weighted_mapped).ge(leadership_gap)
        & residual_pos.ge(residual_threshold)
        & sector_equal_mapped.le(sector_weighted_mapped)
    )
    breakout = prices.gt(prior_high)
    base_entry = (sector_expansion | leadership) & breakout & flow_ok & consensus & members

    held = pd.Series(False, index=prices.columns)
    entry_date = pd.Series(pd.NaT, index=prices.columns, dtype="datetime64[ns]")
    entry_price = pd.Series(float("nan"), index=prices.columns, dtype=float)
    entry_stop = pd.Series(float("nan"), index=prices.columns, dtype=float)
    rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []

    for ts in prices.index:
        current_price = prices.loc[ts]
        current_residual = residual_pos.loc[ts]
        stop_break = held & current_price.lt(entry_stop)
        residual_break = held & current_residual.lt(0.50)
        exit_mask = stop_break | residual_break
        for symbol in exit_mask.index[exit_mask]:
            reason = "stop" if bool(stop_break.loc[symbol]) else "residual_break"
            trades.append(
                {
                    "symbol": str(symbol),
                    "mode": "",
                    "entry_date": entry_date.loc[symbol],
                    "exit_date": ts,
                    "entry_price": float(entry_price.loc[symbol]),
                    "exit_price": float(current_price.loc[symbol]),
                    "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                    "exit_reason": reason,
                }
            )
        held = held & ~exit_mask

        if ts in review_dates:
            open_slots = max_positions - int(held.sum())
            if open_slots > 0:
                eligible = base_entry.loc[ts] & ~held & stop_line.loc[ts].notna() & prices.loc[ts].notna()
                entry_score = (
                    stock_pos.loc[ts].fillna(0.0)
                    + residual_pos.loc[ts].fillna(0.0)
                    + sector_weighted_slope_mapped.loc[ts].clip(lower=0.0).fillna(0.0)
                    + flow_masks["retail_supply_absorption"].loc[ts].astype(float).mul(0.05)
                ).where(eligible)
                selected = entry_score.dropna().sort_values(ascending=False).head(open_slots)
                for symbol in selected.index:
                    mode = "sector_expansion" if bool(sector_expansion.loc[ts, symbol]) else "leadership_breakout"
                    entry_date.loc[symbol] = ts
                    entry_price.loc[symbol] = float(current_price.loc[symbol])
                    entry_stop.loc[symbol] = float(stop_line.loc[ts, symbol])
                    held.loc[symbol] = True
                    candidates.append(
                        {
                            "date": ts,
                            "symbol": str(symbol),
                            "sector": sectors.loc[ts, symbol],
                            "mode": mode,
                            "entry_score": float(selected.loc[symbol]),
                            "stock_pos": float(stock_pos.loc[ts, symbol]),
                            "residual_pos": float(residual_pos.loc[ts, symbol]),
                            "sector_weighted_pos": float(sector_weighted_mapped.loc[ts, symbol]),
                            "sector_weighted_pos_slope": float(sector_weighted_slope_mapped.loc[ts, symbol]),
                            "entry_price": float(current_price.loc[symbol]),
                            "stop_line": float(stop_line.loc[ts, symbol]),
                        }
                    )
        rows.append(held.copy())

    weights = _equal_weight_from_mask(pd.DataFrame(rows, index=prices.index, columns=prices.columns))
    trades_frame = pd.DataFrame(
        trades,
        columns=["symbol", "mode", "entry_date", "exit_date", "entry_price", "exit_price", "return", "exit_reason"],
    )
    candidates_frame = pd.DataFrame(
        candidates,
        columns=[
            "date",
            "symbol",
            "sector",
            "mode",
            "entry_score",
            "stock_pos",
            "residual_pos",
            "sector_weighted_pos",
            "sector_weighted_pos_slope",
            "entry_price",
            "stop_line",
        ],
    )
    return SectorBreakoutStrategyResult(
        weights=weights,
        trades=trades_frame,
        sector_state=sector_state,
        market_state=market_state,
        entry_candidates=candidates_frame,
    )


def build_sector_positivity_event_core_strategy(
    *,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    benchmark_weight: pd.DataFrame,
    sector: pd.DataFrame,
    foreign_flow: pd.DataFrame,
    institution_flow: pd.DataFrame,
    retail_flow: pd.DataFrame,
    consensus_ok: pd.DataFrame,
    max_positions: int = 15,
    positivity_lookback: int = 60,
    min_periods: int | None = None,
    sector_slope_lookback: int = 20,
    breakout_lookback: int = 60,
    stop_lookback: int = 60,
    flow_lookback: int = 60,
    flow_long_lookback: int = 120,
    flow_threshold: float = 0.6,
    residual_threshold: float | None = None,
    leadership_gap: float | None = None,
    min_holding_days: int = 40,
    trail_stop: bool = False,
    market_entry_floor: float | None = None,
    leadership_market_floor: float | None = None,
    require_market_positive_slope: bool = True,
    market_median_lookback: int | None = 756,
    market_median_min_periods: int | None = None,
    relative_signal_groups: int = 3,
    sector_rank_groups: int = 2,
    max_positions_per_sector: int | None = 1,
) -> SectorBreakoutStrategyResult:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if min_holding_days < 0:
        raise ValueError("min_holding_days must be non-negative")
    if relative_signal_groups <= 0:
        raise ValueError("relative_signal_groups must be positive")
    if sector_rank_groups <= 0:
        raise ValueError("sector_rank_groups must be positive")
    if max_positions_per_sector is not None and max_positions_per_sector <= 0:
        raise ValueError("max_positions_per_sector must be positive")

    prices = close.astype(float)
    members = membership.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    sectors = sector.reindex(index=prices.index, columns=prices.columns)
    bm_weights = benchmark_weight.shift(1).reindex(index=prices.index, columns=prices.columns).astype(float).where(members)
    stock_returns = prices.pct_change(fill_method=None)
    stock_pos = positivity_score(stock_returns, lookback=positivity_lookback, min_periods=min_periods).where(members)
    sector_returns = _weighted_average_by_group(values=stock_returns, groups=sectors, weights=bm_weights, mask=members)
    mapped_sector_returns = _map_group_values_to_columns(sector_returns, sectors)
    residual_pos = positivity_score(
        stock_returns.sub(mapped_sector_returns),
        lookback=positivity_lookback,
        min_periods=min_periods,
    ).where(members)

    sector_state = build_sector_positivity_state(
        score=stock_pos,
        membership=members,
        benchmark_weight=benchmark_weight,
        sector=sectors,
        slope_lookback=sector_slope_lookback,
    )
    sector_weighted = sector_state["sector_weighted_pos"].unstack("sector")
    sector_equal = sector_state["sector_equal_pos"].unstack("sector")
    sector_weighted_slope = sector_state["sector_weighted_pos_slope"].unstack("sector")
    sector_equal_slope = sector_state["sector_equal_pos_slope"].unstack("sector")
    sector_weighted_mapped = _map_group_values_to_columns(sector_weighted, sectors)
    sector_equal_mapped = _map_group_values_to_columns(sector_equal, sectors)
    sector_weighted_slope_mapped = _map_group_values_to_columns(sector_weighted_slope, sectors)
    sector_equal_slope_mapped = _map_group_values_to_columns(sector_equal_slope, sectors)
    market_state = _build_market_positivity_state(score=stock_pos, membership=members, benchmark_weight=benchmark_weight)
    market_weighted_pos = market_state["market_weighted_pos"].reindex(prices.index)
    market_weighted_slope = market_weighted_pos.sub(market_weighted_pos.shift(sector_slope_lookback))
    market_entry_ok = pd.Series(True, index=prices.index)
    if market_median_lookback is not None:
        if market_median_lookback <= 0:
            raise ValueError("market_median_lookback must be positive")
        median_periods = (
            max(1, market_median_lookback // 3)
            if market_median_min_periods is None
            else int(market_median_min_periods)
        )
        if median_periods <= 0:
            raise ValueError("market_median_min_periods must be positive")
        market_median = (
            market_weighted_pos.shift(1)
            .rolling(window=market_median_lookback, min_periods=median_periods)
            .median()
        )
        market_entry_ok &= market_weighted_pos.gt(market_median)
    if market_entry_floor is not None:
        market_entry_ok &= market_weighted_pos.ge(market_entry_floor)
    if require_market_positive_slope:
        market_entry_ok &= market_weighted_slope.ge(0.0)
    leadership_market_ok = market_entry_ok.copy()
    if leadership_market_floor is not None:
        leadership_market_ok &= market_weighted_pos.ge(leadership_market_floor)

    flow_masks = _build_flow_confirmation_masks(
        foreign_flow=foreign_flow.reindex(index=prices.index, columns=prices.columns),
        institution_flow=institution_flow.reindex(index=prices.index, columns=prices.columns),
        retail_flow=retail_flow.reindex(index=prices.index, columns=prices.columns),
        lookback=flow_lookback,
        long_lookback=flow_long_lookback,
        threshold=flow_threshold,
    )
    flow_ok = (~flow_masks["dual_sponsorship"]) & (
        flow_masks["no_persistent_sponsorship"] | flow_masks["retail_supply_absorption"]
    )
    consensus = consensus_ok.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
    prior_high = prices.shift(1).rolling(window=breakout_lookback, min_periods=breakout_lookback).max()
    stop_line = prices.shift(1).rolling(window=stop_lookback, min_periods=stop_lookback).min()
    entry_review_dates = _last_dates_by_week(prices.index)
    exit_review_dates = _last_dates_by_month(prices.index)

    residual_leader = _top_group_members(
        values=residual_pos,
        groups=sectors,
        mask=members,
        group_count=relative_signal_groups,
    )
    residual_ok = residual_leader
    if residual_threshold is not None:
        residual_ok &= residual_pos.ge(residual_threshold)
    leadership_gap_score = stock_pos.sub(sector_weighted_mapped)
    leadership_leader = _top_group_members(
        values=leadership_gap_score,
        groups=sectors,
        mask=members,
        group_count=relative_signal_groups,
    )
    leadership_ok = leadership_leader
    if leadership_gap is not None:
        leadership_ok &= leadership_gap_score.ge(leadership_gap)
    sector_rank_ok = _top_row_members(values=sector_weighted, group_count=sector_rank_groups)
    sector_rank_ok_mapped = _map_group_values_to_columns(sector_rank_ok.astype(float), sectors).fillna(0.0).astype(bool)

    sector_expansion = (
        sector_weighted_slope_mapped.gt(0.0)
        & sector_equal_slope_mapped.ge(0.0)
        & sector_rank_ok_mapped
        & residual_ok
    )
    leadership = (
        leadership_ok
        & residual_ok
        & sector_weighted_slope_mapped.gt(0.0)
    )
    breakout_event = prices.gt(prior_high)
    reclaim_event = prices.gt(prices.shift(5)) & prices.shift(1).lt(prior_high) & prices.ge(prior_high.mul(0.98))
    market_entry_ok_mapped = pd.DataFrame(
        {column: market_entry_ok for column in prices.columns},
        index=prices.index,
    )
    leadership_market_ok_mapped = pd.DataFrame(
        {column: leadership_market_ok for column in prices.columns},
        index=prices.index,
    )
    base_entry = (
        ((sector_expansion & market_entry_ok_mapped) | (leadership & leadership_market_ok_mapped))
        & (breakout_event | reclaim_event)
        & flow_ok
        & consensus
        & members
    )

    held = pd.Series(False, index=prices.columns)
    held_mode = pd.Series("", index=prices.columns, dtype=object)
    entry_date = pd.Series(pd.NaT, index=prices.columns, dtype="datetime64[ns]")
    entry_price = pd.Series(float("nan"), index=prices.columns, dtype=float)
    entry_stop = pd.Series(float("nan"), index=prices.columns, dtype=float)
    cooldown_until = pd.Series(pd.Timestamp.min, index=prices.columns)
    rows: list[pd.Series] = []
    trades: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []

    for ts in prices.index:
        current_price = prices.loc[ts]
        holding_days = (ts - entry_date).dt.days.fillna(0)
        active_stop = entry_stop
        if trail_stop:
            active_stop = pd.concat([entry_stop, stop_line.loc[ts]], axis=1).max(axis=1)
        stop_break = held & current_price.lt(active_stop)
        monthly_soft_break = (
            held
            & (ts in exit_review_dates)
            & holding_days.ge(max(min_holding_days, 80))
            & (
                (
                    sector_weighted_slope_mapped.loc[ts].lt(0.0)
                    & sector_equal_slope_mapped.loc[ts].lt(0.0)
                    & stock_pos.loc[ts].lt(sector_weighted_mapped.loc[ts].sub(0.03))
                )
                | ~consensus.loc[ts]
            )
        )
        exit_mask = stop_break | monthly_soft_break
        for symbol in exit_mask.index[exit_mask]:
            reason = "stop" if bool(stop_break.loc[symbol]) else "monthly_soft_exit"
            trades.append(
                {
                    "symbol": str(symbol),
                    "mode": held_mode.loc[symbol],
                    "entry_date": entry_date.loc[symbol],
                    "exit_date": ts,
                    "entry_price": float(entry_price.loc[symbol]),
                    "exit_price": float(current_price.loc[symbol]),
                    "return": float(current_price.loc[symbol] / entry_price.loc[symbol] - 1.0),
                    "exit_reason": reason,
                }
            )
            cooldown_until.loc[symbol] = ts + pd.Timedelta(days=min_holding_days)
            held_mode.loc[symbol] = ""
        held = held & ~exit_mask
        if trail_stop:
            entry_stop = entry_stop.where(~held, active_stop)

        if ts in entry_review_dates:
            open_slots = max_positions - int(held.sum())
            if open_slots > 0:
                eligible = (
                    base_entry.loc[ts]
                    & ~held
                    & stop_line.loc[ts].notna()
                    & prices.loc[ts].notna()
                    & cooldown_until.lt(ts)
                )
                entry_score = (
                    stock_pos.loc[ts].rank(pct=True)
                    + residual_pos.loc[ts].rank(pct=True)
                    + sector_weighted_slope_mapped.loc[ts].rank(pct=True)
                    + flow_masks["retail_supply_absorption"].loc[ts].astype(float)
                ).where(eligible)
                ranked = entry_score.dropna().sort_values(ascending=False)
                selected_symbols: list[object] = []
                sector_counts = sectors.loc[ts, held.index[held]].value_counts(dropna=False).to_dict()
                for symbol in ranked.index:
                    sector_name = sectors.loc[ts, symbol]
                    if max_positions_per_sector is not None:
                        current_count = int(sector_counts.get(sector_name, 0))
                        if current_count >= max_positions_per_sector:
                            continue
                        sector_counts[sector_name] = current_count + 1
                    selected_symbols.append(symbol)
                    if len(selected_symbols) >= open_slots:
                        break
                selected = ranked.loc[selected_symbols]
                for symbol in selected.index:
                    mode = "sector_expansion" if bool(sector_expansion.loc[ts, symbol]) else "leadership_breakout"
                    held.loc[symbol] = True
                    held_mode.loc[symbol] = mode
                    entry_date.loc[symbol] = ts
                    entry_price.loc[symbol] = float(current_price.loc[symbol])
                    entry_stop.loc[symbol] = float(stop_line.loc[ts, symbol])
                    candidates.append(
                        {
                            "date": ts,
                            "symbol": str(symbol),
                            "sector": sectors.loc[ts, symbol],
                            "mode": mode,
                            "entry_score": float(selected.loc[symbol]),
                            "stock_pos": float(stock_pos.loc[ts, symbol]),
                            "residual_pos": float(residual_pos.loc[ts, symbol]),
                            "sector_weighted_pos": float(sector_weighted_mapped.loc[ts, symbol]),
                            "sector_weighted_pos_slope": float(sector_weighted_slope_mapped.loc[ts, symbol]),
                            "entry_price": float(current_price.loc[symbol]),
                            "stop_line": float(stop_line.loc[ts, symbol]),
                        }
                    )
        rows.append(held.copy())

    weights = _equal_weight_from_mask(pd.DataFrame(rows, index=prices.index, columns=prices.columns))
    trades_frame = pd.DataFrame(
        trades,
        columns=["symbol", "mode", "entry_date", "exit_date", "entry_price", "exit_price", "return", "exit_reason"],
    )
    candidates_frame = pd.DataFrame(
        candidates,
        columns=[
            "date",
            "symbol",
            "sector",
            "mode",
            "entry_score",
            "stock_pos",
            "residual_pos",
            "sector_weighted_pos",
            "sector_weighted_pos_slope",
            "entry_price",
            "stop_line",
        ],
    )
    return SectorBreakoutStrategyResult(
        weights=weights,
        trades=trades_frame,
        sector_state=sector_state,
        market_state=market_state,
        entry_candidates=candidates_frame,
    )


def _top_group_members(
    *,
    values: pd.DataFrame,
    groups: pd.DataFrame,
    mask: pd.DataFrame,
    group_count: int,
) -> pd.DataFrame:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    out = pd.DataFrame(False, index=values.index, columns=values.columns)
    aligned_groups = groups.reindex(index=values.index, columns=values.columns)
    aligned_mask = mask.reindex(index=values.index, columns=values.columns).fillna(False).astype(bool)
    for ts in values.index:
        day_values = values.loc[ts].where(aligned_mask.loc[ts])
        day_groups = aligned_groups.loc[ts]
        for group in _unique_groups(day_groups):
            group_members = day_groups.eq(group) & day_values.notna()
            group_values = day_values.loc[group_members]
            if group_values.empty:
                continue
            top_n = max(1, math.ceil(len(group_values) / group_count))
            out.loc[ts, group_values.nlargest(top_n).index] = True
    return out


def _top_row_members(*, values: pd.DataFrame, group_count: int) -> pd.DataFrame:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    out = pd.DataFrame(False, index=values.index, columns=values.columns)
    for ts in values.index:
        row = values.loc[ts].dropna()
        if row.empty:
            continue
        top_n = max(1, math.ceil(len(row) / group_count))
        out.loc[ts, row.nlargest(top_n).index] = True
    return out


def _build_market_positivity_state(
    *,
    score: pd.DataFrame,
    membership: pd.DataFrame,
    benchmark_weight: pd.DataFrame,
) -> pd.DataFrame:
    scores = score.astype(float)
    members = membership.reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool)
    weights = benchmark_weight.shift(1).reindex(index=scores.index, columns=scores.columns).astype(float).where(members)
    weighted_sum = scores.mul(weights).sum(axis=1, min_count=1)
    weight_sum = weights.where(scores.notna()).sum(axis=1)
    weight_sum = weight_sum.where(weight_sum.ne(0.0))
    return pd.DataFrame(
        {
            "market_weighted_pos": weighted_sum.div(weight_sum),
            "market_equal_pos": scores.where(members).mean(axis=1),
        },
        index=scores.index,
    )


def _build_flow_confirmation_masks(
    *,
    foreign_flow: pd.DataFrame,
    institution_flow: pd.DataFrame,
    retail_flow: pd.DataFrame,
    lookback: int,
    long_lookback: int,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    foreign = foreign_flow.astype(float)
    institution = institution_flow.astype(float)
    retail = retail_flow.astype(float)
    foreign_persistent = (
        flow_positivity_score(foreign, lookback=lookback, min_periods=lookback).ge(threshold)
        & foreign.rolling(window=long_lookback, min_periods=long_lookback).sum().gt(0.0)
    )
    institution_persistent = (
        flow_positivity_score(institution, lookback=lookback, min_periods=lookback).ge(threshold)
        & institution.rolling(window=long_lookback, min_periods=long_lookback).sum().gt(0.0)
    )
    institutional_buy = foreign.rolling(window=lookback, min_periods=lookback).sum().add(
        institution.rolling(window=lookback, min_periods=lookback).sum(),
        fill_value=0.0,
    )
    retail_sell = retail.rolling(window=lookback, min_periods=lookback).sum().lt(0.0)
    no_persistent = ~(foreign_persistent | institution_persistent)
    return {
        "foreign_persistent": foreign_persistent.fillna(False),
        "institution_persistent": institution_persistent.fillna(False),
        "dual_sponsorship": (foreign_persistent & institution_persistent).fillna(False),
        "retail_supply_absorption": (institutional_buy.gt(0.0) & retail_sell).fillna(False),
        "no_persistent_sponsorship": no_persistent.fillna(True),
    }


def _weighted_average_by_group(
    *,
    values: pd.DataFrame,
    groups: pd.DataFrame,
    weights: pd.DataFrame,
    mask: pd.DataFrame,
) -> pd.DataFrame:
    all_groups = _unique_groups(groups)
    out = pd.DataFrame(index=values.index, columns=all_groups, dtype=float)
    for group in all_groups:
        group_mask = groups.eq(group) & mask
        weighted_values = values.where(group_mask).mul(weights.where(group_mask))
        numerator = weighted_values.sum(axis=1, min_count=1)
        denominator = weights.where(group_mask & values.notna()).sum(axis=1)
        denominator = denominator.where(denominator.ne(0.0))
        out[group] = numerator.div(denominator)
    return out


def _equal_average_by_group(*, values: pd.DataFrame, groups: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    all_groups = _unique_groups(groups)
    out = pd.DataFrame(index=values.index, columns=all_groups, dtype=float)
    for group in all_groups:
        out[group] = values.where(groups.eq(group) & mask).mean(axis=1)
    return out


def _map_group_values_to_columns(group_values: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=groups.index, columns=groups.columns, dtype=float)
    for group in group_values.columns:
        out = out.where(~groups.eq(group), group_values[group].reindex(groups.index), axis=0)
    return out


def _unique_groups(groups: pd.DataFrame) -> list[object]:
    values = pd.unique(pd.Series(groups.to_numpy().ravel()).dropna())
    return sorted(values)


def _last_dates_by_week(index: pd.Index) -> set[pd.Timestamp]:
    dates = pd.DatetimeIndex(index)
    frame = pd.DataFrame({"date": dates}, index=dates)
    return set(frame.groupby(dates.to_period("W-FRI"))["date"].max())


def _last_dates_by_month(index: pd.Index) -> set[pd.Timestamp]:
    dates = pd.DatetimeIndex(index)
    frame = pd.DataFrame({"date": dates}, index=dates)
    return set(frame.groupby(dates.to_period("M"))["date"].max())


def _equal_weight_from_mask(mask: pd.DataFrame) -> pd.DataFrame:
    clean = mask.fillna(False).astype(bool)
    weights = clean.astype(float)
    counts = clean.sum(axis=1).astype(float)
    return weights.div(counts.where(counts.gt(0.0)), axis=0).fillna(0.0)
