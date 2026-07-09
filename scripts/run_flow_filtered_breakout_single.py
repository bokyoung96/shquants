from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtesting.data.kr_stock_5m import KrStock5mDataset, read_tickers_bars
from backtesting.strategies.positivity import positivity_score
from root import ROOT
from scripts.run_tech_gamma_long_only import TechGammaConfig, build_features, summarize
from scripts.tech_gamma_costs import net_return_after_costs
from scripts.tech_gamma_holding import simulate_continuation_holding
from scripts.tech_gamma_intraday import simulate_intraday
from scripts.tech_gamma_plots import write_performance_outputs
from scripts.tech_gamma_research_filters import (
    ResearchFeatureData,
    _add_factor_filter,
    _aligned,
    _positivity_benchmark,
    _positivity_filter,
    _stack,
    load_research_feature_data,
)
from scripts.tech_gamma_universe import filter_kospi200_historical_members, kospi200_tickers


DEFAULT_CONFIG = ROOT.results_path / "flow_filtered_breakout_single" / "sector_pos90_margin002_flow_or_60d" / "config.json"
DEFAULT_OUTPUT = ROOT.results_path / "flow_filtered_breakout_single" / "sector_pos90_margin002_flow_or_60d_2019start"


def base_output_dir(output_dir: Path) -> Path:
    return output_dir / "base"


def run_batched_single_strategy(
    config: TechGammaConfig,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    batch_size: int = 25,
    dataset: KrStock5mDataset | object | None = None,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved_dataset = dataset or KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    parquet_root = resolved_dataset.root.parent if isinstance(resolved_dataset, KrStock5mDataset) else ROOT.parquet_path
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = base_output_dir(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    tickers = kospi200_tickers(parquet_root, config)
    data = load_research_feature_data(parquet_root, tickers)
    load_start = _load_start(config)
    daily_features = build_daily_research_features(
        dataset=resolved_dataset,
        tickers=tickers,
        config=config,
        data=data,
        start=load_start,
        end=config.end,
        parquet_root=parquet_root,
    )

    trades: list[pd.DataFrame] = []
    for index, batch in enumerate(_ticker_batches(tickers, batch_size), start=1):
        print(f"batch={index} tickers={len(batch)}", flush=True)
        frame = _load_batch_frame(
            dataset=resolved_dataset,
            tickers=batch,
            config=config,
            start=load_start,
            end=config.end,
            parquet_root=parquet_root,
            daily_features=daily_features,
        )
        if frame.empty:
            continue
        if config.holding_mode == "continuation":
            batch_trades = simulate_continuation_holding(frame, config)
        else:
            batch_trades = simulate_intraday(frame, config)
        if not batch_trades.empty:
            trades.append(batch_trades)

    intraday = _combine_trades(trades)
    overnight = _empty_trades()
    intraday.to_csv(base_dir / "intraday_trades.csv", index=False)
    overnight.to_csv(base_dir / "overnight_trades.csv", index=False)
    summarize(intraday, overnight).to_csv(base_dir / "summary.csv", index=False)
    pd.Series(tickers, name="ticker").to_csv(base_dir / "universe_tickers.csv", index=False)
    write_performance_outputs(intraday, overnight, base_dir, "Flow Filtered Breakout Performance 2019 Start")
    (base_dir / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output_dir


def run_fast_prefiltered_single_strategy(
    config: TechGammaConfig,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    batch_size: int = 20,
    dataset: KrStock5mDataset | None = None,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved_dataset = dataset or KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    parquet_root = resolved_dataset.root.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = base_output_dir(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = base_dir / "partial_trades"
    partial_dir.mkdir(exist_ok=True)

    tickers = kospi200_tickers(parquet_root, config)
    load_start = _load_start(config)
    close, high, low = load_daily_5m_matrices(resolved_dataset, tickers, start=load_start, end=config.end)
    data = load_research_feature_data(parquet_root, tickers)
    daily_features = _daily_research_features_from_close(close=close, config=config, data=data, tickers=tickers)
    candidates = prefilter_breakout_candidates(close=close, high=high, low=low, daily_features=daily_features, config=config)
    candidates.to_csv(base_dir / "prefilter_candidates.csv", index=False)
    print(f"prefilter_candidates={len(candidates)}", flush=True)

    daily_exit = _daily_exit_frame(close=close, low=low)
    monthly_trades: list[pd.DataFrame] = []
    if not candidates.empty:
        for month, month_candidates in candidates.groupby(candidates["date"].dt.to_period("M"), sort=True):
            partial_path = partial_dir / f"{month}.csv"
            if partial_path.exists():
                monthly_trades.append(pd.read_csv(partial_path, parse_dates=["signal_time", "entry_time", "exit_time"]))
                print(f"month={month} cached", flush=True)
                continue
            month_trades = _run_candidate_month(
                dataset=resolved_dataset,
                config=config,
                month=str(month),
                candidates=month_candidates,
                daily_exit=daily_exit,
                batch_size=batch_size,
            )
            month_trades.to_csv(partial_path, index=False)
            monthly_trades.append(month_trades)
            print(f"month={month} trades={len(month_trades)}", flush=True)

    intraday = remove_overlapping_trades(_combine_trades(monthly_trades))
    if config.episode_compression:
        intraday = compress_breakout_episodes(intraday, daily_exit)
    overnight = _empty_trades()
    intraday.to_csv(base_dir / "intraday_trades.csv", index=False)
    overnight.to_csv(base_dir / "overnight_trades.csv", index=False)
    summarize(intraday, overnight).to_csv(base_dir / "summary.csv", index=False)
    pd.Series(tickers, name="ticker").to_csv(base_dir / "universe_tickers.csv", index=False)
    write_performance_outputs(intraday, overnight, base_dir, "Flow Filtered Breakout Performance 2019 Start", mark_to_market=daily_exit)
    (base_dir / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output_dir


def load_daily_price_matrices(
    parquet_root: Path,
    tickers: tuple[str, ...],
    *,
    start: pd.Timestamp,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close = _read_daily_matrix(parquet_root / "qw_adj_c.parquet", tickers, start=start, end=end)
    high = _read_daily_matrix(parquet_root / "qw_adj_h.parquet", tickers, start=start, end=end)
    low = _read_daily_matrix(parquet_root / "qw_adj_l.parquet", tickers, start=start, end=end)
    return (
        _apply_daily_membership(close, parquet_root),
        _apply_daily_membership(high, parquet_root),
        _apply_daily_membership(low, parquet_root),
    )


def load_daily_5m_matrices(
    dataset: KrStock5mDataset,
    tickers: tuple[str, ...],
    *,
    start: pd.Timestamp,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close_parts: list[pd.DataFrame] = []
    max_close_parts: list[pd.DataFrame] = []
    low_parts: list[pd.DataFrame] = []
    for month in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        close_path = dataset.field_path(str(month), "c")
        low_path = dataset.field_path(str(month), "l")
        if not close_path.exists() or not low_path.exists():
            continue
        available = set(pq.read_schema(close_path).names)
        selected = [ticker for ticker in tickers if ticker in available]
        if not selected:
            continue
        close_5m = pd.read_parquet(close_path, columns=selected, engine="pyarrow")
        low_5m = pd.read_parquet(low_path, columns=selected, engine="pyarrow")
        close_5m.index = pd.to_datetime(close_5m.index).normalize()
        low_5m.index = pd.to_datetime(low_5m.index).normalize()
        close_parts.append(close_5m.groupby(level=0).last())
        max_close_parts.append(close_5m.groupby(level=0).max())
        low_parts.append(low_5m.groupby(level=0).min())
    if not close_parts:
        empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers)
        return empty, empty, empty
    close = pd.concat(close_parts).sort_index().loc[pd.Timestamp(start).normalize() : pd.Timestamp(end).normalize()].reindex(columns=tickers)
    max_close = pd.concat(max_close_parts).sort_index().loc[close.index.min() : close.index.max()].reindex(columns=tickers)
    low = pd.concat(low_parts).sort_index().loc[close.index.min() : close.index.max()].reindex(columns=tickers)
    parquet_root = dataset.root.parent
    return (
        _apply_daily_membership(close, parquet_root),
        _apply_daily_membership(max_close, parquet_root),
        _apply_daily_membership(low, parquet_root),
    )


def prefilter_breakout_candidates(
    *,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    daily_features: pd.DataFrame,
    config: TechGammaConfig,
) -> pd.DataFrame:
    prior = close.shift(1).rolling(252, min_periods=1).max()
    threshold = prior * (1.0 + config.range_buffer_bps / 10_000.0)
    mask = high.gt(threshold) & high.notna() & prior.notna()
    mask = mask.loc[mask.index >= pd.Timestamp(config.start)]
    stacked = pd.DataFrame(
        {
            "daily_close": _stack(close.reindex(mask.index).where(mask)),
            "daily_low": _stack(low.reindex(mask.index).where(mask)),
            "prior_52w_close_high": _stack(prior.reindex(mask.index).where(mask)),
        }
    ).dropna(subset=["prior_52w_close_high"]).reset_index(names=["date", "ticker"])
    if stacked.empty:
        return stacked
    filters = daily_features[
        [
            column
            for column in (
                "date",
                "ticker",
                "daily_positivity",
                "positivity_benchmark",
                "positivity_spread",
                "positivity_filter_ok",
                "factor_filter_ok",
                "foreign_flow_to_cap",
                "institution_flow_to_cap",
                "sector_name",
            )
            if column in daily_features.columns
        ]
    ]
    merged = stacked.merge(filters, on=["date", "ticker"], how="left", sort=False)
    if "positivity_filter_ok" not in merged.columns:
        merged["positivity_filter_ok"] = True
    if "factor_filter_ok" not in merged.columns:
        merged["factor_filter_ok"] = True
    positivity_ok = merged["positivity_filter_ok"].fillna(False) if config.use_positivity else pd.Series(True, index=merged.index)
    factor_ok = merged["factor_filter_ok"].fillna(False) if config.factor_filter != "none" else pd.Series(True, index=merged.index)
    return merged.loc[positivity_ok & factor_ok].reset_index(drop=True)


def remove_overlapping_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    kept: list[pd.Series] = []
    ordered = trades.copy()
    ordered["signal_time"] = pd.to_datetime(ordered["signal_time"])
    ordered["exit_time"] = pd.to_datetime(ordered["exit_time"])
    for _ticker, ticker_trades in ordered.sort_values(["ticker", "signal_time"]).groupby("ticker", sort=True):
        available_date = pd.Timestamp.min
        for _, trade in ticker_trades.iterrows():
            signal_date = pd.Timestamp(trade["signal_time"]).normalize()
            if signal_date <= available_date:
                continue
            kept.append(trade)
            available_date = pd.Timestamp(trade["exit_time"]).normalize()
    if not kept:
        return trades.iloc[0:0].copy()
    return pd.DataFrame(kept).sort_values(["entry_time", "ticker"]).reset_index(drop=True)


def compress_breakout_episodes(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    kept: list[pd.Series] = []
    ordered = trades.copy()
    ordered["signal_time"] = pd.to_datetime(ordered["signal_time"])
    daily_by_ticker = {
        str(ticker): group.assign(date=pd.to_datetime(group["date"]).dt.normalize()).sort_values("date").reset_index(drop=True)
        for ticker, group in daily.groupby("ticker", sort=True)
    }
    for ticker, ticker_trades in ordered.sort_values(["ticker", "signal_time"]).groupby("ticker", sort=True):
        blocked_until = pd.Timestamp.min
        ticker_daily = daily_by_ticker.get(str(ticker))
        for _, trade in ticker_trades.iterrows():
            signal_date = pd.Timestamp(trade["signal_time"]).normalize()
            if signal_date <= blocked_until:
                continue
            kept.append(trade)
            if ticker_daily is None:
                blocked_until = pd.Timestamp.max
                continue
            resets = ticker_daily.loc[
                ticker_daily["date"].ge(signal_date)
                & ticker_daily["close"].le(ticker_daily["prior_52w_close_high"])
            ]
            blocked_until = pd.Timestamp.max if resets.empty else pd.Timestamp(resets.iloc[0]["date"])
    if not kept:
        return trades.iloc[0:0].copy()
    return pd.DataFrame(kept).sort_values(["entry_time", "ticker"]).reset_index(drop=True)


def build_daily_research_features(
    *,
    dataset: KrStock5mDataset | object,
    tickers: tuple[str, ...],
    config: TechGammaConfig,
    data: ResearchFeatureData,
    start: pd.Timestamp,
    end: str,
    parquet_root: Path,
) -> pd.DataFrame:
    close = _daily_close_from_5m(dataset, tickers, start=start, end=end)
    close = _apply_daily_membership(close, parquet_root)
    dates = pd.DatetimeIndex(close.index)
    positivity = positivity_score(
        close.pct_change(fill_method=None),
        lookback=config.positivity_lookback_days,
        min_periods=config.positivity_lookback_days,
    ).shift(1)
    sector = _aligned(data.sector, dates, tickers)
    cap = _aligned(data.market_cap, dates, tickers).ffill().shift(1)
    benchmark = _positivity_benchmark(positivity, sector, cap, config.positivity_benchmark)
    features = pd.DataFrame(
        {
            "daily_positivity": _stack(positivity),
            "positivity_benchmark": _stack(benchmark),
        }
    ).reset_index(names=["date", "ticker"])
    features["positivity_spread"] = features["daily_positivity"] - features["positivity_benchmark"]
    features["positivity_filter_ok"] = _positivity_filter(features, config)
    return _add_factor_filter(features, config, data, dates, tickers, sector, cap)


def config_from_json(path: Path, *, start: str, end: str | None = None) -> TechGammaConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    values["start"] = start
    if end is not None:
        values["end"] = end
    allowed = {field.name for field in fields(TechGammaConfig)}
    return TechGammaConfig(**{key: value for key, value in values.items() if key in allowed})


def _run_candidate_month(
    *,
    dataset: KrStock5mDataset,
    config: TechGammaConfig,
    month: str,
    candidates: pd.DataFrame,
    daily_exit: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    month_start = pd.Period(month, freq="M").to_timestamp()
    read_start = max(_load_start(config), month_start - pd.Timedelta(days=10))
    read_end = pd.Period(month, freq="M").end_time
    month_trades: list[pd.DataFrame] = []
    tickers = tuple(sorted(candidates["ticker"].drop_duplicates()))
    for batch in _ticker_batches(tickers, batch_size):
        raw = read_tickers_bars(dataset, batch, start=read_start, end=read_end)
        if raw.empty:
            continue
        usable = raw.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
        frame = build_features(usable, config)
        frame = frame.loc[frame["date"].isin(candidates["date"].unique())]
        if frame.empty:
            continue
        candidate_columns = [
            column
            for column in (
                "date",
                "ticker",
                "daily_close",
                "prior_52w_close_high",
                "daily_positivity",
                "positivity_benchmark",
                "positivity_spread",
                "positivity_filter_ok",
                "factor_filter_ok",
                "foreign_flow_to_cap",
                "institution_flow_to_cap",
                "sector_name",
            )
            if column in candidates.columns
        ]
        frame = frame.drop(columns=[column for column in candidate_columns if column in frame.columns and column not in ("date", "ticker")])
        frame = frame.merge(candidates[candidate_columns], on=["date", "ticker"], how="inner", sort=False)
        frame["breakout_52w_bps"] = (frame["close"] / frame["prior_52w_close_high"] - 1.0) * 10_000.0
        frame["high_52w_breakout_score"] = frame["breakout_52w_bps"].clip(lower=0.0).divide(10.0) + frame["volume_spike"].clip(upper=5.0).sub(1.0).clip(lower=0.0)
        frame["signal_score"] = frame["high_52w_breakout_score"]
        entries = _entry_candidates(frame, config)
        if entries.empty:
            continue
        batch_trades = _simulate_daily_continuation(entries, daily_exit, config)
        if not batch_trades.empty:
            month_trades.append(batch_trades)
    return _combine_trades(month_trades)


def _load_batch_frame(
    *,
    dataset: KrStock5mDataset | object,
    tickers: tuple[str, ...],
    config: TechGammaConfig,
    start: pd.Timestamp,
    end: str,
    parquet_root: Path,
    daily_features: pd.DataFrame,
) -> pd.DataFrame:
    raw = read_tickers_bars(dataset, tickers, start=start, end=end)
    if raw.empty:
        return raw
    usable = raw.dropna(subset=[column for column in ("open", "high", "low", "close", "volume") if column in raw.columns]).copy()
    frame = build_features(usable, config)
    if config.universe == "kospi200_historical":
        frame = filter_kospi200_historical_members(frame, parquet_root)
    clean = frame.drop(columns=[column for column in daily_features.columns if column in frame.columns and column not in ("date", "ticker")])
    enriched = clean.merge(daily_features, on=["ticker", "date"], how="left", sort=False)
    return enriched.loc[enriched["ts"].ge(pd.Timestamp(config.start))].reset_index(drop=True)


def _daily_close_from_5m(
    dataset: KrStock5mDataset | object,
    tickers: tuple[str, ...],
    *,
    start: pd.Timestamp,
    end: str,
) -> pd.DataFrame:
    if not isinstance(dataset, KrStock5mDataset):
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers)
    frames: list[pd.DataFrame] = []
    for month in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        path = dataset.field_path(str(month), "c")
        if not path.exists():
            continue
        available = set(pq.read_schema(path).names)
        selected = [ticker for ticker in tickers if ticker in available]
        if not selected:
            continue
        monthly = pd.read_parquet(path, columns=selected, engine="pyarrow")
        monthly.index = pd.to_datetime(monthly.index).normalize()
        frames.append(monthly.groupby(level=0).last())
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers)
    close = pd.concat(frames).sort_index()
    close = close.loc[pd.Timestamp(start).normalize() : pd.Timestamp(end).normalize()]
    return close.reindex(columns=tickers)


def _daily_research_features_from_close(
    *,
    close: pd.DataFrame,
    config: TechGammaConfig,
    data: ResearchFeatureData,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(close.index)
    positivity = positivity_score(
        close.pct_change(fill_method=None),
        lookback=config.positivity_lookback_days,
        min_periods=config.positivity_lookback_days,
    ).shift(1)
    sector = _aligned(data.sector, dates, tickers)
    cap = _aligned(data.market_cap, dates, tickers).ffill().shift(1)
    benchmark = _positivity_benchmark(positivity, sector, cap, config.positivity_benchmark)
    features = pd.DataFrame(
        {
            "daily_positivity": _stack(positivity),
            "positivity_benchmark": _stack(benchmark),
        }
    ).reset_index(names=["date", "ticker"])
    features["positivity_spread"] = features["daily_positivity"] - features["positivity_benchmark"]
    features["positivity_filter_ok"] = _positivity_filter(features, config)
    return _add_factor_filter(features, config, data, dates, tickers, sector, cap)


def _read_daily_matrix(path: Path, tickers: tuple[str, ...], *, start: pd.Timestamp, end: str) -> pd.DataFrame:
    available = set(pq.read_schema(path).names)
    selected = [ticker for ticker in tickers if ticker in available]
    frame = pd.read_parquet(path, columns=selected, engine="pyarrow")
    frame.index = pd.to_datetime(frame.index)
    return frame.loc[pd.Timestamp(start).normalize() : pd.Timestamp(end).normalize()].reindex(columns=tickers)


def _apply_daily_membership(close: pd.DataFrame, parquet_root: Path) -> pd.DataFrame:
    membership_path = parquet_root / "qw_k200_yn.parquet"
    if close.empty or not membership_path.exists():
        return close
    membership = pd.read_parquet(membership_path, engine="pyarrow")
    active = membership.reindex(index=membership.index.union(close.index), columns=close.columns).ffill().reindex(close.index).fillna(0).gt(0)
    return close.where(active)


def _daily_exit_frame(*, close: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    prior = close.shift(1).rolling(252, min_periods=1).max()
    frame = pd.DataFrame(
        {
            "close": _stack(close),
            "daily_low": _stack(low),
            "prior_52w_close_high": _stack(prior),
        }
    ).reset_index(names=["date", "ticker"])
    return frame.dropna(subset=["close", "daily_low", "prior_52w_close_high"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def _entry_candidates(frame: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    working = frame.copy()
    if config.entry_confirmation == "next_close_confirmed":
        grouped = working.groupby(["ticker", "date"], sort=False)
        working["confirmation_close"] = grouped["close"].shift(-1)
        working["confirmed_next_ts"] = grouped["next_ts"].shift(-1)
        working["confirmed_next_open"] = grouped["next_open"].shift(-1)
        working["confirmation_ok"] = _strictly_above_high(working["confirmation_close"], working["prior_52w_close_high"])
    elif config.entry_confirmation == "first_close":
        working["confirmed_next_ts"] = working["next_ts"]
        working["confirmed_next_open"] = working["next_open"]
        working["confirmation_ok"] = True
    else:
        raise ValueError(f"unknown entry_confirmation: {config.entry_confirmation}")
    previous_close = working["previous_intraday_close"]
    positivity_ok = (
        working["positivity_filter_ok"].fillna(False)
        if config.use_positivity and "positivity_filter_ok" in working.columns
        else pd.Series(True, index=working.index)
    )
    factor_ok = (
        working["factor_filter_ok"].fillna(False)
        if config.factor_filter != "none" and "factor_filter_ok" in working.columns
        else pd.Series(True, index=working.index)
    )
    mask = (
        working["confirmed_next_open"].notna()
        & working["signal_score"].notna()
        & working["prior_52w_close_high"].notna()
        & working["atr"].notna()
        & _strictly_above_high(working["close"], working["prior_52w_close_high"])
        & (previous_close.isna() | previous_close.le(working["prior_52w_close_high"]))
        & working["hhmm"].gt(config.range_end_hhmm)
        & working["hhmm"].lt(config.exit_hhmm)
        & working["breakout_52w_bps"].ge(config.range_buffer_bps)
        & positivity_ok
        & factor_ok
        & working["confirmation_ok"].fillna(False)
    )
    columns = ["ticker", "date", "ts", "next_ts", "next_open", "atr", "signal_score"]
    entries = working.loc[mask, columns].copy()
    entries["next_ts"] = working.loc[mask, "confirmed_next_ts"].to_numpy()
    entries["next_open"] = working.loc[mask, "confirmed_next_open"].to_numpy()
    return entries.sort_values(["ticker", "date", "ts"]).groupby(["ticker", "date"], sort=True).head(1)


def _strictly_above_high(value: pd.Series, prior_high: pd.Series) -> pd.Series:
    return value.gt(prior_high + prior_high.abs().mul(1e-12))


def _simulate_daily_continuation(entries: pd.DataFrame, daily: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    daily_groups = {str(ticker): group.reset_index(drop=True) for ticker, group in daily.groupby("ticker", sort=True)}
    for ticker, ticker_entries in entries.groupby("ticker", sort=True):
        available_date = pd.Timestamp.min
        ticker_daily = daily_groups.get(str(ticker))
        if ticker_daily is None:
            continue
        for _, signal in ticker_entries.sort_values("ts").iterrows():
            signal_date = pd.Timestamp(signal["date"])
            if signal_date <= available_date:
                continue
            trade = _daily_continuation_trade(signal, ticker_daily, config)
            if trade is None:
                continue
            rows.append(trade)
            available_date = pd.Timestamp(trade["exit_time"]).normalize()
    return pd.DataFrame(rows)


def _daily_continuation_trade(signal: pd.Series, daily: pd.DataFrame, config: TechGammaConfig) -> dict[str, object] | None:
    entry_date = pd.Timestamp(signal["date"])
    entry_price = float(signal["next_open"])
    stop_price = entry_price - float(signal["atr"]) * config.atr_stop_multiplier
    holding_days = (pd.to_datetime(daily["date"]) - entry_date).dt.days
    exits = daily.loc[
        holding_days.ge(config.min_holding_days)
        & (daily["daily_low"].le(stop_price) | daily["close"].le(daily["prior_52w_close_high"]))
    ]
    if exits.empty:
        if daily.empty or int(holding_days.iloc[-1]) < config.min_holding_days:
            return None
        exit_row = daily.iloc[-1]
        exit_reason = "end_of_data"
    else:
        exit_row = exits.iloc[0]
        exit_reason = "atr_stop" if float(exit_row["daily_low"]) <= stop_price else "new_high_lost"
    exit_price = stop_price if exit_reason == "atr_stop" else float(exit_row["close"])
    gross = exit_price / entry_price - 1.0
    exit_time = pd.Timestamp(exit_row["date"]) + pd.Timedelta(hours=15, minutes=30)
    return {
        "ticker": str(signal["ticker"]),
        "side": "long",
        "signal_time": pd.Timestamp(signal["ts"]),
        "entry_time": pd.Timestamp(signal["next_ts"]),
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "signal_score": float(signal["signal_score"]),
        "gross_return": gross,
        "net_return": net_return_after_costs(gross),
        "exit_reason": exit_reason,
    }


def _load_start(config: TechGammaConfig) -> pd.Timestamp:
    return pd.Timestamp(config.start) - pd.Timedelta(days=config.high_lookback_days)


def _ticker_batches(tickers: tuple[str, ...], batch_size: int) -> list[tuple[str, ...]]:
    return [tickers[index : index + batch_size] for index in range(0, len(tickers), batch_size)]


def _combine_trades(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _empty_trades()
    return pd.concat(frames, ignore_index=True).sort_values(["entry_time", "ticker"]).reset_index(drop=True)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "signal_date",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one flow-filtered breakout strategy without loading all 5-minute data at once.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end")
    parser.add_argument("--batch-size", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_json(args.config, start=args.start, end=args.end)
    output = run_fast_prefiltered_single_strategy(config, output_dir=args.output_dir, batch_size=args.batch_size)
    print(output)


if __name__ == "__main__":
    main()
