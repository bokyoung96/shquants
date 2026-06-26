from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS_PCT = tuple(range(1, 9))
DEFAULT_CODE = "IKS200"


def build_iks200_daily_frame(qw_bm: pd.DataFrame, *, code: str = DEFAULT_CODE) -> pd.DataFrame:
    if not isinstance(qw_bm.columns, pd.MultiIndex):
        raise ValueError("qw_bm must have MultiIndex columns: (code, field)")
    required = {"open", "high", "low", "close"}
    available = set(qw_bm[code].columns) if code in qw_bm.columns.get_level_values(0) else set()
    missing = required.difference(available)
    if missing:
        raise ValueError(f"qw_bm missing {code} fields: {sorted(missing)}")

    daily = qw_bm[code][["open", "high", "low", "close"]].copy()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily = daily.apply(pd.to_numeric, errors="coerce").replace(0.0, np.nan)
    daily = daily.dropna(subset=["open", "high", "low", "close"]).sort_index()
    daily["ret_cc"] = daily["close"].pct_change()
    daily["ret_oc"] = daily["close"] / daily["open"] - 1.0
    daily["date_index"] = np.arange(len(daily), dtype=int)
    return daily


def mark_down_day_events(
    daily: pd.DataFrame,
    *,
    thresholds_pct: Iterable[int] = DEFAULT_THRESHOLDS_PCT,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    thresholds = sorted(int(threshold) for threshold in thresholds_pct)
    for event_date, row in daily.sort_index().iterrows():
        ret_cc = float(row["ret_cc"])
        if not np.isfinite(ret_cc):
            continue
        loss_pct = -ret_cc * 100.0
        for idx, threshold_pct in enumerate(thresholds):
            next_threshold_pct = thresholds[idx + 1] if idx + 1 < len(thresholds) else None
            if loss_pct + 1e-10 >= threshold_pct and (
                next_threshold_pct is None or loss_pct < next_threshold_pct - 1e-10
            ):
                threshold_ret = -threshold_pct / 100.0
                rows.append(
                    {
                        "event_date": pd.Timestamp(event_date),
                        "event_year": int(pd.Timestamp(event_date).year),
                        "event_idx": int(row["date_index"]),
                        "threshold_pct": threshold_pct,
                        "threshold_ret": threshold_ret,
                        "bucket_floor_pct": threshold_pct,
                        "bucket_ceiling_pct": next_threshold_pct,
                        "bucket_label": down_bucket_label(threshold_pct, next_threshold_pct),
                        "event_ret_cc": ret_cc,
                        "event_ret_oc": float(row["ret_oc"]),
                        "event_open": float(row["open"]),
                        "event_high": float(row["high"]),
                        "event_low": float(row["low"]),
                        "event_close": float(row["close"]),
                    }
                )
                break
    return pd.DataFrame(rows)


def down_bucket_label(threshold_pct: int, next_threshold_pct: int | None = None) -> str:
    if next_threshold_pct is None:
        return f"-{threshold_pct}% 이상 하락"
    return f"-{threshold_pct}%~-{next_threshold_pct}% 미만"


def build_next_day_reactions(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    ordered = daily.sort_index()
    dates = list(ordered.index)
    rows: list[dict[str, object]] = []
    for _, event in events.sort_values(["event_date", "threshold_pct"]).iterrows():
        next_idx = int(event["event_idx"]) + 1
        if next_idx >= len(dates):
            continue
        next_date = pd.Timestamp(dates[next_idx])
        next_row = ordered.loc[next_date]
        event_close = float(event["event_close"])
        next_open = float(next_row["open"])
        next_high = float(next_row["high"])
        next_low = float(next_row["low"])
        next_close = float(next_row["close"])

        rows.append(
            {
                "event_date": pd.Timestamp(event["event_date"]),
                "next_date": next_date,
                "event_year": int(event["event_year"]),
                "next_year": int(next_date.year),
                "threshold_pct": int(event["threshold_pct"]),
                "threshold_ret": float(event["threshold_ret"]),
                "bucket_floor_pct": int(event["bucket_floor_pct"])
                if "bucket_floor_pct" in event.index and pd.notna(event["bucket_floor_pct"])
                else int(event["threshold_pct"]),
                "bucket_ceiling_pct": int(event["bucket_ceiling_pct"])
                if "bucket_ceiling_pct" in event.index and pd.notna(event["bucket_ceiling_pct"])
                else np.nan,
                "bucket_label": str(event["bucket_label"])
                if "bucket_label" in event.index
                else down_bucket_label(int(event["threshold_pct"])),
                "event_ret_cc": float(event["event_ret_cc"]),
                "event_ret_oc": float(event["event_ret_oc"]),
                "event_close": event_close,
                "next_open": next_open,
                "next_high": next_high,
                "next_low": next_low,
                "next_close": next_close,
                "gap_ret": next_open / event_close - 1.0,
                "gap_up": bool(next_open > event_close),
                "next_high_ret": next_high / event_close - 1.0,
                "next_low_ret": next_low / event_close - 1.0,
                "next_close_ret": next_close / event_close - 1.0,
                "open_to_close_ret": next_close / next_open - 1.0,
                "gap_to_close_change": next_close / next_open - 1.0,
                "overnight_gain_kept": (next_close - event_close) / (next_open - event_close)
                if not np.isclose(next_open, event_close)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_intraday_next_day_paths(minutes: pd.DataFrame, reactions: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty:
        return pd.DataFrame()
    minute_data = _normalize_minutes(minutes)
    by_day = {
        trade_date: frame.reset_index(drop=True)
        for trade_date, frame in minute_data.groupby("trade_date_kst", sort=True)
    }
    rows: list[dict[str, object]] = []
    for _, reaction in reactions.sort_values(["event_date", "threshold_pct"]).iterrows():
        event_date = pd.Timestamp(reaction["event_date"]).date()
        next_date = pd.Timestamp(reaction["next_date"]).date()
        event_day = by_day.get(event_date)
        next_day = by_day.get(next_date)
        if event_day is None or event_day.empty or next_day is None or next_day.empty:
            continue

        futures_event_close = float(event_day.iloc[-1]["close"])
        next_open = float(next_day.iloc[0]["close"])
        first_ts = next_day.iloc[0].get("ts")
        first_hhmm = str(next_day.iloc[0]["hhmm_kst"])
        for _, bar in next_day.iterrows():
            price = float(bar["close"])
            rows.append(
                {
                    "event_date": pd.Timestamp(reaction["event_date"]),
                    "next_date": pd.Timestamp(reaction["next_date"]),
                    "event_year": int(reaction["event_year"]),
                    "threshold_pct": int(reaction["threshold_pct"]),
                    "event_ret_cc": float(reaction["event_ret_cc"]),
                    "hhmm_kst": str(bar["hhmm_kst"]),
                    "minute_from_open": _minute_offset(first_ts, first_hhmm, bar),
                    "futures_event_close": futures_event_close,
                    "futures_next_open": next_open,
                    "futures_price": price,
                    "ret_from_futures_event_close": price / futures_event_close - 1.0,
                    "ret_from_next_open": price / next_open - 1.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_intraday_paths(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    grouped = paths.groupby(["event_year", "threshold_pct", "minute_from_open"], sort=True)
    summary = grouped[["ret_from_futures_event_close", "ret_from_next_open"]].agg(["count", "mean", "median"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return (
        summary.reset_index()
        .rename(
            columns={
                "ret_from_futures_event_close_count": "n",
                "ret_from_futures_event_close_mean": "mean_ret_from_futures_event_close",
                "ret_from_futures_event_close_median": "median_ret_from_futures_event_close",
                "ret_from_next_open_mean": "mean_ret_from_next_open",
                "ret_from_next_open_median": "median_ret_from_next_open",
            }
        )
        .drop(columns=["ret_from_next_open_count"], errors="ignore")
    )


def summarize_overall(reactions: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty:
        return pd.DataFrame()
    return _summarize(reactions, ["threshold_pct"])


def summarize_yearly(reactions: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty:
        return pd.DataFrame()
    return _summarize(reactions, ["event_year", "threshold_pct"])


def build_yearly_threshold_matrix(yearly: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if yearly.empty:
        return pd.DataFrame()
    return yearly.pivot(index="event_year", columns="threshold_pct", values=value_col).sort_index()


def _normalize_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date_kst", "hhmm_kst", "close"}
    missing = required.difference(minutes.columns)
    if missing:
        raise ValueError(f"minutes missing required columns: {sorted(missing)}")
    data = minutes.copy()
    data["trade_date_kst"] = pd.to_datetime(data["trade_date_kst"]).dt.date
    data["hhmm_kst"] = data["hhmm_kst"].astype(str).str.zfill(4)
    sort_cols = ["trade_date_kst", "hhmm_kst"]
    if "ts" in data.columns:
        data["ts"] = pd.to_datetime(data["ts"], utc=True)
        sort_cols.append("ts")
    return data.sort_values(sort_cols).reset_index(drop=True)


def _minute_offset(first_ts: object, first_hhmm: str, bar: pd.Series) -> int:
    if pd.notna(first_ts) and "ts" in bar.index and pd.notna(bar["ts"]):
        return int((pd.Timestamp(bar["ts"]) - pd.Timestamp(first_ts)).total_seconds() // 60)
    current = str(bar["hhmm_kst"]).zfill(4)
    return (int(current[:2]) * 60 + int(current[2:])) - (int(first_hhmm[:2]) * 60 + int(first_hhmm[2:]))


def _summarize(reactions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in reactions.sort_values("event_date").groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: int(value) for col, value in zip(group_cols, keys, strict=True)}
        if "bucket_label" in group.columns:
            row["bucket_label"] = str(group["bucket_label"].iloc[0])
        if "bucket_floor_pct" in group.columns:
            row["bucket_floor_pct"] = int(group["bucket_floor_pct"].iloc[0])
        if "bucket_ceiling_pct" in group.columns:
            ceiling = group["bucket_ceiling_pct"].iloc[0]
            row["bucket_ceiling_pct"] = int(ceiling) if pd.notna(ceiling) else np.nan
        gap = group["gap_ret"].astype(float)
        close = group["next_close_ret"].astype(float)
        open_to_close = group["open_to_close_ret"].astype(float)
        row.update(
            {
                "n": int(len(group)),
                "mean_event_ret_cc": float(group["event_ret_cc"].mean()),
                "median_event_ret_cc": float(group["event_ret_cc"].median()),
                "gap_up_count": int(group["gap_up"].sum()),
                "gap_up_rate": float(group["gap_up"].mean()),
                "mean_gap_ret": float(gap.mean()),
                "median_gap_ret": float(gap.median()),
                "mean_next_high_ret": float(group["next_high_ret"].mean()),
                "mean_next_low_ret": float(group["next_low_ret"].mean()),
                "mean_next_close_ret": float(close.mean()),
                "median_next_close_ret": float(close.median()),
                "next_close_win_rate": float((close > 0.0).mean()),
                "mean_open_to_close_ret": float(open_to_close.mean()),
                "median_open_to_close_ret": float(open_to_close.median()),
                "open_to_close_win_rate": float((open_to_close > 0.0).mean()),
                "mean_overnight_gain_kept": float(group["overnight_gain_kept"].replace([np.inf, -np.inf], np.nan).mean()),
                "compound_next_close_ret": float((1.0 + close).prod() - 1.0),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
