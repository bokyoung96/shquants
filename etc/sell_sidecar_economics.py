from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


ACTIVATION = "\ubc1c\ub3d9"
RELEASE = "\ubc1c\ub3d9\ud574\uc81c"


@dataclass(frozen=True, slots=True)
class StrategyRule:
    name: str
    economic_role: str
    entry_anchor: str
    entry_delay_minutes: int
    exit_anchor: str
    exit_delay_minutes: int
    thesis: str


def pair_sidecar_events(events: pd.DataFrame, *, direction: str = "sell") -> pd.DataFrame:
    data = events.copy()
    data["event_dt"] = pd.to_datetime(data["event_dt"])
    activations = data[data["action"].astype(str).eq(ACTIVATION)].sort_values("event_dt")
    releases = data[data["action"].astype(str).eq(RELEASE)].sort_values("event_dt")

    rows: list[dict[str, object]] = []
    used_release_indices: set[int] = set()
    for _, activation in activations.iterrows():
        matched_release = None
        candidates = releases[
            (releases["event_dt"].dt.date == activation["event_dt"].date())
            & (releases["event_dt"] > activation["event_dt"])
        ]
        for index, release in candidates.iterrows():
            if index not in used_release_indices:
                used_release_indices.add(index)
                matched_release = release
                break
        if matched_release is None:
            continue

        futures_return = float(activation["futures_return"])
        if direction == "sell" and futures_return >= 0.0:
            continue
        if direction == "buy" and futures_return <= 0.0:
            continue
        rows.append(
            {
                "trade_date": activation["event_dt"].date(),
                "activation_dt": activation["event_dt"],
                "release_dt": matched_release["event_dt"],
                "futures_return_at_trigger": futures_return,
                "minutes_to_release": (
                    matched_release["event_dt"] - activation["event_dt"]
                ).total_seconds()
                / 60.0,
            }
        )
    return pd.DataFrame(rows)


def default_rules() -> list[StrategyRule]:
    return [
        StrategyRule(
            "trigger_plus3_release_plus3",
            "shock_continuation",
            "activation",
            3,
            "release",
            3,
            "Enter after the first forced selling wave confirms, then exit shortly after the halt is released.",
        ),
        StrategyRule(
            "trigger_plus5_release_plus10",
            "shock_continuation",
            "activation",
            5,
            "release",
            10,
            "Let the first minutes absorb quote dislocation, then ride sell-side imbalance through release.",
        ),
        StrategyRule(
            "trigger_plus5_close",
            "day_pressure",
            "activation",
            5,
            "close",
            0,
            "Treat sell-sidecar as a full-day de-risking shock rather than only a halt-window scalp.",
        ),
        StrategyRule(
            "release_nextbar_60m",
            "residual_pressure",
            "release_next_bar",
            0,
            "entry",
            60,
            "Avoid same-minute release lookahead and test whether de-risking persists after reopening.",
        ),
        StrategyRule(
            "release_nextbar_close",
            "residual_pressure",
            "release_next_bar",
            0,
            "close",
            0,
            "Conservative residual-pressure variant using the first full minute after release.",
        ),
    ]


def build_rule_trades(
    pairs: pd.DataFrame,
    inverse_prices: pd.DataFrame,
    rules: list[StrategyRule],
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    prices = inverse_prices.copy()
    prices["dt"] = pd.to_datetime(prices["dt"])
    lookup = prices.set_index("dt").sort_index()["close"]
    close_by_date = _close_by_date(prices)

    rows: list[dict[str, object]] = []
    for _, pair in pairs.iterrows():
        for rule in rules:
            entry_dt = _resolve_entry_dt(pair, rule)
            exit_dt = _resolve_exit_dt(pair, rule, entry_dt, close_by_date)
            entry_price = _close_at(lookup, entry_dt)
            exit_price = _close_at(lookup, exit_dt)
            ret = (
                exit_price / entry_price - 1.0
                if np.isfinite(entry_price) and np.isfinite(exit_price)
                else np.nan
            )
            rows.append(
                {
                    "trade_date": pair["trade_date"],
                    "rule": rule.name,
                    "economic_role": rule.economic_role,
                    "thesis": rule.thesis,
                    "activation_dt": pair["activation_dt"],
                    "release_dt": pair["release_dt"],
                    "entry_dt": entry_dt,
                    "exit_dt": exit_dt,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "ret": ret,
                    "futures_return_at_trigger": pair["futures_return_at_trigger"],
                    "minutes_to_release": _minutes_to_release(pair),
                }
            )
    return pd.DataFrame(rows).dropna(subset=["entry_price", "exit_price", "ret"]).reset_index(drop=True)


def summarize_rule_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["rule", "economic_role"]
    if "thesis" in trades.columns:
        group_cols.append("thesis")
    for keys, group in trades.groupby(group_cols, sort=False):
        if len(group_cols) == 3:
            rule, economic_role, thesis = keys
        else:
            rule, economic_role = keys
            thesis = ""
        returns = group["ret"].astype(float).dropna()
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rows.append(
            {
                "rule": rule,
                "economic_role": economic_role,
                "thesis": thesis,
                "n": int(returns.count()),
                "wins": int((returns > 0.0).sum()),
                "losses": int((returns <= 0.0).sum()),
                "win_rate": float((returns > 0.0).mean()),
                "mean_ret": float(returns.mean()),
                "median_ret": float(returns.median()),
                "sum_ret": float(returns.sum()),
                "compound_ret": float(equity.iloc[-1] - 1.0),
                "max_drawdown": float(drawdown.min()),
                "min_ret": float(returns.min()),
                "max_ret": float(returns.max()),
            }
        )
    return pd.DataFrame(rows)


def recommend_rules(summary: pd.DataFrame, *, min_trades: int = 10) -> pd.DataFrame:
    if summary.empty:
        return summary
    candidates = summary[summary["n"] >= min_trades].copy()
    if candidates.empty:
        return candidates
    candidates["economic_takeaway"] = candidates["economic_role"].map(_takeaway).fillna(
        "unclassified economic role"
    )
    candidates["robust_score"] = (
        candidates["compound_ret"]
        + 0.5 * candidates["mean_ret"]
        + 0.02 * candidates["win_rate"]
        - 0.25 * candidates["max_drawdown"].abs()
    )
    return candidates.sort_values(["robust_score", "compound_ret"], ascending=False).reset_index(drop=True)


def _takeaway(role: str) -> str:
    mapping = {
        "shock_continuation": (
            "shock_continuation: the edge comes from joining confirmed sell-side imbalance "
            "while the sidecar halt is still shaping order flow."
        ),
        "residual_pressure": (
            "residual_pressure: the edge tests whether de-risking continues after release; "
            "use next-bar entry to avoid same-minute lookahead."
        ),
        "day_pressure": (
            "day_pressure: the sidecar is treated as a full-session risk-off regime, not only a halt event."
        ),
    }
    return mapping.get(role, role)


def _resolve_entry_dt(pair: pd.Series, rule: StrategyRule) -> pd.Timestamp:
    if rule.entry_anchor == "activation":
        anchor = pd.Timestamp(pair["activation_dt"])
        return _minute_floor(anchor + timedelta(minutes=rule.entry_delay_minutes))
    if rule.entry_anchor == "release":
        anchor = pd.Timestamp(pair["release_dt"])
        return _minute_floor(anchor + timedelta(minutes=rule.entry_delay_minutes))
    if rule.entry_anchor == "release_next_bar":
        anchor = _next_minute(pd.Timestamp(pair["release_dt"]))
        return anchor + timedelta(minutes=rule.entry_delay_minutes)
    raise ValueError(f"unknown entry_anchor: {rule.entry_anchor}")


def _resolve_exit_dt(
    pair: pd.Series,
    rule: StrategyRule,
    entry_dt: pd.Timestamp,
    close_by_date: dict[object, pd.Timestamp],
) -> pd.Timestamp:
    if rule.exit_anchor == "release":
        return _minute_floor(
            pd.Timestamp(pair["release_dt"]) + timedelta(minutes=rule.exit_delay_minutes)
        )
    if rule.exit_anchor == "entry":
        return entry_dt + timedelta(minutes=rule.exit_delay_minutes)
    if rule.exit_anchor == "close":
        return close_by_date.get(str(pair["trade_date"]), pd.NaT)
    raise ValueError(f"unknown exit_anchor: {rule.exit_anchor}")


def _close_by_date(prices: pd.DataFrame) -> dict[object, pd.Timestamp]:
    if "date" not in prices.columns:
        prices = prices.copy()
        prices["date"] = prices["dt"].dt.date
    out: dict[object, pd.Timestamp] = {}
    for date_value, day in prices.groupby(prices["date"].astype(str), sort=False):
        target = pd.Timestamp(f"{date_value} 15:30:00")
        subset = day[day["dt"] <= target]
        if subset.empty:
            continue
        out[str(date_value)] = pd.Timestamp(subset.iloc[-1]["dt"])
    return out


def _minutes_to_release(pair: pd.Series) -> float:
    if "minutes_to_release" in pair.index and pd.notna(pair["minutes_to_release"]):
        return float(pair["minutes_to_release"])
    return float(
        (
            pd.Timestamp(pair["release_dt"]) - pd.Timestamp(pair["activation_dt"])
        ).total_seconds()
        / 60.0
    )


def _close_at(lookup: pd.Series, timestamp: pd.Timestamp) -> float:
    if pd.isna(timestamp) or timestamp not in lookup.index:
        return np.nan
    value = lookup.loc[timestamp]
    if hasattr(value, "iloc"):
        value = value.iloc[-1]
    return float(value)


def _minute_floor(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).floor("min")


def _next_minute(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    floored = ts.floor("min")
    return floored if ts == floored else floored + pd.Timedelta(minutes=1)


__all__ = (
    "StrategyRule",
    "build_rule_trades",
    "default_rules",
    "pair_sidecar_events",
    "recommend_rules",
    "summarize_rule_trades",
)
