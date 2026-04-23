from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd

from .snapshots import PerformanceSnapshot

__all__ = (
    "TearsheetTableBuilder",
    "build_drawdown_episodes_table",
    "build_performance_summary_table",
    "build_sector_weights_table",
    "build_top_holdings_table",
    "build_validation_appendix_table",
)

_SUMMARY_LABELS = {
    "active_return": "Active Return",
    "active_risk": "Active Risk",
    "alpha": "Alpha",
    "annual_volatility": "Volatility",
    "avg_turnover": "Avg Turnover",
    "beta": "Beta",
    "best_day": "Best Day",
    "best_month": "Best Month",
    "best_year": "Best Year",
    "cagr": "CAGR",
    "calmar": "Calmar",
    "conditional_value_at_risk_95": "CVaR 95",
    "correlation": "Correlation",
    "current_drawdown": "Current Drawdown",
    "cumulative_return": "Cumulative Return",
    "downside_capture": "Downside Capture",
    "downside_deviation": "Downside Deviation",
    "final_equity": "Final Equity",
    "information_ratio": "Information Ratio",
    "kurtosis": "Kurtosis",
    "longest_drawdown_days": "Longest Drawdown Days",
    "max_drawdown": "Max Drawdown",
    "month_hit_ratio": "Month Hit Ratio",
    "payoff_ratio": "Payoff Ratio",
    "profit_factor": "Profit Factor",
    "recovery_days": "Recovery Days",
    "sharpe": "Sharpe",
    "skew": "Skew",
    "sortino": "Sortino",
    "tracking_error": "Tracking Error",
    "upside_capture": "Upside Capture",
    "value_at_risk_95": "VaR 95",
    "win_rate": "Win Rate",
    "worst_day": "Worst Day",
    "worst_month": "Worst Month",
    "worst_year": "Worst Year",
    "year_hit_ratio": "Year Hit Ratio",
}
_SUMMARY_ORDER = {
    "cumulative_return": 0,
    "cagr": 1,
    "sharpe": 2,
    "sortino": 3,
    "calmar": 4,
    "max_drawdown": 5,
    "annual_volatility": 6,
    "downside_deviation": 7,
    "win_rate": 8,
    "payoff_ratio": 9,
    "profit_factor": 10,
    "value_at_risk_95": 11,
    "conditional_value_at_risk_95": 12,
    "alpha": 13,
    "beta": 14,
    "correlation": 15,
    "tracking_error": 16,
    "information_ratio": 17,
    "active_return": 18,
    "active_risk": 19,
    "upside_capture": 20,
    "downside_capture": 21,
    "best_day": 22,
    "worst_day": 23,
    "best_month": 24,
    "worst_month": 25,
    "best_year": 26,
    "worst_year": 27,
    "current_drawdown": 28,
    "longest_drawdown_days": 29,
    "recovery_days": 30,
    "month_hit_ratio": 31,
    "year_hit_ratio": 32,
    "skew": 33,
    "kurtosis": 34,
    "final_equity": 35,
    "avg_turnover": 36,
}
_DRAWDOWN_COLUMNS = ("start", "trough", "end", "drawdown", "duration_days", "recovery_days")
_HOLDING_COLUMNS = ("symbol", "target_weight", "abs_weight")
_SECTOR_COLUMNS = ("sector", "weight", "count")
_VALIDATION_COLUMNS = ("note",)


class TearsheetTableBuilder:
    def build(self, snapshot: PerformanceSnapshot, *, notes: tuple[str, ...] = ()) -> dict[str, pd.DataFrame]:
        performance_rows = [
            {"metric_key": metric, "metric": _metric_label(metric), "value": value}
            for metric, value in asdict(snapshot.metrics).items()
        ]

        drawdown_frame = snapshot.drawdowns.episodes.copy()
        if not drawdown_frame.empty:
            drawdown_frame["duration_days"] = (
                pd.to_datetime(drawdown_frame["trough"]) - pd.to_datetime(drawdown_frame["start"])
            ).dt.days
            drawdown_frame["recovery_days"] = (
                pd.to_datetime(drawdown_frame["end"]) - pd.to_datetime(drawdown_frame["trough"])
            ).dt.days

        sector_frame = pd.DataFrame(
            {
                "sector": snapshot.sectors.latest_weighted.index,
                "weight": snapshot.sectors.latest_weighted.values,
                "count": snapshot.sectors.latest_count.reindex(snapshot.sectors.latest_weighted.index).fillna(0.0).values,
            }
        )

        return {
            "performance_summary": build_performance_summary_table(pd.DataFrame(performance_rows)),
            "drawdown_episodes": build_drawdown_episodes_table(drawdown_frame),
            "top_holdings": build_top_holdings_table(snapshot.exposure.latest_holdings),
            "sector_weights": build_sector_weights_table(sector_frame),
            "validation_appendix": build_validation_appendix_table(pd.DataFrame({"note": list(notes)})),
        }


def build_performance_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=("metric_key", "metric", "value"))

    table = frame.copy()
    table = table.loc[table["value"].notna()].copy()
    if table.empty:
        return pd.DataFrame(columns=("metric_key", "metric", "value"))
    if "metric" not in table.columns:
        table["metric"] = table["metric_key"].map(_metric_label)
    table["_order"] = table["metric_key"].map(_metric_order)
    table["_label"] = table["metric"]
    table = table.sort_values(["_order", "_label"], ascending=[True, True]).drop(columns=["_order", "_label"])
    return table.loc[:, ("metric_key", "metric", "value")].reset_index(drop=True)


def build_drawdown_episodes_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_DRAWDOWN_COLUMNS)

    table = frame.copy()
    table = table.sort_values(["drawdown", "start"], ascending=[True, True])
    return table.loc[:, _ordered_columns(table.columns, _DRAWDOWN_COLUMNS)].reset_index(drop=True)


def build_top_holdings_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_HOLDING_COLUMNS)

    table = frame.copy()
    if "abs_weight" not in table.columns:
        table["abs_weight"] = table["target_weight"].astype(float).abs()
    table = table.sort_values(["abs_weight", "symbol"], ascending=[False, True])
    return table.loc[:, _ordered_columns(table.columns, _HOLDING_COLUMNS)].reset_index(drop=True)


def build_sector_weights_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_SECTOR_COLUMNS)

    table = frame.copy()
    table = table.sort_values(["weight", "sector"], ascending=[False, True])
    return table.loc[:, _ordered_columns(table.columns, _SECTOR_COLUMNS)].reset_index(drop=True)


def build_validation_appendix_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_VALIDATION_COLUMNS)

    table = frame.copy()
    table = table.sort_values(["note"], ascending=[True])
    return table.loc[:, _ordered_columns(table.columns, _VALIDATION_COLUMNS)].reset_index(drop=True)


def _metric_label(metric: object) -> str:
    key = str(metric).strip().lower()
    return _SUMMARY_LABELS.get(key, str(metric).replace("_", " ").title())


def _metric_order(metric: object) -> int:
    return _SUMMARY_ORDER.get(str(metric).strip().lower(), 999)


def _ordered_columns(columns: Iterable[str], preferred: Iterable[str]) -> list[str]:
    ordered = [column for column in preferred if column in columns]
    ordered.extend(column for column in columns if column not in ordered)
    return ordered
