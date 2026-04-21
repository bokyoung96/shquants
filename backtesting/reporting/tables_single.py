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
    "alpha": "Alpha",
    "annual_volatility": "Volatility",
    "avg_turnover": "Avg Turnover",
    "beta": "Beta",
    "cagr": "CAGR",
    "calmar": "Calmar",
    "cumulative_return": "Cumulative Return",
    "final_equity": "Final Equity",
    "information_ratio": "Information Ratio",
    "max_drawdown": "Max Drawdown",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "tracking_error": "Tracking Error",
}
_SUMMARY_ORDER = {
    "cumulative_return": 0,
    "cagr": 1,
    "sharpe": 2,
    "sortino": 3,
    "calmar": 4,
    "max_drawdown": 5,
    "annual_volatility": 6,
    "alpha": 7,
    "beta": 8,
    "tracking_error": 9,
    "information_ratio": 10,
    "final_equity": 11,
    "avg_turnover": 12,
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
