from __future__ import annotations

from typing import Iterable

import pandas as pd

from .snapshots import PerformanceSnapshot

__all__ = (
    "ComparisonTableBuilder",
    "build_benchmark_relative_table",
    "build_holdings_turnover_table",
    "build_ranked_summary_table",
    "build_sector_comparison_table",
)

_RANKED_COLUMNS = ("display_name", "cagr", "sharpe", "max_drawdown", "final_equity")
_RELATIVE_COLUMNS = ("display_name", "alpha", "beta", "tracking_error", "information_ratio")
_HOLDING_TURNOVER_COLUMNS = ("display_name", "holdings_count", "avg_turnover")
_SECTOR_COLUMNS = ("display_name", "top_sector", "top_sector_weight")


class ComparisonTableBuilder:
    def build(self, snapshots: list[PerformanceSnapshot]) -> dict[str, pd.DataFrame]:
        ranked_rows: list[dict[str, object]] = []
        benchmark_rows: list[dict[str, object]] = []
        exposure_rows: list[dict[str, object]] = []
        sector_rows: list[dict[str, object]] = []

        for snapshot in snapshots:
            ranked_rows.append(
                {
                    "display_name": snapshot.display_name,
                    "cagr": snapshot.metrics.cagr,
                    "sharpe": snapshot.metrics.sharpe,
                    "max_drawdown": snapshot.metrics.max_drawdown,
                    "final_equity": snapshot.metrics.final_equity,
                }
            )
            benchmark_rows.append(
                {
                    "display_name": snapshot.display_name,
                    "alpha": snapshot.metrics.alpha,
                    "beta": snapshot.metrics.beta,
                    "tracking_error": snapshot.metrics.tracking_error,
                    "information_ratio": snapshot.metrics.information_ratio,
                }
            )
            exposure_rows.append(
                {
                    "display_name": snapshot.display_name,
                    "holdings_count": float(snapshot.exposure.holdings_count.iloc[-1]),
                    "avg_turnover": snapshot.metrics.avg_turnover,
                }
            )

            sector_weights = snapshot.sectors.latest_weighted.sort_values(ascending=False)
            sector_rows.append(
                {
                    "display_name": snapshot.display_name,
                    "top_sector": "" if sector_weights.empty else str(sector_weights.index[0]),
                    "top_sector_weight": 0.0 if sector_weights.empty else float(sector_weights.iloc[0]),
                }
            )

        return {
            "ranked_summary": build_ranked_summary_table(pd.DataFrame(ranked_rows)),
            "benchmark_relative": build_benchmark_relative_table(pd.DataFrame(benchmark_rows)),
            "exposure_summary": build_holdings_turnover_table(pd.DataFrame(exposure_rows)),
            "sector_summary": build_sector_comparison_table(pd.DataFrame(sector_rows)),
        }


def build_ranked_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_RANKED_COLUMNS)

    table = frame.copy()
    table = table.sort_values(["cagr", "display_name"], ascending=[False, True])
    return table.loc[:, _ordered_columns(table.columns, _RANKED_COLUMNS)].reset_index(drop=True)


def build_benchmark_relative_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_RELATIVE_COLUMNS)

    table = frame.copy().sort_values(["display_name"], ascending=[True])
    return table.loc[:, _ordered_columns(table.columns, _RELATIVE_COLUMNS)].reset_index(drop=True)


def build_holdings_turnover_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_HOLDING_TURNOVER_COLUMNS)

    table = frame.copy()
    table = table.sort_values(["holdings_count", "display_name"], ascending=[False, True])
    return table.loc[:, _ordered_columns(table.columns, _HOLDING_TURNOVER_COLUMNS)].reset_index(drop=True)


def build_sector_comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_SECTOR_COLUMNS)

    table = frame.copy()
    table = table.sort_values(["top_sector_weight", "display_name"], ascending=[False, True])
    return table.loc[:, _ordered_columns(table.columns, _SECTOR_COLUMNS)].reset_index(drop=True)


def _ordered_columns(columns: Iterable[str], preferred: Iterable[str]) -> list[str]:
    ordered = [column for column in preferred if column in columns]
    ordered.extend(column for column in columns if column not in ordered)
    return ordered
