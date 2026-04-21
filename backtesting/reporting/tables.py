from __future__ import annotations

import math

import pandas as pd

from .models import SavedRun

__all__ = (
    "build_appendix_table",
    "build_latest_qty_table",
    "build_latest_weights_table",
    "build_summary_table",
)

_SUMMARY_COLUMNS = ["run_id", "strategy", "cagr", "mdd", "sharpe", "final_equity", "avg_turnover"]
_APPENDIX_COLUMNS = ["run_id", "path", "strategy", "start", "end"]


def build_summary_table(runs: list[SavedRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run.run_id,
                "strategy": str(run.config.get("strategy", "")),
                "cagr": _coerce_metric(run.summary.get("cagr", math.nan)),
                "mdd": _coerce_metric(run.summary.get("mdd", math.nan)),
                "sharpe": _coerce_metric(run.summary.get("sharpe", math.nan)),
                "final_equity": _coerce_metric(run.summary.get("final_equity", math.nan)),
                "avg_turnover": _coerce_metric(run.summary.get("avg_turnover", math.nan)),
            }
        )
    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def build_latest_weights_table(run: SavedRun) -> pd.DataFrame:
    if run.latest_weights is not None:
        return run.latest_weights.copy()
    return _build_latest_table(run.weights, value_column="target_weight", abs_column="abs_weight")


def build_latest_qty_table(run: SavedRun) -> pd.DataFrame:
    if run.latest_qty is not None:
        return run.latest_qty.copy()
    return _build_latest_table(run.qty, value_column="qty", abs_column="abs_qty")


def build_appendix_table(runs: list[SavedRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run.run_id,
                "path": str(run.path),
                "strategy": str(run.config.get("strategy", "")),
                "start": str(run.config.get("start", "")),
                "end": str(run.config.get("end", "")),
            }
        )
    return pd.DataFrame(rows, columns=_APPENDIX_COLUMNS)


def _build_latest_table(frame: pd.DataFrame, *, value_column: str, abs_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", value_column, abs_column])

    latest_date = frame.index.max()
    latest = frame.loc[frame.index == latest_date].iloc[-1]
    table = pd.DataFrame({"symbol": latest.index, value_column: latest.values})
    table = table.loc[table[value_column].ne(0.0)].copy()
    table[abs_column] = table[value_column].abs()
    return table.sort_values([abs_column, "symbol"], ascending=[False, True]).reset_index(drop=True)


def _coerce_metric(value: object) -> float:
    if value is None:
        return math.nan
    return float(value)
