from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import BacktestResult
from .spec import BacktestSpec


@dataclass(slots=True)
class BacktestReport:
    config: BacktestSpec
    result: BacktestResult
    output_dir: Path
    summary: dict[str, Any]


def write_backtest_outputs(
    *,
    config: BacktestSpec,
    result: BacktestResult,
    output_dir: Path,
    active_weights: pd.DataFrame | None = None,
    active_share: pd.DataFrame | None = None,
) -> BacktestReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    series_dir = output_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)

    _write_series(result.equity.rename("equity"), series_dir / "equity.csv")
    _write_series(result.returns.rename("returns"), series_dir / "returns.csv")
    _write_series(result.turnover.rename("turnover"), series_dir / "turnover.csv")
    result.qty.to_parquet(series_dir / "qty.parquet")
    result.weights.to_parquet(series_dir / "weights.parquet")
    if active_weights is not None:
        active_weights.to_parquet(series_dir / "active_weights.parquet")
    if active_share is not None:
        active_share.to_parquet(series_dir / "active_share.parquet")
        _write_frame(active_share.reset_index(), series_dir / "active_share.csv")

    summary = backtest_summary(config=config, result=result, output_dir=output_dir)
    if active_share is not None and not active_share.empty:
        summary["active_share"] = {
            "rows": int(len(active_share)),
            "mean_pct": float(active_share["active_share_pct"].mean()),
            "min_pct": float(active_share["active_share_pct"].min()),
            "max_pct": float(active_share["active_share_pct"].max()),
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return BacktestReport(config=config, result=result, output_dir=output_dir, summary=summary)


def backtest_summary(*, config: BacktestSpec, result: BacktestResult, output_dir: Path) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "config": config.to_dict(),
        "rows": int(len(result.equity)),
        "date_start": result.equity.index.min().date().isoformat(),
        "date_end": result.equity.index.max().date().isoformat(),
        "summary": {
            "final_equity": float(result.equity.iloc[-1]),
            "total_return": float((result.equity.iloc[-1] / result.equity.iloc[0]) - 1.0),
            "mean_daily_return": float(result.returns.mean()),
            "turnover_sum": float(result.turnover.sum()),
        },
    }


def build_active_weight_outputs(
    target_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if benchmark_weights is None:
        return None, None
    aligned_benchmark = benchmark_weights.reindex_like(target_weights).fillna(0.0).astype(float)
    active_weights = target_weights.astype(float) - aligned_benchmark
    active_share = pd.DataFrame(
        {
            "active_share": active_weights.abs().sum(axis=1) * 0.5,
        },
        index=active_weights.index,
    )
    active_share.index.name = "date"
    active_share["active_share_pct"] = active_share["active_share"] * 100.0
    return active_weights, active_share


def _write_series(series: pd.Series, path: Path) -> None:
    frame = series.to_frame()
    frame.index.name = "date"
    _write_frame(frame.reset_index(), path)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")
    serializable.to_csv(path, index=False)
