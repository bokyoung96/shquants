from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class SavedRun:
    run_id: str
    path: Path
    config: dict[str, object]
    summary: dict[str, float]
    equity: pd.Series
    returns: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    qty: pd.DataFrame
    monthly_returns: pd.Series | None = None
    latest_qty: pd.DataFrame | None = None
    latest_weights: pd.DataFrame | None = None
    bucket_ledger: pd.DataFrame | None = None
    validation: dict[str, object] | None = None
    split: dict[str, object] | None = None
    factor: dict[str, object] | None = None
    timing: dict[str, float] | None = None


REQUIRED_RUN_FILES = (
    Path("config.json"),
    Path("summary.json"),
    Path("series") / "equity.csv",
    Path("series") / "returns.csv",
    Path("series") / "turnover.csv",
    Path("positions") / "weights.parquet",
    Path("positions") / "qty.parquet",
)


def is_usable_run_dir(run_dir: Path) -> bool:
    return all((run_dir / path).is_file() for path in REQUIRED_RUN_FILES)


def config_signature(config: dict[str, Any]) -> str | None:
    relevant = {
        key: value
        for key, value in {
            **config,
            "benchmark_code": config.get("benchmark_code", "IKS200"),
            "benchmark_name": config.get("benchmark_name", "KOSPI200"),
            "benchmark_dataset": config.get("benchmark_dataset", "qw_BM"),
            "warmup_days": config.get("warmup_days", 0),
            "universe_id": normalize_universe_id(config.get("universe_id")),
        }.items()
        if key != "name"
    }
    if not relevant:
        return None
    return json.dumps(normalize_value(relevant), sort_keys=True, separators=(",", ":"))


def normalize_universe_id(value: Any) -> Any:
    if value in (None, "legacy_k200"):
        return None
    return value


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    return value
