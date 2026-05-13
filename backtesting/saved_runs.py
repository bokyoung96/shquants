from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backtesting.reporting.models import BenchmarkConfig


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
    benchmark = BenchmarkConfig.default_kospi200()
    relevant = {
        key: value
        for key, value in {
            **config,
            "benchmark_code": config.get("benchmark_code", benchmark.code),
            "benchmark_name": config.get("benchmark_name", benchmark.name),
            "benchmark_dataset": config.get("benchmark_dataset", benchmark.dataset),
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
