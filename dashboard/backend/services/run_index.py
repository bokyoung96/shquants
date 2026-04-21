from __future__ import annotations

import json
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from backtesting.reporting.models import BenchmarkConfig
from dashboard.backend.schemas import RunOptionModel, RunSummaryModel
from root import ROOT


class RunIndexService:
    def __init__(self, runs_root: Path | None = None) -> None:
        self.runs_root = runs_root or (ROOT.results_path / "backtests")

    def list_runs(self, *, dedupe: bool = True) -> list[RunOptionModel]:
        runs: list[RunOptionModel] = []
        seen_signatures: set[str] = set()
        if not self.runs_root.exists():
            return runs

        for run_dir in sorted(self.runs_root.iterdir(), key=self._sort_key, reverse=True):
            if run_dir.name == "_archived":
                continue
            if not run_dir.is_dir():
                continue

            loaded = self._load_run_option(run_dir)
            if loaded is None:
                continue

            run, config = loaded
            if not dedupe:
                runs.append(run)
                continue

            signature = self._config_signature(config)
            if signature is None or signature in seen_signatures:
                continue

            seen_signatures.add(signature)
            runs.append(run)

        return runs

    @staticmethod
    def _sort_key(run_dir: Path) -> tuple[datetime, str]:
        timestamp = RunIndexService._parse_run_timestamp(run_dir.name)
        if timestamp is None:
            try:
                timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime)
            except OSError:
                timestamp = datetime.min
        return timestamp, run_dir.name

    @staticmethod
    def _parse_run_timestamp(run_id: str) -> datetime | None:
        parts = run_id.rsplit("_", 2)
        if len(parts) != 3:
            return None
        suffix = "_".join(parts[-2:])
        if not suffix:
            return None
        try:
            return datetime.strptime(suffix, "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    @staticmethod
    def _load_run_option(run_dir: Path) -> tuple[RunOptionModel, dict[str, Any]] | None:
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        if not RunIndexService._is_usable_run_dir(run_dir):
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not isinstance(summary, dict):
                return None
            return RunOptionModel(
                run_id=run_dir.name,
                label=str(config.get("name") or run_dir.name),
                strategy=str(config.get("strategy") or "unknown"),
                start=config.get("start"),
                end=config.get("end"),
                summary=RunSummaryModel(
                    final_equity=float(summary.get("final_equity")),
                    avg_turnover=float(summary.get("avg_turnover")),
                ),
            ), config
        except (OSError, JSONDecodeError, TypeError, ValueError):
            return None

    @staticmethod
    def _is_usable_run_dir(run_dir: Path) -> bool:
        required_paths = (
            run_dir / "config.json",
            run_dir / "summary.json",
            run_dir / "series" / "equity.csv",
            run_dir / "series" / "returns.csv",
            run_dir / "series" / "turnover.csv",
            run_dir / "positions" / "weights.parquet",
            run_dir / "positions" / "qty.parquet",
        )
        return all(path.is_file() for path in required_paths)

    @staticmethod
    def _config_signature(config: dict[str, Any]) -> str | None:
        benchmark = BenchmarkConfig.default_kospi200()
        relevant = {
            key: value
            for key, value in {
                **config,
                "benchmark_code": config.get("benchmark_code", benchmark.code),
                "benchmark_name": config.get("benchmark_name", benchmark.name),
                "benchmark_dataset": config.get("benchmark_dataset", benchmark.dataset),
                "warmup_days": config.get("warmup_days", 0),
                "universe_id": RunIndexService._normalize_universe_id(config.get("universe_id")),
            }.items()
            if key != "name"
        }
        if not relevant:
            return None
        return json.dumps(RunIndexService._normalize_value(relevant), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _normalize_universe_id(value: object) -> object:
        if value in (None, "legacy_k200"):
            return None
        return value

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: RunIndexService._normalize_value(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [RunIndexService._normalize_value(item) for item in value]
        if isinstance(value, tuple):
            return [RunIndexService._normalize_value(item) for item in value]
        return value
