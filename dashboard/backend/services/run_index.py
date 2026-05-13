from __future__ import annotations

import json
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from backtesting.saved_runs import config_signature, is_usable_run_dir
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

            signature = config_signature(config)
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
        if not is_usable_run_dir(run_dir):
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

