from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from backtesting.reporting.models import BenchmarkConfig
from backtesting.saved_runs import config_signature, is_usable_run_dir, normalize_universe_id, normalize_value
from dashboard.backend.schemas import RunOptionModel
from dashboard.backend.services.run_index import RunIndexService
from dashboard.strategies import DashboardLaunchConfig, StrategyPreset, enabled_strategy_presets


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    run_id: str
    strategy_name: str


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    resolved_runs: tuple[ResolvedRun, ...]
    missing_presets: tuple[StrategyPreset, ...]
    archived_run_ids: tuple[str, ...] = ()

    @property
    def selected_run_ids(self) -> list[str]:
        return [resolved_run.run_id for resolved_run in self.resolved_runs]


class LaunchResolutionService:
    def __init__(self, runs_root: Path | None = None) -> None:
        self.runs_root = runs_root

    def resolve(self, config: DashboardLaunchConfig) -> LaunchPlan:
        run_index_service = RunIndexService(self.runs_root)
        # Own newest-first matching here so resolution does not depend on index ordering.
        available_runs = sorted(
            run_index_service.list_runs(dedupe=False),
            key=self._run_sort_key,
            reverse=True,
        )
        available_runs, archived_run_ids = self._archive_duplicate_runs(available_runs, run_index_service.runs_root)
        resolved_runs: list[ResolvedRun] = []
        missing_presets: list[StrategyPreset] = []

        for preset in enabled_strategy_presets(config.strategies):
            desired_signature = self._build_signature(config, preset)
            matched_run = self._find_matching_run(available_runs, desired_signature, run_index_service.runs_root)
            if matched_run is None:
                missing_presets.append(preset)
                continue

            resolved_runs.append(ResolvedRun(run_id=matched_run.run_id, strategy_name=preset.strategy_name))

        return LaunchPlan(
            resolved_runs=tuple(resolved_runs),
            missing_presets=tuple(missing_presets),
            archived_run_ids=archived_run_ids,
        )

    def _archive_duplicate_runs(
        self,
        available_runs: Sequence[RunOptionModel],
        runs_root: Path,
    ) -> tuple[list[RunOptionModel], tuple[str, ...]]:
        active_runs: list[RunOptionModel] = []
        archived_run_ids: list[str] = []
        seen_signatures: set[str] = set()

        for run in available_runs:
            config = self._load_saved_config(runs_root, run.run_id)
            if config is None or not self._is_usable_saved_run(runs_root, run.run_id):
                active_runs.append(run)
                continue

            signature = config_signature(config)
            if signature is None or signature not in seen_signatures:
                if signature is not None:
                    seen_signatures.add(signature)
                active_runs.append(run)
                continue

            self._archive_run_dir(runs_root, run.run_id)
            archived_run_ids.append(run.run_id)

        return active_runs, tuple(archived_run_ids)

    def _find_matching_run(
        self,
        available_runs: Sequence[RunOptionModel],
        desired_signature: dict[str, Any],
        runs_root: Path,
    ) -> RunOptionModel | None:
        for run in available_runs:
            if not self._is_usable_saved_run(runs_root, run.run_id):
                continue
            config = self._load_saved_config(runs_root, run.run_id)
            if config is None:
                continue
            if self._build_saved_signature(config, desired_signature) == desired_signature:
                return run
        return None

    def _load_saved_config(self, runs_root: Path, run_id: str) -> dict[str, Any] | None:
        config_path = runs_root / run_id / "config.json"
        try:
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

        if not isinstance(raw_config, dict):
            return None
        return raw_config

    @staticmethod
    def _build_signature(config: DashboardLaunchConfig, preset: StrategyPreset) -> dict[str, Any]:
        signature = asdict(config.global_config)
        signature["strategy"] = preset.strategy_name
        if preset.schedule is not None:
            signature["schedule"] = preset.schedule
        if preset.fill_mode is not None:
            signature["fill_mode"] = preset.fill_mode
        signature["universe_id"] = normalize_universe_id(preset.universe_id)
        signature["use_k200"] = LaunchResolutionService._normalize_use_k200(
            signature["use_k200"],
            preset.universe_id,
        )
        signature.update(dict(preset.params))
        signature["benchmark_code"] = preset.benchmark.code
        signature["benchmark_name"] = preset.benchmark.name
        signature["benchmark_dataset"] = preset.benchmark.dataset
        signature["warmup_days"] = preset.warmup.extra_days
        return normalize_value(signature)

    @staticmethod
    def _build_saved_signature(saved_config: dict[str, Any], desired_signature: dict[str, Any]) -> dict[str, Any]:
        benchmark = BenchmarkConfig.default_kospi200()
        compat_defaults = {
            "benchmark_code": benchmark.code,
            "benchmark_name": benchmark.name,
            "benchmark_dataset": benchmark.dataset,
            "warmup_days": 0,
            "universe_id": None,
        }
        return normalize_value(
            {
                key: normalize_universe_id(
                    saved_config.get(key, compat_defaults.get(key))
                )
                for key in desired_signature
            }
        )

    @staticmethod
    def _normalize_use_k200(use_k200: Any, universe_id: Any) -> Any:
        if universe_id not in (None, "legacy_k200"):
            return False
        return use_k200

    @staticmethod
    def _archive_run_dir(runs_root: Path, run_id: str) -> None:
        source = runs_root / run_id
        if not source.exists():
            return

        archive_root = runs_root / "_archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        destination = archive_root / run_id
        suffix = 1
        while destination.exists():
            destination = archive_root / f"{run_id}_{suffix}"
            suffix += 1
        shutil.move(str(source), str(destination))

    @staticmethod
    def _is_usable_saved_run(runs_root: Path, run_id: str) -> bool:
        return is_usable_run_dir(runs_root / run_id)

    @staticmethod
    def _run_sort_key(run: RunOptionModel) -> tuple[datetime, str]:
        timestamp = LaunchResolutionService._parse_run_timestamp(run.run_id)
        return timestamp, run.run_id

    @staticmethod
    def _parse_run_timestamp(run_id: str) -> datetime:
        parts = run_id.rsplit("_", 2)
        if len(parts) != 3:
            return datetime.min

        suffix = "_".join(parts[-2:])
        try:
            return datetime.strptime(suffix, "%Y%m%d_%H%M%S")
        except ValueError:
            return datetime.min

