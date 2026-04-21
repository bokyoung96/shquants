from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from backtesting.reporting.models import BenchmarkConfig
from dashboard.backend.services.launch_resolution import LaunchResolutionService
from dashboard.strategies import DEFAULT_LAUNCH_CONFIG


def test_resolution_reuses_newest_matching_run(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_090000", strategy="momentum")
    _write_matching_run(tmp_path, "momentum_20260405_100000", strategy="momentum")

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["op_fwd_yield"]


def test_resolution_marks_all_presets_missing_when_global_config_changes(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)

    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        global_config=replace(DEFAULT_LAUNCH_CONFIG.global_config, fee=0.001),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == []
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum", "op_fwd_yield"]


def test_resolution_executes_only_missing_strategy_when_partial_matches_exist(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_100000", strategy="momentum")

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["op_fwd_yield"]


def test_resolution_marks_single_strategy_missing_when_strategy_params_change(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)
    updated_strategies = (
        DEFAULT_LAUNCH_CONFIG.strategies[0],
        replace(DEFAULT_LAUNCH_CONFIG.strategies[1], params={"top_n": 25}),
    )
    altered = replace(DEFAULT_LAUNCH_CONFIG, strategies=updated_strategies)

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["op_fwd_yield"]


def test_resolution_marks_strategy_missing_when_benchmark_metadata_changes(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        strategies=(
            replace(
                DEFAULT_LAUNCH_CONFIG.strategies[0],
                benchmark=BenchmarkConfig(code="SPX", name="S&P 500"),
            ),
            DEFAULT_LAUNCH_CONFIG.strategies[1],
        ),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == ["op_fwd_yield_20260405_110000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_marks_strategy_missing_when_warmup_changes(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        strategies=(
            replace(
                DEFAULT_LAUNCH_CONFIG.strategies[0],
                warmup=replace(DEFAULT_LAUNCH_CONFIG.strategies[0].warmup, extra_days=999),
            ),
            DEFAULT_LAUNCH_CONFIG.strategies[1],
        ),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == ["op_fwd_yield_20260405_110000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_marks_strategy_missing_when_universe_changes(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        strategies=(
            replace(DEFAULT_LAUNCH_CONFIG.strategies[0], universe_id="kosdaq150"),
            DEFAULT_LAUNCH_CONFIG.strategies[1],
        ),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == ["op_fwd_yield_20260405_110000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_reuses_legacy_saved_run_when_universe_id_is_legacy_k200(tmp_path: Path) -> None:
    payload = _saved_config("momentum")
    payload["universe_id"] = "legacy_k200"
    _write_saved_run(tmp_path, "momentum_20260405_100000", config=payload)

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["op_fwd_yield"]


def test_resolution_reuses_saved_kosdaq150_run_when_use_k200_is_normalized(tmp_path: Path) -> None:
    payload = _saved_config("op_fwd_yield")
    payload["universe_id"] = "kosdaq150"
    payload["use_k200"] = False
    _write_saved_run(tmp_path, "op_fwd_yield_20260405_110000", config=payload)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        strategies=(
            DEFAULT_LAUNCH_CONFIG.strategies[0],
            replace(DEFAULT_LAUNCH_CONFIG.strategies[1], universe_id="kosdaq150"),
        ),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == ["op_fwd_yield_20260405_110000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_reuses_legacy_saved_run_when_only_compat_fields_are_missing(tmp_path: Path) -> None:
    payload = asdict(DEFAULT_LAUNCH_CONFIG.global_config)
    payload["strategy"] = "op_fwd_yield"
    payload["top_n"] = 20
    _write_saved_run(tmp_path, "op_fwd_yield_20260405_110000", config=payload)

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["op_fwd_yield_20260405_110000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_archives_older_duplicate_run(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_090000", strategy="momentum")
    _write_matching_run(tmp_path, "momentum_20260405_100000", strategy="momentum")

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert plan.archived_run_ids == ("momentum_20260405_090000",)
    assert not (tmp_path / "momentum_20260405_090000").exists()
    assert (tmp_path / "_archived" / "momentum_20260405_090000").is_dir()


def test_resolution_archives_legacy_and_new_schema_duplicates_together(tmp_path: Path) -> None:
    legacy_payload = asdict(DEFAULT_LAUNCH_CONFIG.global_config)
    legacy_payload["strategy"] = "op_fwd_yield"
    legacy_payload["top_n"] = 20
    _write_saved_run(tmp_path, "op_fwd_yield_20260405_090000", config=legacy_payload)
    _write_matching_run(tmp_path, "op_fwd_yield_20260405_100000", strategy="op_fwd_yield")

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert "op_fwd_yield_20260405_100000" in plan.selected_run_ids
    assert plan.archived_run_ids == ("op_fwd_yield_20260405_090000",)
    assert (tmp_path / "_archived" / "op_fwd_yield_20260405_090000").is_dir()


def test_resolution_keeps_older_valid_run_when_newer_duplicate_is_incomplete(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_090000", strategy="momentum")
    _write_saved_run(tmp_path, "momentum_20260405_100000", config=_saved_config("momentum"), artifacts=False)

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_090000"]
    assert plan.archived_run_ids == ()
    assert (tmp_path / "momentum_20260405_090000").is_dir()
    assert (tmp_path / "momentum_20260405_100000").is_dir()


def test_resolution_does_not_match_archived_runs(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path / "_archived",
        "momentum_20260405_100000",
        config=_saved_config("momentum"),
    )

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == []
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum", "op_fwd_yield"]


def test_resolution_ignores_malformed_saved_config(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_100000", strategy="momentum")
    _write_saved_run(
        tmp_path,
        "momentum_20260405_110000",
        config_text="{not valid json",
    )

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["op_fwd_yield"]


def test_resolution_ignores_non_dict_saved_config(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_100000", strategy="momentum")
    _write_saved_run(
        tmp_path,
        "momentum_20260405_110000",
        config=[1, 2, 3],
    )

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert [item.strategy_name for item in plan.missing_presets] == ["op_fwd_yield"]


def _write_matching_default_runs(root: Path) -> None:
    _write_matching_run(root, "momentum_20260405_100000", strategy="momentum")
    _write_matching_run(root, "op_fwd_yield_20260405_110000", strategy="op_fwd_yield")


def _write_matching_run(root: Path, run_id: str, *, strategy: str) -> None:
    _write_saved_run(root, run_id, config=_saved_config(strategy))


def _write_saved_run(
    root: Path,
    run_id: str,
    *,
    config: object | None = None,
    config_text: str | None = None,
    summary: bool = True,
    artifacts: bool = True,
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    if summary:
        (run_dir / "summary.json").write_text(
            json.dumps({"final_equity": 100.0, "avg_turnover": 0.1}),
            encoding="utf-8",
        )

    if config_text is not None:
        (run_dir / "config.json").write_text(config_text, encoding="utf-8")
        return

    payload = _saved_config("momentum") if config is None else config
    (run_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    if artifacts:
        _write_run_artifacts(run_dir)


def _saved_config(strategy: str) -> dict[str, object]:
    payload = asdict(DEFAULT_LAUNCH_CONFIG.global_config)
    payload["strategy"] = strategy

    preset = next(preset for preset in DEFAULT_LAUNCH_CONFIG.strategies if preset.strategy_name == strategy)
    payload.update(dict(preset.params))
    benchmark = getattr(preset, "benchmark", None)
    if benchmark is not None:
        payload["benchmark_code"] = benchmark.code
        payload["benchmark_name"] = benchmark.name
        payload["benchmark_dataset"] = benchmark.dataset

    warmup = getattr(preset, "warmup", None)
    if warmup is not None:
        payload["warmup_days"] = warmup.extra_days
    return payload


def _write_run_artifacts(run_dir: Path) -> None:
    series_dir = run_dir / "series"
    positions_dir = run_dir / "positions"
    series_dir.mkdir(parents=True, exist_ok=True)
    positions_dir.mkdir(parents=True, exist_ok=True)

    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    pd.Series([100.0, 101.0], index=index, name="equity").to_csv(series_dir / "equity.csv", index_label="date")
    pd.Series([0.0, 0.01], index=index, name="returns").to_csv(series_dir / "returns.csv", index_label="date")
    pd.Series([0.0, 0.1], index=index, name="turnover").to_csv(series_dir / "turnover.csv", index_label="date")

    weights = pd.DataFrame({"A": [1.0, 1.0]}, index=index)
    qty = pd.DataFrame({"A": [10.0, 10.0]}, index=index)
    weights.to_parquet(positions_dir / "weights.parquet")
    qty.to_parquet(positions_dir / "qty.parquet")
