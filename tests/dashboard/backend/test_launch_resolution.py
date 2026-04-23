from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from backtesting.reporting.models import BenchmarkConfig
from dashboard.backend.services.launch_resolution import LaunchResolutionService
from dashboard.strategies import DEFAULT_LAUNCH_CONFIG


def test_resolution_reuses_newest_matching_run(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_090000")
    _write_matching_run(tmp_path, "momentum_20260405_100000")

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert plan.missing_presets == ()


def test_resolution_marks_default_preset_missing_when_global_config_changes(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        global_config=replace(DEFAULT_LAUNCH_CONFIG.global_config, fee=0.001),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == []
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_marks_default_preset_missing_when_no_matching_run_exists(tmp_path: Path) -> None:
    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == []
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_marks_strategy_missing_when_strategy_params_change(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        strategies=(replace(DEFAULT_LAUNCH_CONFIG.strategies[0], params={"top_n": 25, "lookback": 20}),),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == []
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_marks_strategy_missing_when_benchmark_metadata_changes(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        strategies=(
            replace(
                DEFAULT_LAUNCH_CONFIG.strategies[0],
                benchmark=BenchmarkConfig(code="SPX", name="S&P 500"),
            ),
        ),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == []
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
        ),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == []
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_marks_strategy_missing_when_universe_changes(tmp_path: Path) -> None:
    _write_matching_default_runs(tmp_path)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        strategies=(replace(DEFAULT_LAUNCH_CONFIG.strategies[0], universe_id="kosdaq150"),),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == []
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_reuses_legacy_saved_run_when_universe_id_is_legacy_k200(tmp_path: Path) -> None:
    payload = _saved_config()
    payload["universe_id"] = "legacy_k200"
    _write_saved_run(tmp_path, "momentum_20260405_100000", config=payload)

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert plan.missing_presets == ()


def test_resolution_reuses_saved_kosdaq150_run_when_use_k200_is_normalized(tmp_path: Path) -> None:
    payload = _saved_config()
    payload["universe_id"] = "kosdaq150"
    payload["use_k200"] = False
    _write_saved_run(tmp_path, "momentum_20260405_110000", config=payload)
    altered = replace(
        DEFAULT_LAUNCH_CONFIG,
        strategies=(replace(DEFAULT_LAUNCH_CONFIG.strategies[0], universe_id="kosdaq150"),),
    )

    plan = LaunchResolutionService(tmp_path).resolve(altered)

    assert plan.selected_run_ids == ["momentum_20260405_110000"]
    assert plan.missing_presets == ()


def test_resolution_archives_older_duplicate_run(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_090000")
    _write_matching_run(tmp_path, "momentum_20260405_100000")

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert plan.archived_run_ids == ("momentum_20260405_090000",)
    assert not (tmp_path / "momentum_20260405_090000").exists()
    assert (tmp_path / "_archived" / "momentum_20260405_090000").is_dir()


def test_resolution_keeps_older_valid_run_when_newer_duplicate_is_incomplete(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_090000")
    _write_saved_run(tmp_path, "momentum_20260405_100000", config=_saved_config(), artifacts=False)

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_090000"]
    assert plan.archived_run_ids == ()
    assert (tmp_path / "momentum_20260405_090000").is_dir()
    assert (tmp_path / "momentum_20260405_100000").is_dir()


def test_resolution_does_not_match_archived_runs(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path / "_archived",
        "momentum_20260405_100000",
        config=_saved_config(),
    )

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == []
    assert [item.strategy_name for item in plan.missing_presets] == ["momentum"]


def test_resolution_ignores_malformed_saved_config(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_100000")
    _write_saved_run(
        tmp_path,
        "momentum_20260405_110000",
        config_text="{not valid json",
    )

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert plan.missing_presets == ()


def test_resolution_ignores_non_dict_saved_config(tmp_path: Path) -> None:
    _write_matching_run(tmp_path, "momentum_20260405_100000")
    _write_saved_run(
        tmp_path,
        "momentum_20260405_110000",
        config=[1, 2, 3],
    )

    plan = LaunchResolutionService(tmp_path).resolve(DEFAULT_LAUNCH_CONFIG)

    assert plan.selected_run_ids == ["momentum_20260405_100000"]
    assert plan.missing_presets == ()


def _write_matching_default_runs(root: Path) -> None:
    _write_matching_run(root, "momentum_20260405_100000")


def _write_matching_run(root: Path, run_id: str) -> None:
    _write_saved_run(root, run_id, config=_saved_config())


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

    payload = _saved_config() if config is None else config
    (run_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    if artifacts:
        _write_run_artifacts(run_dir)


def _saved_config() -> dict[str, object]:
    payload = asdict(DEFAULT_LAUNCH_CONFIG.global_config)
    preset = DEFAULT_LAUNCH_CONFIG.strategies[0]
    payload["strategy"] = preset.strategy_name
    payload.update(dict(preset.params))
    payload["benchmark_code"] = preset.benchmark.code
    payload["benchmark_name"] = preset.benchmark.name
    payload["benchmark_dataset"] = preset.benchmark.dataset
    payload["warmup_days"] = preset.warmup.extra_days
    payload["universe_id"] = preset.universe_id
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
