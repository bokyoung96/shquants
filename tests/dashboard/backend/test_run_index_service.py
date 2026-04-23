from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dashboard.backend.services.run_index import RunIndexService


def _write_run(
    root: Path,
    run_id: str,
    *,
    name: str,
    strategy: str,
    final_equity: float,
    top_n: int = 20,
    create_artifacts: bool = True,
) -> None:
    run_dir = root / run_id
    (run_dir / "series").mkdir(parents=True)
    (run_dir / "positions").mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "name": name,
                "strategy": strategy,
                "start": "2020-01-01",
                "end": "2020-12-31",
                "schedule": "monthly",
                "top_n": top_n,
                "benchmark_code": "IKS200",
                "benchmark_name": "KOSPI200",
                "benchmark_dataset": "qw_BM",
                "warmup_days": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "final_equity": final_equity,
                "avg_turnover": 0.12,
            }
        ),
        encoding="utf-8",
    )
    if not create_artifacts:
        return

    index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    pd.Series([100.0, final_equity], index=index, name="equity").to_csv(run_dir / "series" / "equity.csv", index_label="date")
    pd.Series([0.0, 0.01], index=index, name="returns").to_csv(run_dir / "series" / "returns.csv", index_label="date")
    pd.Series([0.1, 0.12], index=index, name="turnover").to_csv(run_dir / "series" / "turnover.csv", index_label="date")
    pd.DataFrame({"A": [0.5, 0.5]}, index=index).to_parquet(run_dir / "positions" / "weights.parquet")
    pd.DataFrame({"A": [10.0, 10.0]}, index=index).to_parquet(run_dir / "positions" / "qty.parquet")


def test_list_runs_returns_newest_first(tmp_path: Path) -> None:
    _write_run(tmp_path, "zeta_20240405_090000", name="Momentum", strategy="momentum", final_equity=130_000_000.0)
    _write_run(tmp_path, "alpha_20260405_100000", name="Momentum Variant", strategy="momentum", final_equity=125_000_000.0, top_n=25)

    service = RunIndexService(tmp_path)

    runs = service.list_runs()

    assert [run.run_id for run in runs] == ["alpha_20260405_100000", "zeta_20240405_090000"]
    assert runs[0].label == "Momentum Variant"
    assert runs[0].strategy == "momentum"
    assert runs[0].summary.final_equity == 125_000_000.0


def test_list_runs_ignores_archived_and_duplicate_config_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "momentum_20260405_090000", name="Momentum v1", strategy="momentum", final_equity=121_000_000.0)
    _write_run(tmp_path, "momentum_20260405_100000", name="Momentum v2", strategy="momentum", final_equity=122_000_000.0)
    _write_run(tmp_path, "momentum_alt_20260405_110000", name="Momentum Alt", strategy="momentum", final_equity=119_000_000.0, top_n=25)
    _write_run(tmp_path / "_archived", "momentum_20260404_080000", name="Momentum archived", strategy="momentum", final_equity=118_000_000.0)

    service = RunIndexService(tmp_path)

    runs = service.list_runs()

    assert [run.run_id for run in runs] == ["momentum_alt_20260405_110000", "momentum_20260405_100000"]


def test_list_runs_keeps_older_valid_run_when_newer_duplicate_is_incomplete(tmp_path: Path) -> None:
    _write_run(tmp_path, "momentum_20260405_090000", name="Momentum", strategy="momentum", final_equity=121_000_000.0)
    _write_run(
        tmp_path,
        "momentum_20260405_100000",
        name="Momentum latest",
        strategy="momentum",
        final_equity=122_000_000.0,
        create_artifacts=False,
    )

    service = RunIndexService(tmp_path)

    runs = service.list_runs()

    assert [run.run_id for run in runs] == ["momentum_20260405_090000"]


def test_list_runs_dedupes_legacy_and_new_schema_copies_of_same_config(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "momentum_variant_20260405_090000",
        name="Momentum Variant",
        strategy="momentum",
        final_equity=121_000_000.0,
    )
    legacy_dir = tmp_path / "momentum_variant_20260405_080000"
    _write_run(
        tmp_path,
        legacy_dir.name,
        name="Momentum Variant legacy",
        strategy="momentum",
        final_equity=120_000_000.0,
    )
    config_path = legacy_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.pop("benchmark_code", None)
    payload.pop("benchmark_name", None)
    payload.pop("benchmark_dataset", None)
    payload.pop("warmup_days", None)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    service = RunIndexService(tmp_path)

    runs = service.list_runs()

    assert [run.run_id for run in runs] == ["momentum_variant_20260405_090000"]


def test_list_runs_treats_universe_id_as_part_of_signature(tmp_path: Path) -> None:
    _write_run(tmp_path, "momentum_20260405_090000", name="Momentum", strategy="momentum", final_equity=121.0)
    _write_run(tmp_path, "momentum_20260405_100000", name="Momentum KOSDAQ", strategy="momentum", final_equity=122.0)

    first = tmp_path / "momentum_20260405_090000" / "config.json"
    second = tmp_path / "momentum_20260405_100000" / "config.json"
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    second_payload["universe_id"] = "kosdaq150"
    first.write_text(json.dumps(first_payload), encoding="utf-8")
    second.write_text(json.dumps(second_payload), encoding="utf-8")

    runs = RunIndexService(tmp_path).list_runs()

    assert [run.run_id for run in runs] == ["momentum_20260405_100000", "momentum_20260405_090000"]


def test_list_runs_treats_legacy_k200_and_missing_universe_id_as_equivalent(tmp_path: Path) -> None:
    _write_run(tmp_path, "momentum_20260405_090000", name="Momentum", strategy="momentum", final_equity=121.0)
    _write_run(tmp_path, "momentum_20260405_100000", name="Momentum legacy", strategy="momentum", final_equity=122.0)

    first = tmp_path / "momentum_20260405_090000" / "config.json"
    second = tmp_path / "momentum_20260405_100000" / "config.json"
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    second_payload["universe_id"] = "legacy_k200"
    first.write_text(json.dumps(first_payload), encoding="utf-8")
    second.write_text(json.dumps(second_payload), encoding="utf-8")

    runs = RunIndexService(tmp_path).list_runs()

    assert [run.run_id for run in runs] == ["momentum_20260405_100000"]


def test_list_runs_skips_malformed_json_and_invalid_numeric_values(tmp_path: Path) -> None:
    valid_dir = tmp_path / "valid_run_20260405_120000"
    _write_run(valid_dir.parent, valid_dir.name, name="Momentum", strategy="momentum", final_equity=111.0)

    malformed_config_dir = tmp_path / "bad_config_20260405_110000"
    (malformed_config_dir / "series").mkdir(parents=True)
    (malformed_config_dir / "positions").mkdir()
    (malformed_config_dir / "config.json").write_text("{", encoding="utf-8")
    (malformed_config_dir / "summary.json").write_text(json.dumps({"final_equity": 100.0, "avg_turnover": 0.1}), encoding="utf-8")
    pd.Series([100.0], index=pd.to_datetime(["2020-01-31"]), name="equity").to_csv(malformed_config_dir / "series" / "equity.csv", index_label="date")
    pd.Series([0.0], index=pd.to_datetime(["2020-01-31"]), name="returns").to_csv(malformed_config_dir / "series" / "returns.csv", index_label="date")
    pd.Series([0.1], index=pd.to_datetime(["2020-01-31"]), name="turnover").to_csv(malformed_config_dir / "series" / "turnover.csv", index_label="date")
    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-31"])).to_parquet(malformed_config_dir / "positions" / "weights.parquet")
    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-31"])).to_parquet(malformed_config_dir / "positions" / "qty.parquet")

    malformed_summary_dir = tmp_path / "bad_summary_20260405_100000"
    (malformed_summary_dir / "series").mkdir(parents=True)
    (malformed_summary_dir / "positions").mkdir()
    (malformed_summary_dir / "config.json").write_text(json.dumps({"name": "Broken", "strategy": "broken"}), encoding="utf-8")
    (malformed_summary_dir / "summary.json").write_text("{", encoding="utf-8")
    pd.Series([100.0], index=pd.to_datetime(["2020-01-31"]), name="equity").to_csv(malformed_summary_dir / "series" / "equity.csv", index_label="date")
    pd.Series([0.0], index=pd.to_datetime(["2020-01-31"]), name="returns").to_csv(malformed_summary_dir / "series" / "returns.csv", index_label="date")
    pd.Series([0.1], index=pd.to_datetime(["2020-01-31"]), name="turnover").to_csv(malformed_summary_dir / "series" / "turnover.csv", index_label="date")
    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-31"])).to_parquet(malformed_summary_dir / "positions" / "weights.parquet")
    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-31"])).to_parquet(malformed_summary_dir / "positions" / "qty.parquet")

    invalid_numeric_dir = tmp_path / "bad_numbers_20260405_090000"
    (invalid_numeric_dir / "series").mkdir(parents=True)
    (invalid_numeric_dir / "positions").mkdir()
    (invalid_numeric_dir / "config.json").write_text(json.dumps({"name": "Bad Numbers", "strategy": "broken"}), encoding="utf-8")
    (invalid_numeric_dir / "summary.json").write_text(
        json.dumps({"final_equity": "not-a-number", "avg_turnover": 0.3}),
        encoding="utf-8",
    )
    pd.Series([100.0], index=pd.to_datetime(["2020-01-31"]), name="equity").to_csv(invalid_numeric_dir / "series" / "equity.csv", index_label="date")
    pd.Series([0.0], index=pd.to_datetime(["2020-01-31"]), name="returns").to_csv(invalid_numeric_dir / "series" / "returns.csv", index_label="date")
    pd.Series([0.1], index=pd.to_datetime(["2020-01-31"]), name="turnover").to_csv(invalid_numeric_dir / "series" / "turnover.csv", index_label="date")
    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-31"])).to_parquet(invalid_numeric_dir / "positions" / "weights.parquet")
    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-31"])).to_parquet(invalid_numeric_dir / "positions" / "qty.parquet")

    service = RunIndexService(tmp_path)

    runs = service.list_runs()

    assert [run.run_id for run in runs] == ["valid_run_20260405_120000"]
