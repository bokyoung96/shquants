from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtesting.saved_runs import config_signature, is_usable_run_dir


def test_config_signature_normalizes_legacy_defaults_and_excludes_name() -> None:
    current = {
        "name": "Current",
        "strategy": "trend_rank",
        "start": "2020-01-01",
        "end": "2020-12-31",
        "benchmark_code": "IKS200",
        "benchmark_name": "KOSPI200",
        "benchmark_dataset": "qw_BM",
        "warmup_days": 0,
        "universe_id": None,
    }
    legacy = {
        "name": "Legacy",
        "strategy": "trend_rank",
        "start": "2020-01-01",
        "end": "2020-12-31",
        "universe_id": "legacy_k200",
    }

    assert config_signature(current) == config_signature(legacy)


def test_is_usable_run_dir_requires_saved_run_contract_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260405_100000"
    (run_dir / "series").mkdir(parents=True)
    (run_dir / "positions").mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pd.Series([100.0], index=pd.to_datetime(["2024-01-02"]), name="equity").to_csv(
        run_dir / "series" / "equity.csv",
        index_label="date",
    )
    pd.Series([0.0], index=pd.to_datetime(["2024-01-02"]), name="returns").to_csv(
        run_dir / "series" / "returns.csv",
        index_label="date",
    )
    pd.Series([0.0], index=pd.to_datetime(["2024-01-02"]), name="turnover").to_csv(
        run_dir / "series" / "turnover.csv",
        index_label="date",
    )

    assert is_usable_run_dir(run_dir) is False

    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02"])).to_parquet(
        run_dir / "positions" / "weights.parquet"
    )
    pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02"])).to_parquet(
        run_dir / "positions" / "qty.parquet"
    )

    assert is_usable_run_dir(run_dir) is True
