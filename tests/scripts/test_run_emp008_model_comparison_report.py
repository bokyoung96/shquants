from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtesting.strategies.emp008.run_model_comparison_report import common_weight_dates


def test_common_weight_dates_keeps_only_shared_rebalance_period(tmp_path: Path) -> None:
    modified = tmp_path / "modified.csv"
    original = tmp_path / "original.csv"
    pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])).to_csv(modified)
    pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=pd.to_datetime(["2024-02-29", "2024-03-29", "2024-04-30"])).to_csv(original)

    dates = common_weight_dates(modified, original)

    assert dates == ("2024-02-29", "2024-03-29")
