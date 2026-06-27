from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.tech_gamma_is_oos_report import (
    daily_returns_from_trades,
    metrics_table,
    monthly_heatmap,
    rolling_metrics,
    write_report_plots,
)


def test_is_oos_report_metrics_and_plots_are_written(tmp_path: Path) -> None:
    trades = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(["2021-01-04 15:30", "2021-01-05 15:30", "2021-02-01 15:30"]),
            "net_return": [0.02, -0.01, 0.03],
        }
    )

    returns = daily_returns_from_trades(trades)
    metrics = metrics_table(returns, {"is": ("2021-01-01", "2021-01-31"), "oos": ("2021-02-01", "2021-12-31")})
    rolling = rolling_metrics(returns, windows=(2,))
    heatmap = monthly_heatmap(returns)
    write_report_plots(returns, metrics, rolling, heatmap, tmp_path)

    assert metrics["segment"].tolist() == ["full", "is", "oos"]
    assert len(rolling) == 2
    assert heatmap.loc[2021, 1] != 0.0
    assert (tmp_path / "is_oos_equity_dashboard.png").stat().st_size > 0
    assert (tmp_path / "monthly_return_heatmap.png").stat().st_size > 0
