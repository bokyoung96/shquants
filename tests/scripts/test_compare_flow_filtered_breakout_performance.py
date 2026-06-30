from __future__ import annotations

import pandas as pd
import pytest


def test_write_comparison_outputs_creates_metrics_and_png(tmp_path) -> None:
    from scripts.compare_flow_filtered_breakout_performance import write_comparison_outputs

    baseline = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2019-01-02 09:05", "2019-01-03 09:05", "2019-01-04 09:05"]),
            "exit_time": pd.to_datetime(["2019-01-02 15:30", "2019-01-03 15:30", "2019-01-04 15:30"]),
            "net_return": [0.01, -0.02, 0.03],
        }
    )
    strengthened = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2019-01-02 09:05", "2019-01-04 09:05"]),
            "exit_time": pd.to_datetime(["2019-01-02 15:30", "2019-01-04 15:30"]),
            "net_return": [0.005, 0.015],
        }
    )

    write_comparison_outputs(
        {"baseline": baseline, "strengthened": strengthened},
        output_dir=tmp_path,
        title="Comparison",
    )

    metrics = pd.read_csv(tmp_path / "comparison_metrics.csv")
    curves = pd.read_csv(tmp_path / "comparison_equity_curves.csv")
    assert metrics["strategy"].tolist() == ["baseline", "strengthened"]
    assert curves.loc[curves["date"].eq("2019-01-04"), "baseline"].iloc[0] == pytest.approx(1.01 * 0.98 * 1.03)
    assert "position_slots" in metrics.columns
    assert metrics.loc[metrics["strategy"].eq("baseline"), "trades"].iloc[0] == 3
    assert metrics.loc[metrics["strategy"].eq("baseline"), "entry_reduction_vs_baseline"].iloc[0] == 0.0
    assert metrics.loc[metrics["strategy"].eq("strengthened"), "entry_reduction_vs_baseline"].iloc[0] == 1 / 3
    assert "profit_factor" in metrics.columns
    assert "avg_holding_days" in metrics.columns
    assert "max_concurrent_positions" in metrics.columns
    assert (tmp_path / "yearly_entries.csv").exists()
    assert (tmp_path / "right_tail_preservation.csv").exists()
    right_tail = pd.read_csv(tmp_path / "right_tail_preservation.csv")
    assert right_tail["strategy"].tolist() == ["baseline", "strengthened"]
    assert "baseline_top5_threshold" in right_tail.columns
    assert (tmp_path / "cumulative_mdd_comparison.png").stat().st_size > 0
