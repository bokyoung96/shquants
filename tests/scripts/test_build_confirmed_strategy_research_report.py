from __future__ import annotations

import pandas as pd

from scripts.build_confirmed_strategy_research_report import (
    factor_impact_table,
    fixed20_metrics,
    write_comparison_dashboard,
    write_factor_impact_dashboard,
)


def test_fixed20_metrics_reports_return_mdd_and_trade_quality() -> None:
    trades = pd.DataFrame({"net_return": [0.10, -0.04, 0.02]})
    ledger = pd.DataFrame(
        {
            "equity": [1.00, 1.05, 1.02, 1.08],
            "drawdown": [0.00, 0.00, -0.0285714286, 0.00],
            "active_positions": [0, 1, 2, 1],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )

    row = fixed20_metrics("example", trades, ledger)

    assert row["strategy"] == "example"
    assert row["trades"] == 3
    assert row["final_return"] == 0.08
    assert row["mdd"] == -0.0285714286
    assert row["hit_rate"] == 2 / 3
    assert row["profit_factor"] == 3.0
    assert row["max_active_positions"] == 2


def test_factor_impact_table_orders_variants_and_computes_incremental_change() -> None:
    metrics = pd.DataFrame(
        [
            {"strategy": "5m_new_high_only", "trades": 100, "final_return": 0.20, "mdd": -0.10},
            {"strategy": "positivity_only", "trades": 70, "final_return": 0.25, "mdd": -0.08},
            {"strategy": "flow_only", "trades": 80, "final_return": 0.18, "mdd": -0.12},
            {"strategy": "current", "trades": 50, "final_return": 0.30, "mdd": -0.05},
        ]
    )

    impact = factor_impact_table(metrics)

    assert impact["strategy"].tolist() == ["5m_new_high_only", "positivity_only", "flow_only", "current"]
    assert impact.loc[impact["strategy"].eq("current"), "trade_reduction_vs_baseline"].iloc[0] == 0.5
    assert impact.loc[impact["strategy"].eq("positivity_only"), "return_delta_vs_baseline"].iloc[0] == 0.05


def test_dashboards_create_png_files(tmp_path) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    ledgers = {
        "5m_new_high_only": pd.DataFrame({"equity": [1.0, 1.02, 1.01], "drawdown": [0.0, 0.0, -0.0098], "active_positions": [0, 1, 0]}, index=dates),
        "current": pd.DataFrame({"equity": [1.0, 1.01, 1.03], "drawdown": [0.0, 0.0, 0.0], "active_positions": [0, 1, 1]}, index=dates),
    }
    metrics = pd.DataFrame(
        [
            {"strategy": "5m_new_high_only", "trades": 2, "final_return": 0.01, "mdd": -0.0098, "hit_rate": 0.5, "profit_factor": 1.2, "max_active_positions": 1},
            {"strategy": "current", "trades": 2, "final_return": 0.03, "mdd": 0.0, "hit_rate": 1.0, "profit_factor": 3.0, "max_active_positions": 1},
        ]
    )

    write_comparison_dashboard(ledgers, metrics, tmp_path / "comparison.png")
    write_factor_impact_dashboard(metrics, tmp_path / "factor.png")

    assert (tmp_path / "comparison.png").stat().st_size > 0
    assert (tmp_path / "factor.png").stat().st_size > 0
