from __future__ import annotations

import pandas as pd

from scripts.build_confirmed_strategy_research_report import (
    factor_impact_table,
    fixed20_metrics,
    write_markdown_report,
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


def test_markdown_report_declares_canonical_full_and_compressed_schema(tmp_path) -> None:
    metrics = pd.DataFrame(
        [
            {"strategy": "5m_new_high_only", "prefilter_candidates": 100, "input_trades": 12, "trades": 10, "skipped_trades": 2, "final_return": 0.20, "mdd": -0.03, "hit_rate": 0.4, "profit_factor": 1.5},
            {"strategy": "positivity_only", "prefilter_candidates": 80, "input_trades": 12, "trades": 10, "skipped_trades": 2, "final_return": 0.20, "mdd": -0.03, "hit_rate": 0.4, "profit_factor": 1.5},
            {"strategy": "flow_only", "prefilter_candidates": 90, "input_trades": 11, "trades": 9, "skipped_trades": 2, "final_return": 0.18, "mdd": -0.025, "hit_rate": 0.45, "profit_factor": 1.6},
            {"strategy": "current", "prefilter_candidates": 70, "input_trades": 11, "trades": 9, "skipped_trades": 2, "final_return": 0.18, "mdd": -0.025, "hit_rate": 0.45, "profit_factor": 1.6},
        ]
    )
    impact = factor_impact_table(metrics)
    yearly = pd.DataFrame(
        [
            {"strategy": "flow_only", "year": 2024, "year_return_pct": 18.0, "year_end_equity": 1.18},
        ]
    )
    validation = {
        "return_accounting": {"net_return_mismatches": 0},
        "source_entry_exit": {
            "entry_price_mismatches": 0,
            "signal_confirmation_violations": 0,
            "exit_condition_violations": 0,
        },
        "universe_membership": {"kospi200_membership_violations": 0},
    }
    compressed_dir = tmp_path / "multi_timeframe_filter_comparison"
    compressed_dir.mkdir()
    pd.DataFrame(
        [
            {
                "strategy": "weekly_sector_rs_plus_daily_vol_compression",
                "selected_trades": 4,
                "compression_vs_current": 0.60,
                "fixed_return": 0.10,
                "mdd": -0.02,
                "avg_trade_return": 0.006,
                "profit_factor": 1.8,
                "max_active_positions": 3,
            }
        ]
    ).to_csv(compressed_dir / "multi_timeframe_filter_metrics.csv", index=False)

    report = write_markdown_report(tmp_path, metrics, impact, yearly, validation, {})
    text = report.read_text(encoding="utf-8")

    assert "Flow-Confirmed 52-Week High Breakout Strategy" in text
    assert "Sector-Relative Volatility-Compressed Breakout Strategy" in text
    assert "Excluded from canonical schema: positivity hard filter" in text
    assert "Compressed variant evidence" in text
    assert "Next Improvement Candidates" in text
