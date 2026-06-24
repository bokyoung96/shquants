from __future__ import annotations

import json
import pandas as pd

from scripts.run_pos_strategy_grid import (
    _build_data_overlay_ranks,
    _pareto_frontier,
    _sector_percentile_rank,
    _split_validation_metrics,
    _write_grid_outputs,
    build_strategy_specs,
    rank_strategy_summary,
)


def test_build_strategy_specs_creates_50_structural_specs_without_thresholds() -> None:
    specs = build_strategy_specs()

    assert len(specs) == 50
    assert len({spec.name for spec in specs}) == 50
    assert {spec.family for spec in specs} >= {"stable_sleeve", "new_high", "pullback_reclaim"}
    assert {spec.params["data_overlay"] for spec in specs} >= {
        "price_only",
        "sponsorship",
        "retail_contrarian",
        "op_revision",
        "eps_revision",
    }
    for spec in specs:
        assert "data_overlay" in spec.params
        assert "overlay_group_count" in spec.params
        assert "threshold" not in spec.params
        assert "floor" not in spec.params
        assert "absolute_cutoff" not in spec.params


def test_rank_strategy_summary_prefers_high_return_low_mdd_and_event_viability() -> None:
    summary = pd.DataFrame(
        [
            {
                "strategy": "low_drawdown",
                "cagr": 0.12,
                "mdd": -0.10,
                "sharpe": 0.9,
                "late_cagr": 0.12,
                "late_mdd": -0.10,
                "validation_pass": 1,
                "trade_count": 20,
                "active_day_ratio": 0.5,
            },
            {
                "strategy": "high_return_bad_mdd",
                "cagr": 0.20,
                "mdd": -0.60,
                "sharpe": 0.7,
                "late_cagr": 0.20,
                "late_mdd": -0.60,
                "validation_pass": 1,
                "trade_count": 40,
                "active_day_ratio": 0.8,
            },
            {
                "strategy": "no_events",
                "cagr": 0.15,
                "mdd": -0.12,
                "sharpe": 1.0,
                "late_cagr": 0.15,
                "late_mdd": -0.12,
                "validation_pass": 1,
                "trade_count": 0,
                "active_day_ratio": 1.0,
            },
        ]
    )

    ranked = rank_strategy_summary(summary)

    assert ranked.iloc[0]["strategy"] == "low_drawdown"
    assert ranked.loc[ranked["strategy"].eq("no_events"), "event_viable"].iloc[0] == 0
    assert ranked.iloc[0]["selection_score"] == ranked.iloc[0]["robust_score"]


def test_rank_strategy_summary_penalizes_late_period_failure() -> None:
    summary = pd.DataFrame(
        [
            {
                "strategy": "full_sample_winner_late_failure",
                "cagr": 0.50,
                "mdd": -0.20,
                "sharpe": 1.4,
                "late_cagr": -0.10,
                "late_mdd": -0.20,
                "validation_pass": 0,
                "trade_count": 30,
            },
            {
                "strategy": "robust_balanced",
                "cagr": 0.20,
                "mdd": -0.25,
                "sharpe": 0.8,
                "late_cagr": 0.20,
                "late_mdd": -0.25,
                "validation_pass": 1,
                "trade_count": 20,
            },
        ]
    )

    ranked = rank_strategy_summary(summary)

    assert ranked.iloc[0]["strategy"] == "robust_balanced"


def test_pareto_frontier_keeps_return_drawdown_tradeoff_candidates() -> None:
    summary = pd.DataFrame(
        [
            {"strategy": "high_return", "robust_return": 0.40, "worst_mdd": -0.40, "event_viable": 1, "validation_pass": 1},
            {"strategy": "low_mdd", "robust_return": 0.16, "worst_mdd": -0.20, "event_viable": 1, "validation_pass": 1},
            {"strategy": "dominated", "robust_return": 0.12, "worst_mdd": -0.35, "event_viable": 1, "validation_pass": 1},
            {"strategy": "not_validated", "robust_return": 0.50, "worst_mdd": -0.10, "event_viable": 1, "validation_pass": 0},
        ]
    )

    frontier = _pareto_frontier(summary)

    assert frontier["strategy"].tolist() == ["high_return", "low_mdd"]


def test_sector_percentile_rank_ranks_names_inside_each_sector_only() -> None:
    idx = pd.to_datetime(["2024-01-02"])
    values = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [2.0], "D": [4.0]}, index=idx)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"], "C": ["Other"], "D": ["Other"]}, index=idx)
    membership = pd.DataFrame(True, index=idx, columns=values.columns)

    ranked = _sector_percentile_rank(values=values, sector=sector, membership=membership)

    assert ranked.loc[idx[0], "A"] == 1.0
    assert ranked.loc[idx[0], "B"] == 0.5
    assert ranked.loc[idx[0], "D"] == 1.0
    assert ranked.loc[idx[0], "C"] == 0.5


def test_build_data_overlay_ranks_uses_flow_and_consensus_inside_sector() -> None:
    idx = pd.bdate_range("2024-01-02", periods=22)
    columns = ["A", "B", "C", "D"]
    membership = pd.DataFrame(True, index=idx, columns=columns)
    sector = pd.DataFrame(
        [["Tech", "Tech", "Other", "Other"] for _ in idx],
        index=idx,
        columns=columns,
    )
    foreign = pd.DataFrame(0.0, index=idx, columns=columns)
    institution = pd.DataFrame(0.0, index=idx, columns=columns)
    retail = pd.DataFrame(0.0, index=idx, columns=columns)
    op = pd.DataFrame(100.0, index=idx, columns=columns)
    eps = pd.DataFrame(100.0, index=idx, columns=columns)
    foreign["A"] = 2.0
    institution["A"] = 1.0
    retail["B"] = -3.0
    op.loc[idx[-1], "C"] = 130.0
    eps.loc[idx[-1], "D"] = 125.0

    ranks = _build_data_overlay_ranks(
        data={
            "foreign": foreign,
            "institution": institution,
            "retail": retail,
            "op": op,
            "eps": eps,
        },
        sector=sector,
        membership=membership,
        lookback=20,
    )

    assert ranks["sponsorship"].loc[idx[-1], "A"] == 1.0
    assert ranks["retail_contrarian"].loc[idx[-1], "B"] == 1.0
    assert ranks["op_revision"].loc[idx[-1], "C"] == 1.0
    assert ranks["eps_revision"].loc[idx[-1], "D"] == 1.0


def test_split_validation_metrics_reports_early_and_late_periods() -> None:
    idx = pd.bdate_range("2024-01-02", periods=8)
    returns = pd.Series([0.01, 0.01, 0.0, 0.01, 0.02, 0.0, 0.01, 0.01], index=idx)

    metrics = _split_validation_metrics(returns)

    assert metrics["early_observations"] == 4
    assert metrics["late_observations"] == 4
    assert metrics["early_cagr"] > 0
    assert metrics["late_cagr"] > 0
    assert metrics["validation_pass"] == 1


def test_write_grid_outputs_persists_selected_strategy(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "strategy": "winner",
                "family": "new_high",
                "selection_score": 1.0,
                "robust_score": 1.0,
                "robust_return": 0.40,
                "worst_mdd": -0.40,
                "validation_pass": 1,
                "event_viable": 1,
                "trade_count": 10,
                "data_overlay": "price_only",
                "missing_param": float("nan"),
            },
            {
                "strategy": "runner_up",
                "family": "stable_sleeve",
                "selection_score": 0.5,
                "robust_score": 0.9,
                "robust_return": 0.35,
                "worst_mdd": -0.30,
                "validation_pass": 1,
                "event_viable": 1,
                "trade_count": 8,
                "data_overlay": "sponsorship",
                "missing_param": 20.0,
            },
            {
                "strategy": "low_mdd",
                "family": "pullback_reclaim",
                "selection_score": 0.4,
                "robust_score": 0.4,
                "robust_return": 0.12,
                "worst_mdd": -0.10,
                "validation_pass": 1,
                "event_viable": 1,
                "trade_count": 6,
                "data_overlay": "op_revision",
                "missing_param": 30.0,
            },
        ]
    )

    _write_grid_outputs(
        output_dir=tmp_path,
        summary=summary,
        start=pd.Timestamp("2024-01-02"),
        end_ts=pd.Timestamp("2024-12-30"),
        specs=build_strategy_specs()[:2],
    )

    selected = json.loads((tmp_path / "selected_strategy.json").read_text(encoding="utf-8"))
    selected_non_price = json.loads((tmp_path / "selected_non_price_strategy.json").read_text(encoding="utf-8"))
    selected_low_mdd = json.loads((tmp_path / "selected_low_mdd_strategy.json").read_text(encoding="utf-8"))
    selected_final_manager = json.loads((tmp_path / "selected_final_manager_strategy.json").read_text(encoding="utf-8"))

    assert (tmp_path / "strategy_grid_summary.csv").exists()
    assert (tmp_path / "top10_strategy_grid_summary.csv").exists()
    assert (tmp_path / "pareto_frontier.csv").exists()
    assert selected["strategy"] == "winner"
    assert selected["missing_param"] is None
    assert selected_non_price["strategy"] == "runner_up"
    assert selected_low_mdd["strategy"] == "low_mdd"
    assert selected_final_manager["strategy"] == "low_mdd"
