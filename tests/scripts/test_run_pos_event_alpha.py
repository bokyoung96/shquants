from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.run_pos_event_alpha as event_alpha
from scripts.run_pos_event_alpha import (
    _entry_membership_window,
    _warmup_start_for_specs,
    _write_event_outputs,
    build_event_strategy_specs,
    rank_event_summary,
)


def test_build_event_strategy_specs_covers_queue_size_stop_and_entry_mode_grid() -> None:
    specs = build_event_strategy_specs()

    assert {spec.params["max_positions"] for spec in specs} == {1, 3, 5}
    assert {spec.params["atr_multiplier"] for spec in specs} == {2.0, 2.5, 3.0}
    assert {spec.params["entry_mode"] for spec in specs} == {"near_high", "breakout"}
    assert len({spec.name for spec in specs}) == len(specs)


def test_rank_event_summary_prefers_viable_positive_alpha_with_robust_score() -> None:
    summary = pd.DataFrame(
        [
            {
                "strategy": "no_events",
                "active_alpha_cagr": 0.30,
                "active_alpha_mdd": -0.10,
                "late_active_alpha_cagr": 0.30,
                "late_active_alpha_mdd": -0.10,
                "trade_count": 0,
                "event_count": 0,
                "active_day_ratio": 0.0,
            },
            {
                "strategy": "fragile_winner",
                "active_alpha_cagr": 0.50,
                "active_alpha_mdd": -0.50,
                "late_active_alpha_cagr": -0.05,
                "late_active_alpha_mdd": -0.30,
                "trade_count": 20,
                "event_count": 25,
                "active_day_ratio": 0.4,
            },
            {
                "strategy": "robust_candidate",
                "active_alpha_cagr": 0.20,
                "active_alpha_mdd": -0.18,
                "late_active_alpha_cagr": 0.18,
                "late_active_alpha_mdd": -0.16,
                "trade_count": 12,
                "event_count": 15,
                "active_day_ratio": 0.3,
            },
        ]
    )

    ranked = rank_event_summary(summary)

    assert ranked.iloc[0]["strategy"] == "robust_candidate"
    assert ranked.loc[ranked["strategy"].eq("no_events"), "event_viable"].iloc[0] == 0
    assert ranked.iloc[0]["selection_score"] == ranked.iloc[0]["robust_score"]


def test_write_event_outputs_persists_summary_and_selected_strategy(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "strategy": "winner",
                "event_viable": 1,
                "validation_pass": 1,
                "selection_score": 2.0,
                "robust_score": 2.0,
                "active_alpha_cagr": 0.20,
                "active_alpha_mdd": -0.10,
                "missing_param": float("nan"),
            },
            {
                "strategy": "runner_up",
                "event_viable": 1,
                "validation_pass": 1,
                "selection_score": 1.0,
                "robust_score": 1.0,
                "active_alpha_cagr": 0.15,
                "active_alpha_mdd": -0.12,
                "missing_param": 3.0,
            },
        ]
    )

    _write_event_outputs(
        output_dir=tmp_path,
        summary=summary,
        start=pd.Timestamp("2024-01-02"),
        end_ts=pd.Timestamp("2024-12-30"),
        specs=build_event_strategy_specs()[:2],
    )

    selected = json.loads((tmp_path / "selected_event_alpha_strategy.json").read_text(encoding="utf-8"))
    config = json.loads((tmp_path / "event_alpha_config.json").read_text(encoding="utf-8"))

    assert (tmp_path / "event_alpha_summary.csv").exists()
    assert (tmp_path / "top10_event_alpha_summary.csv").exists()
    assert selected["strategy"] == "winner"
    assert selected["missing_param"] is None
    assert config["analysis"] == "positivity event-driven long alpha grid"


def test_run_event_alpha_grid_uses_explicit_data_root(monkeypatch, tmp_path) -> None:
    captured_roots: list[object] = []

    class FakeStore:
        def __init__(self, root):
            captured_roots.append(root)
            raise RuntimeError("stop after root capture")

    monkeypatch.setattr(event_alpha, "ParquetStore", FakeStore)

    with pytest.raises(RuntimeError, match="stop after root capture"):
        event_alpha.run_event_alpha_grid(
            start="2024-01-01",
            end="2024-01-31",
            output_dir=tmp_path / "out",
            data_root=tmp_path / "parquet",
        )

    assert captured_roots == [tmp_path / "parquet"]


def test_entry_membership_window_keeps_history_but_blocks_pre_start_entries() -> None:
    idx = pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02"])
    membership = pd.DataFrame(True, index=idx, columns=["A"])

    eligible = _entry_membership_window(membership=membership, start=pd.Timestamp("2024-01-01"))

    assert bool(eligible.loc[idx[0], "A"]) is False
    assert bool(eligible.loc[idx[1], "A"]) is False
    assert bool(eligible.loc[idx[2], "A"]) is True


def test_warmup_start_for_specs_keeps_largest_lookback_plus_buffer() -> None:
    specs = build_event_strategy_specs()

    warmup_start = _warmup_start_for_specs(start=pd.Timestamp("2024-01-01"), specs=specs)

    assert warmup_start == pd.Timestamp("2024-01-01") - pd.offsets.BDay(332)
