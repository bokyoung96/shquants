import pandas as pd
import pytest

from etc.sell_sidecar_economics import (
    StrategyRule,
    build_rule_trades,
    pair_sidecar_events,
    recommend_rules,
    summarize_rule_trades,
)


def test_pair_sidecar_events_keeps_only_sell_sidecar_activations() -> None:
    events = pd.DataFrame(
        [
            {
                "event_dt": "2026-01-02 09:06:20",
                "action": "발동",
                "futures_return": -0.052,
            },
            {
                "event_dt": "2026-01-02 09:11:20",
                "action": "발동해제",
                "futures_return": -0.041,
            },
            {
                "event_dt": "2026-01-03 09:06:20",
                "action": "발동",
                "futures_return": 0.052,
            },
            {
                "event_dt": "2026-01-03 09:11:20",
                "action": "발동해제",
                "futures_return": 0.041,
            },
        ]
    )

    pairs = pair_sidecar_events(events, direction="sell")

    assert len(pairs) == 1
    assert pairs.iloc[0]["activation_dt"] == pd.Timestamp("2026-01-02 09:06:20")
    assert pairs.iloc[0]["release_dt"] == pd.Timestamp("2026-01-02 09:11:20")


def test_build_rule_trades_separates_trigger_and_release_economic_regimes() -> None:
    pairs = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-01-02").date(),
                "activation_dt": pd.Timestamp("2026-01-02 09:06:20"),
                "release_dt": pd.Timestamp("2026-01-02 09:11:20"),
                "futures_return_at_trigger": -0.052,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"dt": "2026-01-02 09:09:00", "close": 100.0},
            {"dt": "2026-01-02 09:14:00", "close": 101.0},
            {"dt": "2026-01-02 09:12:00", "close": 101.5},
            {"dt": "2026-01-02 09:42:00", "close": 103.0},
        ]
    )
    rules = [
        StrategyRule(
            name="trigger_follow",
            economic_role="shock_continuation",
            entry_anchor="activation",
            entry_delay_minutes=3,
            exit_anchor="release",
            exit_delay_minutes=3,
            thesis="Follow sell-side imbalance while the halt is active.",
        ),
        StrategyRule(
            name="release_residual",
            economic_role="residual_pressure",
            entry_anchor="release_next_bar",
            entry_delay_minutes=0,
            exit_anchor="entry",
            exit_delay_minutes=30,
            thesis="Capture residual de-risking after the halt ends.",
        ),
    ]

    trades = build_rule_trades(pairs, prices, rules)

    assert trades.loc[trades["rule"] == "trigger_follow", "ret"].iloc[0] == pytest.approx(0.01)
    assert trades.loc[trades["rule"] == "release_residual", "ret"].iloc[0] == pytest.approx(103.0 / 101.5 - 1.0)
    assert set(trades["economic_role"]) == {"shock_continuation", "residual_pressure"}


def test_recommend_rules_prefers_robust_economic_rule_over_tiny_overfit_edge() -> None:
    trades = pd.DataFrame(
        [
            {"rule": "overfit", "economic_role": "residual_pressure", "ret": 0.10},
            {"rule": "core", "economic_role": "shock_continuation", "ret": 0.01},
            {"rule": "core", "economic_role": "shock_continuation", "ret": 0.02},
            {"rule": "core", "economic_role": "shock_continuation", "ret": 0.00},
        ]
    )
    summary = summarize_rule_trades(trades)

    recommended = recommend_rules(summary, min_trades=2)

    assert recommended.iloc[0]["rule"] == "core"
    assert "shock_continuation" in recommended.iloc[0]["economic_takeaway"]
