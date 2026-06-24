from __future__ import annotations

import pandas as pd
import pytest

from backtesting.strategies.positivity import (
    build_band_holding_weights,
    build_positivity_stable_sleeve_strategy,
    build_positivity_pullback_reclaim_strategy,
    build_positivity_new_high_long_only_strategy,
    build_sector_neutral_positivity_long_short_weights,
    build_sector_positivity_event_core_strategy,
    build_sector_positivity_breakout_strategy,
    build_sector_positivity_state,
    build_pure_signal_tilt_strategy,
    build_reacceleration_entry_weights,
    build_signal_band_strategy,
    build_positivity_quintile_returns,
    build_positivity_quintile_weights,
    build_sponsorship_group_weights,
    flow_positivity_score,
    return_momentum_score,
    positivity_score,
)


def test_positivity_score_counts_non_negative_returns_over_rolling_window() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.0, 0.03],
            "B": [-0.01, -0.02, 0.01, 0.02],
        },
        index=idx,
    )

    score = positivity_score(returns, lookback=3, min_periods=3)

    assert pd.isna(score.loc[idx[1], "A"])
    assert score.loc[idx[2], "A"] == pytest.approx(2 / 3)
    assert score.loc[idx[2], "B"] == pytest.approx(1 / 3)
    assert score.loc[idx[3], "A"] == pytest.approx(2 / 3)
    assert score.loc[idx[3], "B"] == pytest.approx(2 / 3)


def test_sector_neutral_positivity_long_short_weights_pairs_best_and_worst_within_sector() -> None:
    idx = pd.to_datetime(["2024-01-05"])
    score = pd.DataFrame({"A": [0.9], "B": [0.2], "C": [0.8], "D": [0.4]}, index=idx)
    membership = pd.DataFrame(True, index=idx, columns=score.columns)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"], "C": ["Other"], "D": ["Other"]}, index=idx)

    weights = build_sector_neutral_positivity_long_short_weights(
        score=score,
        membership=membership,
        sector=sector,
        max_sectors=1,
        pairs_per_sector=1,
    )

    assert weights.loc[idx[0], "A"] == pytest.approx(0.5)
    assert weights.loc[idx[0], "B"] == pytest.approx(-0.5)
    assert weights.loc[idx[0], ["C", "D"]].sum() == pytest.approx(0.0)
    assert weights.loc[idx[0]].sum() == pytest.approx(0.0)
    assert weights.loc[idx[0]].abs().sum() == pytest.approx(1.0)


def test_positivity_new_high_long_only_strategy_enters_breakout_and_exits_20d_low() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 101.0, 104.0, 103.0, 97.0],
            "B": [100.0, 99.0, 100.0, 101.0, 102.0, 103.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Other"] * len(idx)}, index=idx)

    result = build_positivity_new_high_long_only_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        breakout_lookback=2,
        stop_lookback=2,
        relative_signal_groups=1,
    )

    assert result.weights.loc[idx[3], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[4], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[5], "A"] == pytest.approx(0.0)
    assert result.trades["exit_reason"].tolist() == ["stop"]


def test_positivity_new_high_long_only_strategy_requires_price_breakout() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    close = pd.DataFrame({"A": [100.0, 99.0, 101.0, 100.5]}, index=idx)
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx)}, index=idx)

    result = build_positivity_new_high_long_only_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        breakout_lookback=2,
        stop_lookback=2,
        relative_signal_groups=1,
    )

    assert result.weights.to_numpy().sum() == pytest.approx(0.0)
    assert result.trades.empty


def test_positivity_new_high_long_only_strategy_can_use_sector_relative_breakout() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 101.0, 100.5],
            "B": [100.0, 100.0, 80.0, 70.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Tech"] * len(idx)}, index=idx)

    result = build_positivity_new_high_long_only_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        breakout_lookback=2,
        stop_lookback=2,
        relative_signal_groups=1,
        breakout_basis="sector_relative",
    )

    assert result.weights.loc[idx[3], "A"] == pytest.approx(1.0)
    assert result.trades.empty


def test_positivity_new_high_long_only_strategy_caps_one_name_per_sector() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 101.0, 106.0],
            "B": [100.0, 99.0, 101.0, 105.0],
            "C": [100.0, 99.0, 100.0, 104.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    sector = pd.DataFrame(
        {"A": ["Tech"] * len(idx), "B": ["Tech"] * len(idx), "C": ["Other"] * len(idx)},
        index=idx,
    )

    result = build_positivity_new_high_long_only_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=3,
        max_positions_per_sector=1,
        positivity_lookback=2,
        min_periods=2,
        breakout_lookback=2,
        stop_lookback=2,
        relative_signal_groups=1,
    )

    held = result.weights.loc[idx[3]]
    assert int(held.gt(0.0).sum()) == 2
    assert held[["A", "B"]].gt(0.0).sum() == 1


def test_positivity_pullback_reclaim_strategy_enters_after_reclaim_and_exits_pullback_low() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 103.0, 106.0, 102.0, 101.0, 104.0, 103.0, 100.0],
            "B": [100.0, 101.0, 102.0, 101.0, 100.0, 101.0, 102.0, 103.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Other"] * len(idx)}, index=idx)

    result = build_positivity_pullback_reclaim_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        high_lookback=3,
        reclaim_lookback=2,
        pullback_low_lookback=3,
        relative_signal_groups=1,
    )

    assert result.weights.loc[idx[5], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[6], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[7], "A"] == pytest.approx(0.0)
    assert result.trades["exit_reason"].tolist() == ["pullback_low"]


def test_positivity_pullback_reclaim_strategy_requires_prior_high_setup() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    close = pd.DataFrame({"A": [100.0, 99.0, 100.0, 101.0]}, index=idx)
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx)}, index=idx)

    result = build_positivity_pullback_reclaim_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        high_lookback=3,
        reclaim_lookback=2,
        pullback_low_lookback=3,
        relative_signal_groups=1,
    )

    assert result.weights.to_numpy().sum() == pytest.approx(0.0)
    assert result.trades.empty


def test_positivity_stable_sleeve_strategy_rebalances_only_on_month_end() -> None:
    idx = pd.to_datetime(["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01"])
    close = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0], "B": [100.0, 99.0, 98.0, 97.0]}, index=idx)
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Tech"] * len(idx)}, index=idx)

    result = build_positivity_stable_sleeve_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=1,
        max_positions_per_sector=1,
        short_lookback=2,
        mid_lookback=2,
        long_lookback=2,
        min_periods=2,
        entry_group_count=1,
        hold_group_count=1,
    )

    assert result.weights.loc[idx[1]].sum() == pytest.approx(0.0)
    assert result.weights.loc[idx[2], "A"] == pytest.approx(1.0)


def test_positivity_stable_sleeve_strategy_keeps_existing_holder_when_hold_rank_survives() -> None:
    idx = pd.to_datetime(["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29"])
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 101.0, 100.0],
            "B": [100.0, 99.0, 98.0, 103.0, 104.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Tech"] * len(idx)}, index=idx)

    result = build_positivity_stable_sleeve_strategy(
        close=close,
        membership=membership,
        sector=sector,
        max_positions=1,
        max_positions_per_sector=1,
        short_lookback=2,
        mid_lookback=2,
        long_lookback=2,
        min_periods=2,
        entry_group_count=1,
        hold_group_count=1,
    )

    assert result.weights.loc[idx[2], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[4], "A"] == pytest.approx(1.0)
    assert result.trades.empty


def test_build_positivity_quintile_returns_masks_to_k200_and_uses_next_day_returns() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 98.0, 97.0, 99.0],
            "B": [100.0, 99.0, 98.0, 99.0, 100.0],
            "C": [100.0, 101.0, 100.0, 101.0, 102.0],
            "D": [100.0, 100.0, 101.0, 102.0, 103.0],
            "X": [100.0, 130.0, 80.0, 160.0, 40.0],
        },
        index=idx,
    )
    k200 = pd.DataFrame(True, index=idx, columns=close.columns)
    k200["X"] = False

    result = build_positivity_quintile_returns(close=close, membership=k200, lookback=3, q=5, min_periods=3)

    assert result.columns.tolist() == ["q1", "q2", "q3", "q4", "q5"]
    assert result.index.tolist() == [idx[3]]
    assert result.loc[idx[3], "q1"] == pytest.approx(close.loc[idx[4], "A"] / close.loc[idx[3], "A"] - 1.0)
    assert result.loc[idx[3], "q2"] == pytest.approx(close.loc[idx[4], "B"] / close.loc[idx[3], "B"] - 1.0)
    assert result.loc[idx[3], "q3"] == pytest.approx(close.loc[idx[4], "C"] / close.loc[idx[3], "C"] - 1.0)
    assert result.loc[idx[3], "q4"] == pytest.approx(close.loc[idx[4], "D"] / close.loc[idx[3], "D"] - 1.0)
    assert pd.isna(result.loc[idx[3], "q5"])


def test_build_positivity_quintile_weights_assigns_equal_weight_by_score_bucket() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    score = pd.DataFrame(
        {
            "A": [0.1, 0.2],
            "B": [0.2, 0.3],
            "C": [0.3, 0.4],
            "D": [0.4, 0.5],
            "E": [0.5, 0.6],
            "X": [0.9, 0.9],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=score.columns)
    membership["X"] = False

    weights = build_positivity_quintile_weights(score=score, membership=membership, q=5)

    assert set(weights) == {"q1", "q2", "q3", "q4", "q5"}
    assert weights["q1"].loc[idx[0], "A"] == pytest.approx(1.0)
    assert weights["q5"].loc[idx[0], "E"] == pytest.approx(1.0)
    assert weights["q5"].loc[idx[0], "X"] == pytest.approx(0.0)
    for frame in weights.values():
        assert frame.sum(axis=1).eq(1.0).all()


def test_positivity_score_does_not_change_when_future_returns_change() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        ]
    )
    returns = pd.DataFrame({"A": [0.01, -0.01, 0.02, -0.02, 0.50]}, index=idx)
    changed_future = returns.copy()
    changed_future.loc[idx[4], "A"] = -0.50

    score = positivity_score(returns, lookback=3, min_periods=3)
    changed_score = positivity_score(changed_future, lookback=3, min_periods=3)

    pd.testing.assert_series_equal(score.loc[idx[:4], "A"], changed_score.loc[idx[:4], "A"])


def test_return_momentum_score_uses_trailing_total_return_without_future_prices() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        ]
    )
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 10_000.0]}, index=idx)
    changed_future = close.copy()
    changed_future.loc[idx[4], "A"] = 1.0

    score = return_momentum_score(close, lookback=2)
    changed_score = return_momentum_score(changed_future, lookback=2)

    assert score.loc[idx[2], "A"] == pytest.approx(0.21)
    assert score.loc[idx[3], "A"] == pytest.approx(0.21)
    pd.testing.assert_series_equal(score.loc[idx[:4], "A"], changed_score.loc[idx[:4], "A"])


def test_flow_positivity_score_counts_persistent_net_buy_days() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    flow = pd.DataFrame({"A": [1.0, -1.0, 2.0, 3.0]}, index=idx)

    score = flow_positivity_score(flow, lookback=3, min_periods=3)

    assert pd.isna(score.loc[idx[1], "A"])
    assert score.loc[idx[2], "A"] == pytest.approx(2 / 3)
    assert score.loc[idx[3], "A"] == pytest.approx(2 / 3)


def test_sponsorship_group_weights_split_q5_by_persistent_flow() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    columns = ["A", "B", "C", "D", "E"]
    q5_mask = pd.DataFrame(False, index=idx, columns=columns)
    q5_mask.loc[idx[-1], :] = True
    foreign = pd.DataFrame(
        {
            "A": [1.0, 1.0, -0.5, 1.0],
            "B": [-1.0, -1.0, 0.5, -1.0],
            "C": [1.0, 1.0, 1.0, 1.0],
            "D": [1.0, -0.5, 1.0, 1.0],
            "E": [-1.0, -1.0, -1.0, -1.0],
        },
        index=idx,
    )
    institution = pd.DataFrame(
        {
            "A": [-1.0, -1.0, -1.0, -1.0],
            "B": [1.0, 1.0, -0.5, 1.0],
            "C": [1.0, 1.0, 1.0, 1.0],
            "D": [1.0, 0.5, 1.0, 1.0],
            "E": [-1.0, -1.0, -1.0, -1.0],
        },
        index=idx,
    )
    retail = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, 1.0],
            "B": [1.0, 1.0, 1.0, 1.0],
            "C": [1.0, 1.0, 1.0, 1.0],
            "D": [-1.0, -1.0, -1.0, -1.0],
            "E": [1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )

    groups = build_sponsorship_group_weights(
        q5_mask=q5_mask,
        foreign_flow=foreign,
        institution_flow=institution,
        retail_flow=retail,
        lookback=3,
        long_lookback=3,
        threshold=2 / 3,
    )

    assert groups["foreign_persistent"].loc[idx[-1], "A"] > 0.0
    assert groups["institution_persistent"].loc[idx[-1], "B"] > 0.0
    assert groups["dual_sponsorship"].loc[idx[-1], "C"] > 0.0
    assert groups["retail_supply_absorption"].loc[idx[-1], "D"] > 0.0
    assert groups["no_persistent_sponsorship"].loc[idx[-1], "E"] > 0.0


def test_reacceleration_entry_weights_buy_q5_reentry_after_q4_pause_with_sponsorship() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    buckets = pd.DataFrame(
        {
            "A": [5, 5, 4, 4, 5],
            "B": [4, 4, 4, 4, 5],
            "C": [5, 5, 4, 4, 5],
        },
        index=idx,
    )
    sponsorship = pd.DataFrame(False, index=idx, columns=buckets.columns)
    sponsorship.loc[idx[-1], ["A", "B"]] = True

    weights = build_reacceleration_entry_weights(buckets=buckets, sponsorship=sponsorship, prior_lookback=4)

    assert weights.loc[idx[-1], "A"] == pytest.approx(1.0)
    assert weights.loc[idx[-1], "B"] == pytest.approx(0.0)
    assert weights.loc[idx[-1], "C"] == pytest.approx(0.0)


def test_band_holding_weights_enter_on_reentry_and_hold_until_bucket_exits_band() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    buckets = pd.DataFrame(
        {
            "A": [5, 4, 5, 4, 3, 5],
            "B": [5, 4, 5, 5, 5, 5],
            "C": [5, 4, 5, 5, 5, 5],
        },
        index=idx,
    )
    no_sponsor = pd.DataFrame(False, index=idx, columns=buckets.columns)
    retail_supply = pd.DataFrame(False, index=idx, columns=buckets.columns)
    dual = pd.DataFrame(False, index=idx, columns=buckets.columns)
    no_sponsor.loc[idx[2], "A"] = True
    retail_supply.loc[idx[2], "B"] = True
    dual.loc[idx[2], "C"] = True

    weights = build_band_holding_weights(
        buckets=buckets,
        no_sponsor=no_sponsor,
        retail_supply=retail_supply,
        dual_sponsorship=dual,
        prior_lookback=3,
    )

    assert weights.loc[idx[2], "A"] == pytest.approx(0.5)
    assert weights.loc[idx[2], "B"] == pytest.approx(0.5)
    assert weights.loc[idx[2], "C"] == pytest.approx(0.0)
    assert weights.loc[idx[3], "A"] == pytest.approx(0.5)
    assert weights.loc[idx[3], "B"] == pytest.approx(0.5)
    assert weights.loc[idx[4], "A"] == pytest.approx(0.0)
    assert weights.loc[idx[4], "B"] == pytest.approx(1.0)
    assert weights.loc[idx[5], "A"] == pytest.approx(0.0)


def test_signal_band_strategy_enters_with_vetoes_and_exits_on_band_break_or_stop() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    buckets = pd.DataFrame(
        {
            "A": [5, 4, 5, 4, 4, 5],
            "B": [5, 4, 5, 5, 3, 5],
            "C": [5, 4, 5, 5, 5, 5],
            "D": [5, 4, 5, 5, 5, 5],
        },
        index=idx,
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 96.0, 101.0, 100.0, 95.0, 102.0],
            "B": [100.0, 96.0, 101.0, 102.0, 103.0, 104.0],
            "C": [100.0, 96.0, 101.0, 102.0, 103.0, 104.0],
            "D": [100.0, 96.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=idx,
    )
    no_sponsor = pd.DataFrame(False, index=idx, columns=buckets.columns)
    retail_supply = pd.DataFrame(False, index=idx, columns=buckets.columns)
    dual = pd.DataFrame(False, index=idx, columns=buckets.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=buckets.columns)
    no_sponsor.loc[idx[2], ["A", "B"]] = True
    retail_supply.loc[idx[2], "C"] = True
    dual.loc[idx[2], "C"] = True
    consensus_ok.loc[idx[2], "D"] = False
    no_sponsor.loc[idx[5], "A"] = True

    result = build_signal_band_strategy(
        buckets=buckets,
        close=close,
        no_sponsor=no_sponsor,
        retail_supply=retail_supply,
        dual_sponsorship=dual,
        consensus_ok=consensus_ok,
        prior_lookback=3,
        stop_lookback=2,
        breakout_lookback=2,
    )

    assert result.weights.loc[idx[2], "A"] == pytest.approx(0.5)
    assert result.weights.loc[idx[2], "B"] == pytest.approx(0.5)
    assert result.weights.loc[idx[2], "C"] == pytest.approx(0.0)
    assert result.weights.loc[idx[2], "D"] == pytest.approx(0.0)
    assert result.weights.loc[idx[3], "A"] == pytest.approx(0.5)
    assert result.weights.loc[idx[3], "B"] == pytest.approx(0.5)
    assert result.weights.loc[idx[4], "A"] == pytest.approx(0.0)
    assert result.weights.loc[idx[4], "B"] == pytest.approx(0.0)
    assert result.weights.loc[idx[5], "A"] == pytest.approx(1.0)
    assert result.trades["exit_reason"].tolist() == ["stop", "band_exit"]


def test_signal_band_strategy_requires_prior_close_high_breakout_for_entry() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    buckets = pd.DataFrame({"A": [5, 4, 5], "B": [5, 4, 5]}, index=idx)
    close = pd.DataFrame(
        {
            "A": [100.0, 96.0, 101.0],
            "B": [100.0, 96.0, 99.0],
        },
        index=idx,
    )
    no_sponsor = pd.DataFrame(False, index=idx, columns=buckets.columns)
    no_sponsor.loc[idx[2], ["A", "B"]] = True
    retail_supply = pd.DataFrame(False, index=idx, columns=buckets.columns)
    dual = pd.DataFrame(False, index=idx, columns=buckets.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=buckets.columns)

    result = build_signal_band_strategy(
        buckets=buckets,
        close=close,
        no_sponsor=no_sponsor,
        retail_supply=retail_supply,
        dual_sponsorship=dual,
        consensus_ok=consensus_ok,
        prior_lookback=2,
        stop_lookback=2,
        breakout_lookback=2,
    )

    assert result.weights.loc[idx[2], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[2], "B"] == pytest.approx(0.0)


def test_pure_signal_tilt_strategy_limits_holdings_to_top_ranked_entries() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    buckets = pd.DataFrame({"A": [5, 4, 5], "B": [5, 4, 5], "C": [5, 4, 5]}, index=idx)
    close = pd.DataFrame(
        {
            "A": [100.0, 96.0, 101.0],
            "B": [100.0, 96.0, 101.0],
            "C": [100.0, 96.0, 101.0],
        },
        index=idx,
    )
    signal_score = pd.DataFrame(
        {
            "A": [0.7, 0.7, 0.8],
            "B": [0.7, 0.7, 0.5],
            "C": [0.7, 0.7, 0.9],
        },
        index=idx,
    )
    no_sponsor = pd.DataFrame(False, index=idx, columns=buckets.columns)
    no_sponsor.loc[idx[2], ["A", "B", "C"]] = True
    retail_supply = pd.DataFrame(False, index=idx, columns=buckets.columns)
    dual = pd.DataFrame(False, index=idx, columns=buckets.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=buckets.columns)

    result = build_pure_signal_tilt_strategy(
        buckets=buckets,
        close=close,
        signal_score=signal_score,
        no_sponsor=no_sponsor,
        retail_supply=retail_supply,
        dual_sponsorship=dual,
        consensus_ok=consensus_ok,
        max_positions=2,
        prior_lookback=2,
        stop_lookback=2,
        breakout_lookback=2,
    )

    assert result.weights.loc[idx[2], "C"] == pytest.approx(0.5)
    assert result.weights.loc[idx[2], "A"] == pytest.approx(0.5)
    assert result.weights.loc[idx[2], "B"] == pytest.approx(0.0)


def test_sector_positivity_state_uses_prior_benchmark_weights() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    score = pd.DataFrame({"A": [0.2, 0.8], "B": [0.8, 0.2]}, index=idx)
    membership = pd.DataFrame(True, index=idx, columns=score.columns)
    benchmark_weight = pd.DataFrame({"A": [0.25, 0.90], "B": [0.75, 0.10]}, index=idx)
    sector = pd.DataFrame({"A": ["Tech", "Tech"], "B": ["Tech", "Tech"]}, index=idx)

    state = build_sector_positivity_state(
        score=score,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        slope_lookback=1,
    )

    tech = state.loc[(idx[1], "Tech")]
    assert tech["sector_weighted_pos"] == pytest.approx(0.35)
    assert tech["sector_equal_pos"] == pytest.approx(0.5)


def test_sector_positivity_breakout_strategy_enters_on_weekly_expansion_breakout() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 101.0, 103.0],
            "B": [100.0, 99.0, 98.0, 97.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    benchmark_weight = pd.DataFrame(0.5, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * 4, "B": ["Other"] * 4}, index=idx)
    zero_flow = pd.DataFrame(0.0, index=idx, columns=close.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_sector_positivity_breakout_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=zero_flow,
        institution_flow=zero_flow,
        retail_flow=zero_flow,
        consensus_ok=consensus_ok,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        sector_slope_lookback=1,
        breakout_lookback=2,
        stop_lookback=2,
        flow_lookback=2,
        flow_long_lookback=2,
    )

    assert result.weights.loc[idx[-1], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[-1], "B"] == pytest.approx(0.0)
    assert result.entry_candidates.loc[0, "symbol"] == "A"
    assert result.entry_candidates.loc[0, "mode"] == "sector_expansion"


def test_sector_positivity_event_core_strategy_holds_until_stop_after_event_entry() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 101.0, 104.0, 103.0, 97.0],
            "B": [100.0, 99.0, 100.0, 101.0, 102.0, 103.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    benchmark_weight = pd.DataFrame(0.5, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Other"] * len(idx)}, index=idx)
    zero_flow = pd.DataFrame(0.0, index=idx, columns=close.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_sector_positivity_event_core_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=zero_flow,
        institution_flow=zero_flow,
        retail_flow=zero_flow,
        consensus_ok=consensus_ok,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        sector_slope_lookback=1,
        breakout_lookback=2,
        stop_lookback=2,
        flow_lookback=2,
        flow_long_lookback=2,
        min_holding_days=2,
        market_median_lookback=None,
    )

    assert result.weights.loc[idx[3], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[4], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[5], "A"] == pytest.approx(0.0)
    assert result.trades["exit_reason"].tolist() == ["stop"]


def test_sector_positivity_event_core_strategy_keeps_entry_close_low_stop_by_default() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 98.0, 101.0, 105.0, 110.0, 109.0, 104.0],
            "B": [100.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    benchmark_weight = pd.DataFrame(0.5, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Other"] * len(idx)}, index=idx)
    zero_flow = pd.DataFrame(0.0, index=idx, columns=close.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_sector_positivity_event_core_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=zero_flow,
        institution_flow=zero_flow,
        retail_flow=zero_flow,
        consensus_ok=consensus_ok,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        sector_slope_lookback=1,
        breakout_lookback=2,
        stop_lookback=2,
        flow_lookback=2,
        flow_long_lookback=2,
        min_holding_days=2,
        market_median_lookback=None,
    )

    assert result.weights.loc[idx[5], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[6], "A"] == pytest.approx(1.0)
    assert result.trades.empty


def test_sector_positivity_event_core_strategy_uses_trailing_close_low_stop_when_enabled() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 98.0, 101.0, 105.0, 110.0, 109.0, 104.0],
            "B": [100.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    benchmark_weight = pd.DataFrame(0.5, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Other"] * len(idx)}, index=idx)
    zero_flow = pd.DataFrame(0.0, index=idx, columns=close.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_sector_positivity_event_core_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=zero_flow,
        institution_flow=zero_flow,
        retail_flow=zero_flow,
        consensus_ok=consensus_ok,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        sector_slope_lookback=1,
        breakout_lookback=2,
        stop_lookback=2,
        flow_lookback=2,
        flow_long_lookback=2,
        min_holding_days=2,
        trail_stop=True,
        market_median_lookback=None,
    )

    assert result.weights.loc[idx[5], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[6], "A"] == pytest.approx(0.0)
    assert result.trades["exit_reason"].tolist() == ["stop"]


def test_sector_positivity_event_core_strategy_skips_entries_when_market_gate_fails() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 101.0, 104.0, 105.0, 106.0],
            "B": [100.0, 101.0, 100.0, 99.0, 98.0, 97.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    benchmark_weight = pd.DataFrame(0.5, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Other"] * len(idx)}, index=idx)
    zero_flow = pd.DataFrame(0.0, index=idx, columns=close.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_sector_positivity_event_core_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=zero_flow,
        institution_flow=zero_flow,
        retail_flow=zero_flow,
        consensus_ok=consensus_ok,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        sector_slope_lookback=1,
        breakout_lookback=2,
        stop_lookback=2,
        flow_lookback=2,
        flow_long_lookback=2,
        min_holding_days=2,
        market_entry_floor=0.8,
    )

    assert result.weights.to_numpy().sum() == pytest.approx(0.0)
    assert result.entry_candidates.empty


def test_sector_positivity_event_core_strategy_uses_relative_market_median_gate() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 101.0, 104.0, 105.0, 106.0],
            "B": [100.0, 99.0, 100.0, 101.0, 102.0, 103.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    benchmark_weight = pd.DataFrame(0.5, index=idx, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech"] * len(idx), "B": ["Other"] * len(idx)}, index=idx)
    zero_flow = pd.DataFrame(0.0, index=idx, columns=close.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_sector_positivity_event_core_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=zero_flow,
        institution_flow=zero_flow,
        retail_flow=zero_flow,
        consensus_ok=consensus_ok,
        max_positions=1,
        positivity_lookback=2,
        min_periods=2,
        sector_slope_lookback=1,
        breakout_lookback=2,
        stop_lookback=2,
        flow_lookback=2,
        flow_long_lookback=2,
        min_holding_days=2,
        market_entry_floor=None,
        leadership_market_floor=None,
        market_median_lookback=2,
    )

    assert result.entry_candidates["symbol"].tolist() == ["A"]
    assert result.weights.loc[idx[3], "A"] == pytest.approx(1.0)


def test_sector_positivity_event_core_strategy_caps_entries_per_sector() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 99.0, 101.0, 106.0, 107.0, 108.0],
            "B": [100.0, 99.0, 101.0, 105.0, 106.0, 107.0],
            "C": [100.0, 99.0, 100.0, 104.0, 105.0, 106.0],
        },
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    benchmark_weight = pd.DataFrame(1.0 / 3.0, index=idx, columns=close.columns)
    sector = pd.DataFrame(
        {"A": ["Tech"] * len(idx), "B": ["Tech"] * len(idx), "C": ["Other"] * len(idx)},
        index=idx,
    )
    zero_flow = pd.DataFrame(0.0, index=idx, columns=close.columns)
    consensus_ok = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_sector_positivity_event_core_strategy(
        close=close,
        membership=membership,
        benchmark_weight=benchmark_weight,
        sector=sector,
        foreign_flow=zero_flow,
        institution_flow=zero_flow,
        retail_flow=zero_flow,
        consensus_ok=consensus_ok,
        max_positions=3,
        max_positions_per_sector=1,
        positivity_lookback=2,
        min_periods=2,
        sector_slope_lookback=1,
        breakout_lookback=2,
        stop_lookback=2,
        flow_lookback=2,
        flow_long_lookback=2,
        min_holding_days=2,
        market_entry_floor=None,
        leadership_market_floor=None,
        market_median_lookback=2,
    )

    entry_sectors = result.entry_candidates["sector"].tolist()
    assert entry_sectors.count("Tech") == 1
    assert entry_sectors.count("Other") == 1
