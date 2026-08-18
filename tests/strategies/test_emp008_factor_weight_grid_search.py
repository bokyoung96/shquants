from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.strategies.emp008.experiments.factor_weight_grid_search import (
    DEFAULT_WEIGHT_OPTIONS,
    _build_annual_tables,
    build_default_candidates,
    percentages_to_multipliers,
    validate_percentages,
)
from backtesting.strategies.emp008.reports.factor_weight_grid_heatmap import (
    _build_display_metrics,
    _yearly_cumulative_excess,
)


def test_default_candidates_are_positive_unique_and_sum_to_100() -> None:
    candidates = build_default_candidates()

    assert len(candidates) == 9
    assert len({candidate.id for candidate in candidates}) == 9
    assert len({tuple(candidate.percentages.items()) for candidate in candidates}) == 9
    assert candidates[0].id == "equal_25"
    for candidate in candidates:
        assert sum(candidate.percentages.values()) == pytest.approx(100.0)
        assert min(candidate.percentages.values()) >= 10.0


def test_tilted_grid_keeps_size_and_momentum_as_the_larger_pair() -> None:
    candidates = build_default_candidates()

    assert tuple(DEFAULT_WEIGHT_OPTIONS) == (
        "ln_market_cap",
        "momentum_12m",
        "earnings_momentum",
        "value",
    )
    for candidate in candidates[1:]:
        assert candidate.percentages["ln_market_cap"] >= 30.0
        assert candidate.percentages["momentum_12m"] >= 30.0
        assert candidate.percentages["earnings_momentum"] in {10.0, 20.0}
        assert candidate.percentages["value"] in {10.0, 20.0}


def test_equal_percentages_map_to_unit_multipliers_for_dynamic_factor_count() -> None:
    percentages = {"a": 100.0 / 3.0, "b": 100.0 / 3.0, "c": 100.0 / 3.0}

    result = percentages_to_multipliers(percentages)

    pd.testing.assert_series_equal(
        result, pd.Series({"a": 1.0, "b": 1.0, "c": 1.0}, dtype=float)
    )


def test_annual_tables_drop_the_prior_year_zero_baseline_row() -> None:
    daily = pd.DataFrame(
        {"IKS200": [0.0, 0.10], "candidate": [0.0, 0.20]},
        index=pd.to_datetime(["2019-12-30", "2020-01-02"]),
    )

    annual_returns, annual_excess = _build_annual_tables(daily)

    assert annual_returns.index.tolist() == [2020]
    assert annual_returns.loc[2020, "candidate"] == pytest.approx(20.0)
    assert annual_excess.loc[2020, "candidate"] == pytest.approx(
        (1.2 / 1.1 - 1.0) * 100.0
    )


def test_heatmap_uses_cumulative_return_difference_and_daily_annualized_ir() -> None:
    summary = pd.DataFrame({"candidate_id": ["candidate"]})
    daily = pd.DataFrame(
        {
            "IKS200": [0.00, 0.01, -0.02],
            "candidate": [0.01, 0.03, -0.01],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )

    result = _build_display_metrics(summary, daily).iloc[0]

    benchmark_total = (1.0 + daily["IKS200"]).prod() - 1.0
    candidate_total = (1.0 + daily["candidate"]).prod() - 1.0
    active = daily["candidate"] - daily["IKS200"]
    expected_ir = active.mean() / active.std(ddof=1) * np.sqrt(252.0)
    assert result["cumulative_excess_pct_point"] == pytest.approx(
        (candidate_total - benchmark_total) * 100.0
    )
    assert result["information_ratio"] == pytest.approx(expected_ir)


def test_yearly_cumulative_excess_resets_compounding_each_year() -> None:
    daily = pd.DataFrame(
        {
            "IKS200": [0.10, 0.00, 0.20, 0.00],
            "candidate": [0.20, 0.10, 0.30, 0.10],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2021-01-04", "2021-01-05"]),
    )

    result = _yearly_cumulative_excess(daily, ["candidate"])

    assert list(result) == [2020, 2021]
    assert result[2020].iloc[-1, 0] == pytest.approx(((1.2 * 1.1) - 1.1) * 100.0)
    assert result[2021].iloc[-1, 0] == pytest.approx(((1.3 * 1.1) - 1.2) * 100.0)


@pytest.mark.parametrize(
    "percentages",
    [
        {"a": 0.0, "b": 100.0},
        {"a": -10.0, "b": 110.0},
        {"a": 40.0, "b": 50.0},
        {"a": float("nan"), "b": 100.0},
    ],
)
def test_validate_percentages_rejects_zero_invalid_or_non_100_weights(
    percentages: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        validate_percentages(percentages)
