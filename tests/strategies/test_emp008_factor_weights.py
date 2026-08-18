from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from backtesting.strategies.emp008.data import Emp008Config
from backtesting.strategies.emp008.factor_pipeline import PreparedEmp008Factors
from backtesting.strategies.emp008.factor_registry import FactorSetId, get_factor_set_definition
from backtesting.strategies.emp008.optimize import OptimizationResult
from backtesting.strategies.emp008.risk import compute_expected_alpha
from backtesting.strategies.emp008 import strategy


def test_default_factor_weights_are_one_for_any_factor_count() -> None:
    resolve = getattr(strategy, "resolve_factor_weights", None)
    percentages = getattr(strategy, "factor_weight_percentages", None)

    assert resolve is not None
    assert percentages is not None

    weights = resolve(("size", "momentum", "value"))

    pd.testing.assert_series_equal(
        weights,
        pd.Series({"size": 1.0, "momentum": 1.0, "value": 1.0}, dtype=float),
    )
    pd.testing.assert_series_equal(
        percentages(weights),
        pd.Series({"size": 100.0 / 3.0, "momentum": 100.0 / 3.0, "value": 100.0 / 3.0}, dtype=float),
    )


def test_factor_weight_overrides_keep_unspecified_factors_at_one() -> None:
    resolve = getattr(strategy, "resolve_factor_weights", None)
    percentages = getattr(strategy, "factor_weight_percentages", None)

    assert resolve is not None
    assert percentages is not None

    weights = resolve(("size", "momentum", "value"), {"value": 2.0})

    pd.testing.assert_series_equal(
        weights,
        pd.Series({"size": 1.0, "momentum": 1.0, "value": 2.0}, dtype=float),
    )
    pd.testing.assert_series_equal(
        percentages(weights),
        pd.Series({"size": 25.0, "momentum": 25.0, "value": 50.0}, dtype=float),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": 1.0}, "unknown factor weights"),
        ({"size": -1.0}, "non-negative"),
        ({"size": float("nan")}, "finite"),
        ({"size": 0.0, "momentum": 0.0}, "at least one factor weight"),
    ],
)
def test_factor_weights_reject_invalid_values(overrides: dict[str, float], message: str) -> None:
    resolve = getattr(strategy, "resolve_factor_weights", None)

    assert resolve is not None
    with pytest.raises(ValueError, match=message):
        resolve(("size", "momentum"), overrides)


def test_factor_weights_scale_alpha_factors_without_touching_sector_entries() -> None:
    apply_weights = getattr(strategy, "apply_factor_weights", None)

    assert apply_weights is not None

    expected_alpha = pd.Series(
        {"ln_market_cap": -0.03, "momentum_12_1m": 0.02, "sector_tech": 0.0},
        dtype=float,
    )
    weights = pd.Series({"ln_market_cap": 2.0, "momentum_12_1m": 0.5}, dtype=float)

    result = apply_weights(expected_alpha, weights)

    pd.testing.assert_series_equal(
        result,
        pd.Series({"ln_market_cap": -0.06, "momentum_12_1m": 0.01, "sector_tech": 0.0}, dtype=float),
    )


def test_expected_alpha_defaults_to_trailing_arithmetic_mean() -> None:
    factor_returns = pd.DataFrame(
        {
            "size": [0.01, 0.03, -0.02],
            "momentum": [-0.01, 0.02, 0.04],
            "sector_tech": [0.05, 0.06, 0.07],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )

    actual = compute_expected_alpha(
        factor_returns,
        alpha_factor_names=["size", "momentum"],
        sector_factor_names=["sector_tech"],
        window=2,
    )

    expected = factor_returns.tail(2).mean(axis=0)
    expected.loc["sector_tech"] = 0.0
    pd.testing.assert_series_equal(actual, expected)


def test_expected_alpha_ewma36_matches_pandas_span_36_on_trailing_window() -> None:
    factor_returns = pd.DataFrame(
        {
            "size": [value / 10_000.0 for value in range(1, 41)],
            "momentum": [value / 20_000.0 for value in range(40, 0, -1)],
            "sector_tech": [0.01] * 40,
        },
        index=pd.date_range("2021-01-31", periods=40, freq="ME"),
    )

    actual = compute_expected_alpha(
        factor_returns,
        alpha_factor_names=["size", "momentum"],
        sector_factor_names=["sector_tech"],
        window=36,
        estimator="ewma36",
    )

    expected = factor_returns.tail(36).ewm(span=36, adjust=True).mean().iloc[-1]
    expected.loc["sector_tech"] = 0.0
    pd.testing.assert_series_equal(actual, expected)


def test_expected_alpha_mean_1se_shrinks_each_mean_toward_zero() -> None:
    positive = pd.Series([0.001 + value / 100_000.0 for value in range(36)])
    negative = -positive
    weak = pd.Series([0.010] * 18 + [-0.009] * 18)
    factor_returns = pd.DataFrame(
        {
            "positive": positive.to_list(),
            "negative": negative.to_list(),
            "weak": weak.to_list(),
            "sector_tech": [0.01] * 36,
        },
        index=pd.date_range("2021-01-31", periods=36, freq="ME"),
    )

    actual = compute_expected_alpha(
        factor_returns,
        alpha_factor_names=["positive", "negative", "weak"],
        sector_factor_names=["sector_tech"],
        window=36,
        estimator="mean_1se",
    )

    recent = factor_returns.tail(36)
    mean = recent.mean(axis=0)
    standard_error = recent.std(axis=0, ddof=1).div(recent.count().pow(0.5))
    expected = np.sign(mean).mul(mean.abs().sub(standard_error).clip(lower=0.0))
    expected.loc["sector_tech"] = 0.0
    pd.testing.assert_series_equal(actual, expected)
    assert actual.loc["positive"] > 0.0
    assert actual.loc["negative"] < 0.0
    assert actual.loc["weak"] == 0.0


def test_expected_alpha_rejects_unknown_estimator() -> None:
    factor_returns = pd.DataFrame({"size": [0.01]}, index=pd.to_datetime(["2024-01-31"]))

    with pytest.raises(ValueError, match="expected alpha estimator"):
        compute_expected_alpha(
            factor_returns,
            alpha_factor_names=["size"],
            sector_factor_names=[],
            window=1,
            estimator="unknown",
        )


def test_run_emp008_passes_explicit_unit_weights_to_every_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Emp008Config(factor_set=FactorSetId.SIZE_MOMENTUM_12_1M)
    dates = tuple(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"]))
    columns = ["A", "B"]
    close = pd.DataFrame([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], index=dates, columns=columns)
    benchmark_weights = pd.DataFrame([[0.5, 0.5]] * 3, index=dates, columns=columns)
    factor_frame = pd.DataFrame([[1.0, -1.0]] * 3, index=dates, columns=columns)
    prepared = PreparedEmp008Factors(
        config=config,
        market=object(),  # type: ignore[arg-type]
        factor_set_definition=get_factor_set_definition(config.factor_set),
        raw_factors={"ln_market_cap": factor_frame, "momentum_12_1m": factor_frame},
        alpha_factors={"ln_market_cap": factor_frame, "momentum_12_1m": factor_frame},
        sector_factors={},
        close=close,
        market_cap=close,
        float_market_cap=close,
        universe=close.notna(),
        sector=pd.DataFrame(index=dates, columns=columns),
        benchmark_weights=benchmark_weights,
        monthly_dates=dates,
    )
    seen: list[pd.Series] = []

    def fake_optimize_month(**kwargs: object) -> OptimizationResult:
        seen.append(cast(pd.Series, kwargs["factor_weights"]).copy())
        return OptimizationResult(
            success=True,
            final_weights=pd.Series({"A": 0.5, "B": 0.5}),
            active_weights=pd.Series({"A": 0.0, "B": 0.0}),
            objective_value=0.0,
            tracking_error=0.0,
            sector_active_exposure_abs_max=0.0,
        )

    monkeypatch.setattr(strategy, "_optimize_month", fake_optimize_month)

    strategy.run_emp008(
        parquet_dir=Path("parquet"),
        start="2024-02-29",
        end="2024-03-29",
        config=config,
        prepared=prepared,
        factor_weights={"ln_market_cap": 1.0, "momentum_12_1m": 1.0},
    )

    expected = pd.Series({"ln_market_cap": 1.0, "momentum_12_1m": 1.0}, dtype=float)
    assert len(seen) == 2
    for weights in seen:
        pd.testing.assert_series_equal(weights, expected)


def test_explicit_unit_weights_match_unweighted_size_momentum_pipeline() -> None:
    config = Emp008Config(factor_set=FactorSetId.SIZE_MOMENTUM_12_1M, risk_window=1)
    dates = tuple(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"]))
    columns = ["A", "B", "C"]
    close = pd.DataFrame(
        [[10.0, 20.0, 30.0], [11.0, 19.0, 31.0], [12.0, 18.0, 32.0]],
        index=dates,
        columns=columns,
    )
    benchmark_weights = pd.DataFrame([[1.0 / 3.0] * 3] * 3, index=dates, columns=columns)
    size = pd.DataFrame([[-1.0, 0.0, 1.0]] * 3, index=dates, columns=columns)
    momentum = pd.DataFrame([[1.0, -1.0, 0.0]] * 3, index=dates, columns=columns)
    prepared = PreparedEmp008Factors(
        config=config,
        market=object(),  # type: ignore[arg-type]
        factor_set_definition=get_factor_set_definition(config.factor_set),
        raw_factors={"ln_market_cap": size, "momentum_12_1m": momentum},
        alpha_factors={"ln_market_cap": size, "momentum_12_1m": momentum},
        sector_factors={},
        close=close,
        market_cap=close,
        float_market_cap=close,
        universe=close.notna(),
        sector=pd.DataFrame(index=dates, columns=columns),
        benchmark_weights=benchmark_weights,
        monthly_dates=dates,
    )
    baseline = strategy.run_emp008(
        parquet_dir=Path("parquet"),
        start="2024-02-29",
        end="2024-03-29",
        config=config,
        prepared=prepared,
    )
    explicit = strategy.run_emp008(
        parquet_dir=Path("parquet"),
        start="2024-02-29",
        end="2024-03-29",
        config=config,
        prepared=prepared,
        factor_weights={"ln_market_cap": 1.0, "momentum_12_1m": 1.0},
    )

    pd.testing.assert_frame_equal(explicit.target_weights, baseline.target_weights, check_exact=True)
    pd.testing.assert_frame_equal(explicit.active_weights, baseline.active_weights, check_exact=True)
    pd.testing.assert_frame_equal(explicit.diagnostics, baseline.diagnostics, check_exact=True)
