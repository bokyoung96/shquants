import pandas as pd

from backtesting.strategies.emp008.mfbt_emp008_risk import (
    compute_expected_alpha,
    fit_cross_sectional_factor_returns,
    residual_variance,
)


def test_regression_returns_factor_returns_and_residuals() -> None:
    exposures = pd.DataFrame(
        {
            "alpha_1": [1.0, 0.0, -1.0],
            "alpha_2": [0.0, 1.0, -1.0],
        },
        index=["A", "B", "C"],
    )
    returns = pd.Series({"A": 0.02, "B": 0.01, "C": -0.03})

    result = fit_cross_sectional_factor_returns(exposures, returns)

    assert result.factor_returns.index.tolist() == ["alpha_1", "alpha_2"]
    assert result.residuals.index.tolist() == ["A", "B", "C"]
    assert abs(result.residuals.sum()) < 1e-10


def test_expected_alpha_sets_sector_factors_to_zero() -> None:
    factor_returns = pd.DataFrame(
        {
            "price_momentum": [0.01, 0.03],
            "G10": [0.20, -0.10],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    alpha = compute_expected_alpha(factor_returns, alpha_factor_names=["price_momentum"], sector_factor_names=["G10"], window=2)

    assert alpha["price_momentum"] == 0.02
    assert alpha["G10"] == 0.0


def test_residual_variance_uses_recent_squared_residuals() -> None:
    residuals = pd.DataFrame(
        {"A": [0.01, -0.01], "B": [0.02, 0.00]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    variance = residual_variance(residuals, window=2)

    assert variance["A"] == 0.0001
    assert variance["B"] == 0.0002
