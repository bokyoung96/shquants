import pandas as pd
import pytest

from backtesting.strategies.emp008.mfbt_emp008_optimize import optimize_active_weights


def _optimizer_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    tickers = ["A", "B", "C"]
    exposures = pd.DataFrame(
        {
            "price_momentum": [1.0, 0.0, -1.0],
            "G10": [0.5, 0.5, -1.0],
        },
        index=tickers,
    )
    factor_cov = pd.DataFrame(
        [[0.01, 0.0], [0.0, 0.01]],
        index=["price_momentum", "G10"],
        columns=["price_momentum", "G10"],
    )
    residual_var = pd.Series({"A": 0.01, "B": 0.01, "C": 0.01})
    alpha = pd.Series({"price_momentum": 0.01, "G10": 0.0})
    bm = pd.Series({"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0})
    return exposures, factor_cov, residual_var, alpha, bm


def test_optimizer_returns_long_only_benchmark_relative_weights() -> None:
    exposures, factor_cov, residual_var, alpha, bm = _optimizer_inputs()

    result = optimize_active_weights(
        exposures=exposures,
        factor_cov=factor_cov,
        residual_var=residual_var,
        expected_alpha=alpha,
        bm_weights=bm,
        sector_factor_names=["G10"],
        tracking_error=0.20,
    )

    assert result.success
    assert abs(result.final_weights.sum() - 1.0) < 1e-8
    assert abs(result.active_weights.sum()) < 1e-8
    assert result.final_weights.ge(-1e-10).all()
    assert abs(exposures["G10"].dot(result.active_weights)) < 1e-7
    assert result.tracking_error <= 0.20 + 1e-8
    assert result.sector_active_exposure_abs_max < 1e-7


def test_optimizer_keeps_sector_constraint_when_sector_alpha_missing() -> None:
    exposures, factor_cov, residual_var, alpha, bm = _optimizer_inputs()
    alpha = alpha.drop(index="G10")

    result = optimize_active_weights(
        exposures=exposures,
        factor_cov=factor_cov,
        residual_var=residual_var,
        expected_alpha=alpha,
        bm_weights=bm,
        sector_factor_names=["G10"],
        tracking_error=0.20,
    )

    assert result.success
    assert abs(exposures["G10"].dot(result.active_weights)) < 1e-7
    assert result.sector_active_exposure_abs_max < 1e-7


def test_optimizer_rejects_missing_benchmark_constituent_inputs() -> None:
    exposures, factor_cov, residual_var, alpha, bm = _optimizer_inputs()
    residual_var = residual_var.drop(index="C")

    with pytest.raises(ValueError, match="missing optimizer inputs"):
        optimize_active_weights(
            exposures=exposures,
            factor_cov=factor_cov,
            residual_var=residual_var,
            expected_alpha=alpha,
            bm_weights=bm,
            sector_factor_names=["G10"],
            tracking_error=0.20,
        )
