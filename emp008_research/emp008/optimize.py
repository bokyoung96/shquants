from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


MIN_FINAL_WEIGHT = 1e-8


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    success: bool
    final_weights: pd.Series
    active_weights: pd.Series
    objective_value: float
    tracking_error: float
    sector_active_exposure_abs_max: float


def optimize_active_weights(
    *,
    exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    residual_var: pd.Series,
    expected_alpha: pd.Series,
    bm_weights: pd.Series,
    sector_factor_names: list[str],
    tracking_error: float,
) -> OptimizationResult:
    common_tickers = bm_weights.index
    missing_exposures = common_tickers.difference(exposures.index)
    missing_residuals = common_tickers.difference(residual_var.index)
    if len(missing_exposures) > 0 or len(missing_residuals) > 0:
        missing = sorted({*(str(name) for name in missing_exposures), *(str(name) for name in missing_residuals)})
        raise ValueError(f"missing optimizer inputs for benchmark constituents: {', '.join(missing)}")

    common_factors = exposures.columns.intersection(factor_cov.index).intersection(factor_cov.columns)
    missing_sector_factors = [name for name in sector_factor_names if name not in common_factors]
    if missing_sector_factors:
        raise ValueError(f"missing sector factors: {', '.join(missing_sector_factors)}")

    z = exposures.loc[common_tickers, common_factors].astype(float)
    cov = factor_cov.loc[common_factors, common_factors].astype(float)
    resid = residual_var.loc[common_tickers].astype(float)
    alpha = expected_alpha.reindex(common_factors).fillna(0.0).astype(float)
    bm = bm_weights.loc[common_tickers].astype(float)

    m = _full_covariance(z, cov, resid)
    return _solve_active_weight_problem(
        exposures=z,
        covariance=m,
        expected_alpha=alpha,
        bm_weights=bm,
        sector_factor_names=sector_factor_names,
        tracking_error=tracking_error,
    )


def _solve_active_weight_problem(
    *,
    exposures: pd.DataFrame,
    covariance: np.ndarray,
    expected_alpha: pd.Series,
    bm_weights: pd.Series,
    sector_factor_names: list[str],
    tracking_error: float,
) -> OptimizationResult:
    z_array = exposures.to_numpy()
    alpha_array = expected_alpha.to_numpy()
    bm_array = bm_weights.to_numpy()
    final_weight_floor = np.minimum(bm_array * 0.5, MIN_FINAL_WEIGHT)
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x)},
        {"type": "ineq", "fun": lambda x: tracking_error**2 - float(x.T @ covariance @ x)},
    ]
    for sector_name in _independent_sector_factor_names(exposures, sector_factor_names):
        sector_vector = exposures[sector_name].to_numpy()
        constraints.append({"type": "eq", "fun": lambda x, v=sector_vector: float(v @ x)})

    minimize_kwargs = {
        "fun": lambda x: -float(alpha_array.T @ z_array.T @ x),
        "x0": np.zeros(len(exposures)),
        "method": "SLSQP",
        "bounds": [(-weight + floor, None) for weight, floor in zip(bm_array, final_weight_floor, strict=True)],
    }
    result = minimize(
        **minimize_kwargs,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        result = minimize(
            **minimize_kwargs,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
    active = pd.Series(result.x, index=exposures.index, dtype=float)
    final = bm_weights.add(active, fill_value=0.0)
    sector_residual = 0.0
    if sector_factor_names:
        sector_cols = [name for name in sector_factor_names if name in exposures.columns]
        if sector_cols:
            sector_residual = float((exposures.loc[:, sector_cols].T @ active).abs().max())
    return OptimizationResult(
        success=bool(result.success),
        final_weights=final,
        active_weights=active,
        objective_value=-float(result.fun),
        tracking_error=float((active.to_numpy().T @ covariance @ active.to_numpy()) ** 0.5),
        sector_active_exposure_abs_max=sector_residual,
    )


def _independent_sector_factor_names(
    exposures: pd.DataFrame,
    sector_factor_names: list[str],
) -> list[str]:
    basis = np.ones((1, len(exposures)), dtype=float)
    rank = int(np.linalg.matrix_rank(basis))
    independent: list[str] = []
    for name in sector_factor_names:
        if name not in exposures.columns:
            continue
        candidate = np.vstack([basis, exposures[name].to_numpy(dtype=float)])
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank <= rank:
            continue
        independent.append(name)
        basis = candidate
        rank = candidate_rank
    return independent


def _full_covariance(exposures: pd.DataFrame, factor_cov: pd.DataFrame, residual_var: pd.Series) -> np.ndarray:
    z = exposures.to_numpy()
    cov = factor_cov.to_numpy()
    d = np.diag(residual_var.reindex(exposures.index).fillna(0.0).to_numpy())
    return d + z @ cov @ z.T
