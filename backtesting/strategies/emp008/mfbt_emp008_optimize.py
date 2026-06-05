from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
    common_tickers = exposures.index.intersection(residual_var.index).intersection(bm_weights.index)
    common_factors = exposures.columns.intersection(factor_cov.index).intersection(expected_alpha.index)
    z = exposures.loc[common_tickers, common_factors].astype(float)
    cov = factor_cov.loc[common_factors, common_factors].astype(float)
    resid = residual_var.loc[common_tickers].astype(float)
    alpha = expected_alpha.loc[common_factors].astype(float)
    bm = bm_weights.loc[common_tickers].astype(float)

    m = _full_covariance(z, cov, resid)
    z_array = z.to_numpy()
    alpha_array = alpha.to_numpy()
    bm_array = bm.to_numpy()

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x)},
        {"type": "ineq", "fun": lambda x: tracking_error**2 - float(x.T @ m @ x)},
    ]
    for sector_name in sector_factor_names:
        if sector_name in z.columns:
            sector_vector = z[sector_name].to_numpy()
            constraints.append({"type": "eq", "fun": lambda x, v=sector_vector: float(v @ x)})

    result = minimize(
        fun=lambda x: -float(alpha_array.T @ z_array.T @ x),
        x0=np.zeros(len(common_tickers)),
        method="SLSQP",
        bounds=[(-weight, None) for weight in bm_array],
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    active = pd.Series(result.x, index=common_tickers, dtype=float)
    final = bm.add(active, fill_value=0.0)
    sector_residual = 0.0
    if sector_factor_names:
        sector_cols = [name for name in sector_factor_names if name in z.columns]
        if sector_cols:
            sector_residual = float((z.loc[:, sector_cols].T @ active).abs().max())
    return OptimizationResult(
        success=bool(result.success),
        final_weights=final,
        active_weights=active,
        objective_value=-float(result.fun),
        tracking_error=float((active.to_numpy().T @ m @ active.to_numpy()) ** 0.5),
        sector_active_exposure_abs_max=sector_residual,
    )


def _full_covariance(exposures: pd.DataFrame, factor_cov: pd.DataFrame, residual_var: pd.Series) -> np.ndarray:
    z = exposures.to_numpy()
    cov = factor_cov.to_numpy()
    d = np.diag(residual_var.reindex(exposures.index).fillna(0.0).to_numpy())
    return d + z @ cov @ z.T
