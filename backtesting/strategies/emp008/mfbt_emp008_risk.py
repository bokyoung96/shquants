from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CrossSectionalRegressionResult:
    factor_returns: pd.Series
    residuals: pd.Series


def fit_cross_sectional_factor_returns(
    exposures: pd.DataFrame,
    returns: pd.Series,
) -> CrossSectionalRegressionResult:
    common = exposures.index.intersection(returns.dropna().index)
    x = exposures.loc[common].dropna(how="all", axis=1).astype(float)
    y = returns.loc[common].astype(float)
    valid = x.notna().all(axis=1) & y.notna()
    x = x.loc[valid]
    y = y.loc[valid]
    if x.empty:
        raise ValueError("no valid regression observations")
    beta, *_ = np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)
    factor_returns = pd.Series(beta, index=x.columns, dtype=float)
    predicted = pd.Series(x.to_numpy() @ beta, index=x.index, dtype=float)
    residuals = y.sub(predicted)
    return CrossSectionalRegressionResult(factor_returns=factor_returns, residuals=residuals)


def factor_covariance(factor_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    recent = factor_returns.tail(window).astype(float)
    return recent.cov(ddof=0)


def residual_variance(residuals: pd.DataFrame, window: int) -> pd.Series:
    recent = residuals.tail(window).astype(float)
    return recent.pow(2).sum(axis=0).div(len(recent))


def compute_expected_alpha(
    factor_returns: pd.DataFrame,
    *,
    alpha_factor_names: list[str],
    sector_factor_names: list[str],
    window: int,
) -> pd.Series:
    recent = factor_returns.tail(window)
    alpha = recent.mean(axis=0).astype(float)
    for sector_name in sector_factor_names:
        alpha.loc[sector_name] = 0.0
    return alpha.reindex([*alpha_factor_names, *sector_factor_names]).fillna(0.0)
