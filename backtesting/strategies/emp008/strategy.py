from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Emp008Config
from .factor_pipeline import PreparedEmp008Factors, load_and_prepare_emp008_factors, validate_prepared_emp008_factors
from .factor_registry import FactorDirection, factor_definitions_for_set, get_factor_set_definition
from .optimize import OptimizationResult, optimize_active_weights, optimize_active_weights_with_covariance
from .preprocess import combine_exposures
from .risk import (
    compute_expected_alpha,
    factor_covariance,
    fit_cross_sectional_factor_returns,
    residual_variance,
)


@dataclass(frozen=True, slots=True)
class Emp008Result:
    target_weights: pd.DataFrame
    active_weights: pd.DataFrame
    diagnostics: pd.DataFrame

    def weights_for_export(self) -> pd.DataFrame:
        return self.target_weights.T

    def write_outputs(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.target_weights.to_parquet(output_dir / "target_weights.parquet", engine="pyarrow")
        self.active_weights.to_parquet(output_dir / "active_weights.parquet", engine="pyarrow")
        self.diagnostics.to_parquet(output_dir / "diagnostics.parquet", engine="pyarrow")
        with pd.ExcelWriter(output_dir / "weights_export.xlsx", engine="openpyxl") as writer:
            self.weights_for_export().to_excel(writer, sheet_name="weights_ticker_by_date")
            self.diagnostics.to_excel(writer, sheet_name="summary", index=False)
            self.active_weights.T.to_excel(writer, sheet_name="active_ticker_by_date")


def build_diagnostics_row(
    *,
    target_date: pd.Timestamp,
    result: OptimizationResult,
    alpha_factor_names: list[str],
    sector_factor_names: list[str],
    risk_model: str,
) -> dict[str, object]:
    return {
        "target_date": target_date,
        "success": result.success,
        "risk_model": risk_model,
        "objective_value": result.objective_value,
        "tracking_error": result.tracking_error,
        "n_active_positions": int(result.active_weights.abs().gt(0.001).sum()),
        "max_weight": float(result.final_weights.max()),
        "min_weight": float(result.final_weights.min()),
        "sum_final_weight": float(result.final_weights.sum()),
        "sum_active_weight": float(result.active_weights.sum()),
        "sector_active_exposure_abs_max": result.sector_active_exposure_abs_max,
        "alpha_factor_names": tuple(alpha_factor_names),
        "sector_factor_names": tuple(sector_factor_names),
    }


def run_emp008(
    *,
    parquet_dir: Path,
    start: str,
    end: str,
    config: Emp008Config | None = None,
    output_dir: Path | None = None,
    prepared: PreparedEmp008Factors | None = None,
) -> Emp008Result:
    requested_config = config or Emp008Config()
    if prepared is not None:
        validate_prepared_emp008_factors(prepared, config=requested_config, start=start, end=end)
    active_config = prepared.config if prepared is not None else requested_config
    factor_bundle = prepared or load_and_prepare_emp008_factors(parquet_dir, start, end, active_config)
    alpha_factor_names = list(factor_bundle.raw_factors)
    sector_factor_names = list(factor_bundle.sector_factors)

    factor_return_rows: list[pd.Series] = []
    residual_rows: list[pd.Series] = []
    factor_return_dates: list[pd.Timestamp] = []
    stock_excess_return_rows: list[pd.Series] = []
    stock_excess_return_dates: list[pd.Timestamp] = []
    target_rows: list[pd.Series] = []
    active_rows: list[pd.Series] = []
    diagnostics: list[dict[str, object]] = []

    for idx in range(1, len(factor_bundle.monthly_dates)):
        factor_date = factor_bundle.monthly_dates[idx - 1]
        return_date = factor_bundle.monthly_dates[idx]
        if return_date > pd.Timestamp(end):
            break
        should_output = return_date >= pd.Timestamp(start)
        try:
            optimization = _optimize_month(
                close=factor_bundle.close,
                bm_weights=factor_bundle.benchmark_weights,
                alpha_factors=factor_bundle.alpha_factors,
                sector_factors=factor_bundle.sector_factors,
                factor_date=factor_date,
                return_date=return_date,
                factor_return_rows=factor_return_rows,
                residual_rows=residual_rows,
                factor_return_dates=factor_return_dates,
                stock_excess_return_rows=stock_excess_return_rows,
                stock_excess_return_dates=stock_excess_return_dates,
                alpha_factor_names=alpha_factor_names,
                sector_factor_names=sector_factor_names,
                config=active_config,
                run_optimization=should_output,
            )
        except (ValueError, KeyError):
            if should_output:
                raise
            continue
        if not should_output or optimization is None:
            continue
        optimization = _validated_optimization(return_date, optimization)
        target_rows.append(optimization.final_weights.rename(return_date))
        active_rows.append(optimization.active_weights.rename(return_date))
        diagnostics.append(
            build_diagnostics_row(
                target_date=return_date,
                result=optimization,
                alpha_factor_names=alpha_factor_names,
                sector_factor_names=sector_factor_names,
                risk_model=active_config.risk_model,
            )
        )

    result = Emp008Result(
        target_weights=pd.DataFrame(target_rows).fillna(0.0),
        active_weights=pd.DataFrame(active_rows).fillna(0.0),
        diagnostics=pd.DataFrame(diagnostics),
    )
    if output_dir is not None:
        result.write_outputs(output_dir)
    return result


def _optimize_month(
    *,
    close: pd.DataFrame,
    bm_weights: pd.DataFrame,
    alpha_factors: dict[str, pd.DataFrame],
    sector_factors: dict[str, pd.DataFrame],
    factor_date: pd.Timestamp,
    return_date: pd.Timestamp,
    factor_return_rows: list[pd.Series],
    residual_rows: list[pd.Series],
    factor_return_dates: list[pd.Timestamp],
    stock_excess_return_rows: list[pd.Series],
    stock_excess_return_dates: list[pd.Timestamp],
    alpha_factor_names: list[str],
    sector_factor_names: list[str],
    config: Emp008Config,
    run_optimization: bool,
) -> OptimizationResult | None:
    exposures = combine_exposures(alpha_factors, sector_factors, factor_date)
    stock_returns = close.loc[return_date].divide(close.loc[factor_date]).sub(1.0)
    bm = _positive_benchmark_weights(bm_weights.reindex(index=[return_date], columns=stock_returns.index).iloc[0])
    excess_returns = stock_returns.sub(stock_returns.reindex(bm.index).mul(bm).sum())
    regression = fit_cross_sectional_factor_returns(exposures, excess_returns)
    factor_return_rows.append(regression.factor_returns)
    residual_rows.append(regression.residuals)
    factor_return_dates.append(return_date)
    stock_excess_return_rows.append(excess_returns)
    stock_excess_return_dates.append(return_date)

    factor_returns = pd.DataFrame(factor_return_rows, index=factor_return_dates).fillna(0.0)
    residuals = pd.DataFrame(residual_rows, index=factor_return_dates)
    stock_excess_returns = pd.DataFrame(stock_excess_return_rows, index=stock_excess_return_dates)
    if not _has_sufficient_risk_history(factor_returns, config) or not run_optimization:
        return None

    expected_alpha = compute_expected_alpha(
        factor_returns,
        alpha_factor_names=alpha_factor_names,
        sector_factor_names=sector_factor_names,
        window=config.risk_window,
    )
    expected_alpha = _apply_expected_alpha_policy(expected_alpha, config)
    target_exposures = combine_exposures(alpha_factors, sector_factors, return_date)
    target_bm = _positive_benchmark_weights(
        bm_weights.reindex(index=[return_date], columns=target_exposures.index).iloc[0]
    )
    if config.risk_model == "factor_idio":
        factor_cov = factor_covariance(factor_returns, config.risk_window)
        resid_var = _residual_variance_for_target_universe(
            residual_variance(residuals, config.risk_window),
            target_bm.index,
        )
        return optimize_active_weights(
            exposures=target_exposures,
            factor_cov=factor_cov,
            residual_var=resid_var,
            expected_alpha=expected_alpha,
            bm_weights=target_bm,
            sector_factor_names=sector_factor_names,
            tracking_error=config.tracking_error,
        )
    if config.risk_model == "direct_covariance":
        stock_cov = _stock_excess_covariance_for_target_universe(
            stock_excess_returns,
            target_tickers=target_bm.index,
            window=config.risk_window,
        )
        return optimize_active_weights_with_covariance(
            exposures=target_exposures,
            stock_cov=stock_cov,
            expected_alpha=expected_alpha,
            bm_weights=target_bm,
            sector_factor_names=sector_factor_names,
            tracking_error=config.tracking_error,
        )
    raise ValueError(f"unsupported risk_model: {config.risk_model}")


def _validated_optimization(target_date: pd.Timestamp, result: OptimizationResult) -> OptimizationResult:
    if not result.success:
        raise RuntimeError(f"optimization failed for {target_date:%Y-%m-%d}")
    return result


def _has_sufficient_risk_history(factor_returns: pd.DataFrame, config: Emp008Config) -> bool:
    return len(factor_returns) >= config.risk_window


def _apply_expected_alpha_policy(expected_alpha: pd.Series, config: Emp008Config) -> pd.Series:
    factor_set_definition = get_factor_set_definition(config.factor_set)
    if not factor_set_definition.constrain_expected_alpha_to_direction:
        return expected_alpha

    adjusted = expected_alpha.copy()
    for definition in factor_definitions_for_set(config.factor_set):
        factor_name = definition.id.value
        if factor_name not in adjusted:
            continue
        if definition.direction is FactorDirection.HIGH and adjusted.loc[factor_name] < 0.0:
            adjusted.loc[factor_name] = 0.0
        if definition.direction is FactorDirection.LOW and adjusted.loc[factor_name] > 0.0:
            adjusted.loc[factor_name] = 0.0
    return adjusted


def _positive_benchmark_weights(weights: pd.Series) -> pd.Series:
    positive = weights.astype(float).fillna(0.0)
    positive = positive.loc[positive.gt(0.0)]
    total = positive.sum()
    if total <= 0.0:
        raise ValueError("no positive benchmark weights")
    return positive.div(total)


def _residual_variance_for_target_universe(residual_var: pd.Series, target_tickers: pd.Index) -> pd.Series:
    aligned = residual_var.reindex(target_tickers).astype(float)
    fallback = aligned.dropna().median()
    if pd.isna(fallback):
        fallback = 0.0
    return aligned.fillna(float(fallback))


def _stock_excess_covariance_for_target_universe(
    stock_excess_returns: pd.DataFrame,
    *,
    target_tickers: pd.Index,
    window: int,
) -> pd.DataFrame:
    recent = stock_excess_returns.tail(window).reindex(columns=target_tickers).astype(float).fillna(0.0)
    cov = recent.cov(ddof=0).reindex(index=target_tickers, columns=target_tickers).astype(float)
    diag = pd.Series(np.diag(cov.to_numpy()), index=target_tickers)
    fallback = diag.where(diag.gt(0.0)).dropna().median()
    if pd.isna(fallback):
        fallback = 0.0
    cov = cov.fillna(0.0)
    for ticker in target_tickers:
        if cov.at[ticker, ticker] <= 0.0:
            cov.at[ticker, ticker] = float(fallback)
    return cov
