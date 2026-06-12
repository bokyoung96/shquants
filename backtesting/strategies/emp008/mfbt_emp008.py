from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .mfbt_emp008_data import MfbtEmp008Config, load_mfbt_emp008_bm_weights, load_mfbt_emp008_market
from .mfbt_emp008_factors import build_raw_mfbt_factors
from .mfbt_emp008_optimize import OptimizationResult, optimize_active_weights, optimize_active_weights_with_covariance
from .mfbt_emp008_preprocess import build_sector_active_exposures, combine_exposures, preprocess_factor_frame
from .mfbt_emp008_risk import (
    compute_expected_alpha,
    factor_covariance,
    fit_cross_sectional_factor_returns,
    residual_variance,
)


@dataclass(frozen=True, slots=True)
class MfbtEmp008Result:
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


def run_mfbt_emp008_smoke(*, parquet_dir: Path, start: str, end: str) -> MfbtEmp008Result:
    config = MfbtEmp008Config()
    bm = load_mfbt_emp008_bm_weights(parquet_dir=parquet_dir, start=start, end=end, config=config).astype(float).copy()
    row_sum = bm.sum(axis=1)
    usable = row_sum.gt(0.0)
    if not usable.any():
        raise ValueError("no usable benchmark weight rows")
    bm = bm.loc[usable].div(row_sum.loc[usable], axis=0)
    diagnostics = pd.DataFrame(
        {
            "target_date": bm.index,
            "success": True,
            "sum_final_weight": bm.sum(axis=1).to_numpy(),
            "n_active_positions": bm.gt(0.0).sum(axis=1).to_numpy(),
        }
    )
    return MfbtEmp008Result(target_weights=bm, active_weights=bm * 0.0, diagnostics=diagnostics)


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


def run_mfbt_emp008(
    *,
    parquet_dir: Path,
    start: str,
    end: str,
    config: MfbtEmp008Config | None = None,
    output_dir: Path | None = None,
) -> MfbtEmp008Result:
    active_config = config or MfbtEmp008Config()
    market = load_mfbt_emp008_market(parquet_dir=parquet_dir, start=start, end=end, config=active_config)
    raw_factors = build_raw_mfbt_factors(market, active_config)
    alpha_factor_names = list(raw_factors)

    close = market.frames["close"].astype(float)
    float_mktcap = market.frames["float_market_cap"].reindex(index=close.index, columns=close.columns).astype(float)
    universe = market.frames["k200_yn"].reindex(index=close.index, columns=close.columns).fillna(0).astype(bool)
    sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
    bm_weights = market.frames["bm_weights"].reindex(index=close.index, columns=close.columns).astype(float)

    monthly_dates = _common_month_end_dates(raw_factors)
    alpha_factors = {
        name: preprocess_factor_frame(
            frame,
            float_mktcap,
            universe,
            rank_transform=name in active_config.rank_transform_factors,
        )
        for name, frame in raw_factors.items()
    }
    alpha_factors = _neutralize_large_benchmark_weight_factor_exposures(
        alpha_factors,
        bm_weights,
        active_config,
    )
    sector_factors = build_sector_active_exposures(sector, float_mktcap, universe)
    sector_factor_names = list(sector_factors)

    factor_return_rows: list[pd.Series] = []
    residual_rows: list[pd.Series] = []
    factor_return_dates: list[pd.Timestamp] = []
    stock_excess_return_rows: list[pd.Series] = []
    stock_excess_return_dates: list[pd.Timestamp] = []
    target_rows: list[pd.Series] = []
    active_rows: list[pd.Series] = []
    diagnostics: list[dict[str, object]] = []

    for idx in range(1, len(monthly_dates)):
        factor_date = monthly_dates[idx - 1]
        return_date = monthly_dates[idx]
        if return_date > pd.Timestamp(end):
            break
        should_output = return_date >= pd.Timestamp(start)
        try:
            optimization = _optimize_month(
                close=close,
                bm_weights=bm_weights,
                alpha_factors=alpha_factors,
                sector_factors=sector_factors,
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

    result = MfbtEmp008Result(
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
    config: MfbtEmp008Config,
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


def _common_month_end_dates(factors: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    non_empty = [set(frame.dropna(how="all").index) for frame in factors.values()]
    if not non_empty:
        return []
    return sorted(set.intersection(*non_empty))


def _validated_optimization(target_date: pd.Timestamp, result: OptimizationResult) -> OptimizationResult:
    if not result.success:
        raise RuntimeError(f"optimization failed for {target_date:%Y-%m-%d}")
    return result


def _has_sufficient_risk_history(factor_returns: pd.DataFrame, config: MfbtEmp008Config) -> bool:
    return len(factor_returns) >= config.risk_window


def _apply_expected_alpha_policy(expected_alpha: pd.Series, config: MfbtEmp008Config) -> pd.Series:
    if config.expected_alpha_policy == "mean":
        return expected_alpha
    if config.expected_alpha_policy != "origin_sign":
        raise ValueError(f"unsupported expected_alpha_policy: {config.expected_alpha_policy}")

    adjusted = expected_alpha.copy()
    for factor in ("DY", "Momentum_12M"):
        if factor in adjusted and adjusted.loc[factor] < 0.0:
            adjusted.loc[factor] = 0.0
    if "LnMktcap" in adjusted and adjusted.loc["LnMktcap"] > 0.0:
        adjusted.loc["LnMktcap"] = 0.0
    return adjusted


def _positive_benchmark_weights(weights: pd.Series) -> pd.Series:
    positive = weights.astype(float).fillna(0.0)
    positive = positive.loc[positive.gt(0.0)]
    total = positive.sum()
    if total <= 0.0:
        raise ValueError("no positive benchmark weights")
    return positive.div(total)


def _neutralize_large_benchmark_weight_factor_exposures(
    alpha_factors: dict[str, pd.DataFrame],
    bm_weights: pd.DataFrame,
    config: MfbtEmp008Config,
) -> dict[str, pd.DataFrame]:
    threshold = config.large_bm_neutral_weight_threshold
    if threshold <= 0.0 or not config.large_bm_neutral_factor_names:
        return alpha_factors

    neutralized = dict(alpha_factors)
    for name in config.large_bm_neutral_factor_names:
        if name not in neutralized:
            continue
        frame = neutralized[name]
        large_bm = bm_weights.reindex(index=frame.index, columns=frame.columns).fillna(0.0).ge(threshold)
        neutralized[name] = frame.mask(large_bm, 0.0)
    return neutralized


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
