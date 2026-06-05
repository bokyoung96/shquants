from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .mfbt_emp008_data import MfbtEmp008Config, load_mfbt_emp008_bm_weights, load_mfbt_emp008_market
from .mfbt_emp008_factors import build_raw_mfbt_factors
from .mfbt_emp008_optimize import OptimizationResult, optimize_active_weights
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
) -> dict[str, object]:
    return {
        "target_date": target_date,
        "success": result.success,
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
        name: preprocess_factor_frame(frame, float_mktcap, universe)
        for name, frame in raw_factors.items()
    }
    sector_factors = build_sector_active_exposures(sector, float_mktcap, universe)
    sector_factor_names = list(sector_factors)

    factor_return_rows: list[pd.Series] = []
    residual_rows: list[pd.Series] = []
    factor_return_dates: list[pd.Timestamp] = []
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
                alpha_factor_names=alpha_factor_names,
                sector_factor_names=sector_factor_names,
                config=active_config,
                run_optimization=should_output,
            )
        except (ValueError, KeyError):
            continue
        if not should_output or optimization is None:
            continue
        target_rows.append(optimization.final_weights.rename(return_date))
        active_rows.append(optimization.active_weights.rename(return_date))
        diagnostics.append(
            build_diagnostics_row(
                target_date=return_date,
                result=optimization,
                alpha_factor_names=alpha_factor_names,
                sector_factor_names=sector_factor_names,
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

    factor_returns = pd.DataFrame(factor_return_rows, index=factor_return_dates).fillna(0.0)
    residuals = pd.DataFrame(residual_rows, index=factor_return_dates)
    if len(factor_returns) < 2 or not run_optimization:
        return None

    expected_alpha = compute_expected_alpha(
        factor_returns,
        alpha_factor_names=alpha_factor_names,
        sector_factor_names=sector_factor_names,
        window=config.risk_window,
    )
    factor_cov = factor_covariance(factor_returns, config.risk_window)
    resid_var = residual_variance(residuals, config.risk_window).fillna(0.0)
    target_exposures = combine_exposures(alpha_factors, sector_factors, return_date)
    target_bm = _positive_benchmark_weights(
        bm_weights.reindex(index=[return_date], columns=target_exposures.index).iloc[0]
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


def _common_month_end_dates(factors: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    non_empty = [set(frame.dropna(how="all").index) for frame in factors.values()]
    if not non_empty:
        return []
    return sorted(set.intersection(*non_empty))


def _positive_benchmark_weights(weights: pd.Series) -> pd.Series:
    positive = weights.astype(float).fillna(0.0)
    positive = positive.loc[positive.gt(0.0)]
    total = positive.sum()
    if total <= 0.0:
        raise ValueError("no positive benchmark weights")
    return positive.div(total)
