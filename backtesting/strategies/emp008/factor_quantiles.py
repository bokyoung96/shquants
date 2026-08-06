from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.testing import assert_frame_equal

from .factor_pipeline import PreparedEmp008Factors
from .factor_registry import FactorDirection, FactorSetId, factor_definitions_for_set


class Emp008FactorQuantilesUnavailableError(ValueError):
    """Raised when quantile diagnostics are unavailable for an otherwise valid run."""


@unique
class QuantileWeighting(StrEnum):
    EQUAL = "equal_weight"
    MARKET_CAP = "market_cap_weight"


@dataclass(frozen=True, slots=True)
class Emp008FactorQuantileResult:
    monthly_returns: pd.DataFrame
    portfolio_weights: pd.DataFrame
    rank_ic: pd.DataFrame
    cumulative_returns: pd.DataFrame
    daily_cumulative_returns: pd.DataFrame
    summary: pd.DataFrame

    def write_outputs(
        self,
        output_dir: Path | str,
        *,
        factor_set: FactorSetId | str,
        q: int,
    ) -> dict[str, object]:
        destination = Path(output_dir)
        _validate_result_for_output(self, factor_set=factor_set, q=q)
        manifest = _build_manifest(self, factor_set=factor_set, q=q, output_dir=destination)

        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as tmp_dir:
            staging_root = Path(tmp_dir)
            staging_dir = staging_root / "artifacts"
            staging_dir.mkdir()
            monthly_returns_csv = staging_dir / "monthly_returns.csv"
            monthly_returns_parquet = staging_dir / "monthly_returns.parquet"
            portfolio_weights_parquet = staging_dir / "portfolio_weights.parquet"
            rank_ic_csv = staging_dir / "rank_ic.csv"
            rank_ic_parquet = staging_dir / "rank_ic.parquet"
            cumulative_returns_csv = staging_dir / "cumulative_returns.csv"
            cumulative_quintiles_equal_weight_png = staging_dir / "cumulative_quintiles_equal_weight.png"
            cumulative_quintiles_market_cap_weight_png = staging_dir / "cumulative_quintiles_market_cap_weight.png"
            summary_csv = staging_dir / "summary.csv"
            summary_json = staging_dir / "summary.json"
            manifest_json = staging_dir / "manifest.json"

            self.monthly_returns.to_csv(monthly_returns_csv, index=False)
            self.monthly_returns.to_parquet(monthly_returns_parquet, engine="pyarrow", index=False)
            self.portfolio_weights.to_parquet(portfolio_weights_parquet, engine="pyarrow", index=False)
            self.rank_ic.to_csv(rank_ic_csv, index=False)
            self.rank_ic.to_parquet(rank_ic_parquet, engine="pyarrow", index=False)
            self.cumulative_returns.to_csv(cumulative_returns_csv, index=False)
            directions = {
                definition.id.value: definition.direction
                for definition in factor_definitions_for_set(factor_set)
            }
            _write_cumulative_quintile_plot(
                path=cumulative_quintiles_equal_weight_png,
                cumulative_returns=self.cumulative_returns,
                directions=directions,
                weighting=QuantileWeighting.EQUAL,
                q=q,
            )
            _write_cumulative_quintile_plot(
                path=cumulative_quintiles_market_cap_weight_png,
                cumulative_returns=self.cumulative_returns,
                directions=directions,
                weighting=QuantileWeighting.MARKET_CAP,
                q=q,
            )
            self.summary.to_csv(summary_csv, index=False)
            summary_json.write_text(
                json.dumps(_json_safe_records(self.summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

            artifact_paths = [
                monthly_returns_csv,
                monthly_returns_parquet,
                portfolio_weights_parquet,
                rank_ic_csv,
                rank_ic_parquet,
                cumulative_returns_csv,
                cumulative_quintiles_equal_weight_png,
                cumulative_quintiles_market_cap_weight_png,
                summary_csv,
                summary_json,
                manifest_json,
            ]
            _publish_artifacts_atomically(artifact_paths=artifact_paths, destination=destination, staging_root=staging_root)

        return {
            "monthly_returns_csv": str(destination / "monthly_returns.csv"),
            "monthly_returns_parquet": str(destination / "monthly_returns.parquet"),
            "portfolio_weights_parquet": str(destination / "portfolio_weights.parquet"),
            "rank_ic_csv": str(destination / "rank_ic.csv"),
            "rank_ic_parquet": str(destination / "rank_ic.parquet"),
            "cumulative_returns_csv": str(destination / "cumulative_returns.csv"),
            "cumulative_quintiles_equal_weight_png": str(destination / "cumulative_quintiles_equal_weight.png"),
            "cumulative_quintiles_market_cap_weight_png": str(destination / "cumulative_quintiles_market_cap_weight.png"),
            "summary_csv": str(destination / "summary.csv"),
            "summary_json": str(destination / "summary.json"),
            "manifest_json": str(destination / "manifest.json"),
            "monthly_returns_rows": int(len(self.monthly_returns)),
            "weights_rows": int(len(self.portfolio_weights)),
            "rank_ic_rows": int(len(self.rank_ic)),
            "cumulative_returns_rows": int(len(self.cumulative_returns)),
            "summary_rows": int(len(self.summary)),
        }


_MONTHLY_RETURNS_COLUMNS = (
    "signal_date",
    "return_date",
    "factor",
    "weighting",
    "portfolio",
    "return",
    "constituent_count",
)
_PORTFOLIO_WEIGHTS_COLUMNS = (
    "signal_date",
    "return_date",
    "factor",
    "weighting",
    "quantile",
    "ticker",
    "weight",
)
_RANK_IC_COLUMNS = (
    "signal_date",
    "return_date",
    "factor",
    "rank_ic",
    "directional_rank_ic",
    "n_obs",
)
_CUMULATIVE_RETURNS_COLUMNS = (
    "signal_date",
    "return_date",
    "factor",
    "weighting",
    "portfolio",
    "cumulative_return",
)
_DAILY_CUMULATIVE_RETURNS_COLUMNS = (
    "signal_date",
    "date",
    "factor",
    "weighting",
    "portfolio",
    "cumulative_return",
)
_SUMMARY_COLUMNS = (
    "factor",
    "weighting",
    "portfolio",
    "observations",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "positive_month_rate",
    "mean_monthly_return",
    "average_constituent_count",
    "average_one_way_turnover",
    "mean_rank_ic",
    "directional_mean_rank_ic",
    "ic_information_ratio",
    "ic_positive_rate",
    "quantile_monotonicity",
)
_ARTIFACT_FILENAMES = (
    "monthly_returns.csv",
    "monthly_returns.parquet",
    "portfolio_weights.parquet",
    "rank_ic.csv",
    "rank_ic.parquet",
    "cumulative_returns.csv",
    "cumulative_quintiles_equal_weight.png",
    "cumulative_quintiles_market_cap_weight.png",
    "summary.csv",
    "summary.json",
    "manifest.json",
)


def run_emp008_factor_quantiles(
    *,
    prepared: PreparedEmp008Factors,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    q: int = 5,
) -> Emp008FactorQuantileResult:
    directions = {
        definition.id.value: definition.direction
        for definition in factor_definitions_for_set(prepared.factor_set_definition.id)
    }
    try:
        return evaluate_factor_quantiles(
            factors=prepared.alpha_factors,
            directions=directions,
            close=prepared.close,
            market_cap=prepared.market_cap,
            universe=prepared.universe,
            monthly_dates=prepared.monthly_dates,
            start=start,
            end=end,
            q=q,
        )
    except Emp008FactorQuantilesUnavailableError as exc:
        start_ts = pd.Timestamp(start).date().isoformat()
        end_ts = pd.Timestamp(end).date().isoformat()
        raise Emp008FactorQuantilesUnavailableError(
            f"no factor quantile observations for {prepared.factor_set_definition.id.value} "
            f"in requested range {start_ts} to {end_ts}"
        ) from exc


def evaluate_factor_quantiles(
    *,
    factors: Mapping[str, pd.DataFrame],
    directions: Mapping[str, FactorDirection],
    close: pd.DataFrame,
    market_cap: pd.DataFrame,
    universe: pd.DataFrame,
    monthly_dates: Sequence[pd.Timestamp],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    q: int,
) -> Emp008FactorQuantileResult:
    if q < 2:
        raise ValueError("q must be at least 2")

    if not factors:
        raise ValueError("at least one factor is required")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start must be on or before end")

    normalized_monthly_dates = tuple(pd.Timestamp(date) for date in monthly_dates)
    if len(normalized_monthly_dates) < 2:
        raise Emp008FactorQuantilesUnavailableError("at least two monthly dates are required")
    if len(set(normalized_monthly_dates)) != len(normalized_monthly_dates):
        raise ValueError("duplicate monthly dates are not allowed")
    if any(left >= right for left, right in zip(normalized_monthly_dates[:-1], normalized_monthly_dates[1:])):
        raise ValueError("monthly dates must be strictly increasing")

    factor_names = list(factors)
    missing_directions = sorted(name for name in factor_names if name not in directions)
    if missing_directions:
        missing_text = ", ".join(missing_directions)
        raise ValueError(f"missing directions for factor(s): {missing_text}")
    normalized_directions = {name: FactorDirection(directions[name]) for name in factor_names}

    _validate_frame_axes(
        close=close,
        market_cap=market_cap,
        universe=universe,
        factors=factors,
    )
    aligned_close = close.astype(float)
    aligned_market_cap = market_cap.reindex(index=aligned_close.index, columns=aligned_close.columns).astype(float)
    aligned_universe = universe.reindex(index=aligned_close.index, columns=aligned_close.columns).fillna(False).astype(bool)
    aligned_factors = {
        name: frame.reindex(index=aligned_close.index, columns=aligned_close.columns).astype(float)
        for name, frame in factors.items()
    }

    monthly_return_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    rank_ic_rows: list[dict[str, object]] = []

    for signal_date, return_date in zip(normalized_monthly_dates[:-1], normalized_monthly_dates[1:]):
        if return_date < start_ts or return_date > end_ts:
            continue
        signal_prices = aligned_close.loc[signal_date]
        return_prices = aligned_close.loc[return_date]
        signal_market_cap = aligned_market_cap.loc[signal_date]
        signal_universe = aligned_universe.loc[signal_date]
        for factor_name, frame in aligned_factors.items():
            direction = normalized_directions[factor_name]
            signal_values = frame.loc[signal_date]
            base_eligibility = (
                signal_universe
                & signal_values.map(np.isfinite)
                & signal_prices.map(np.isfinite)
                & return_prices.map(np.isfinite)
                & signal_prices.gt(0.0)
                & signal_market_cap.map(np.isfinite)
                & signal_market_cap.gt(0.0)
            )
            eligible_names = signal_values.index[base_eligibility]
            if len(eligible_names) == 0:
                continue

            eligible_signals = signal_values.loc[eligible_names]
            eligible_signal_prices = signal_prices.loc[eligible_names]
            eligible_return_prices = return_prices.loc[eligible_names]
            next_returns = eligible_return_prices.divide(eligible_signal_prices).sub(1.0)
            memberships = _build_quantile_memberships(eligible_signals, q=q)
            if not memberships:
                continue

            for weighting in QuantileWeighting:
                quantile_returns: dict[str, float] = {}
                quantile_counts: dict[str, int] = {}
                for quantile, names in memberships.items():
                    quantile_counts[quantile] = len(names)
                    quantile_weights = _quantile_weights(names, weighting=weighting, signal_market_cap=signal_market_cap)
                    quantile_return = (
                        float("nan")
                        if quantile_weights.empty
                        else float(next_returns.loc[names].mul(quantile_weights).sum())
                    )
                    quantile_returns[quantile] = quantile_return
                    monthly_return_rows.append(
                        {
                            "signal_date": signal_date,
                            "return_date": return_date,
                            "factor": factor_name,
                            "weighting": weighting,
                            "portfolio": quantile,
                            "return": quantile_return,
                            "constituent_count": len(names),
                        }
                    )
                    if not quantile_weights.empty:
                        for ticker, weight in quantile_weights.items():
                            weight_rows.append(
                                {
                                    "signal_date": signal_date,
                                    "return_date": return_date,
                                    "factor": factor_name,
                                    "weighting": weighting,
                                    "quantile": quantile,
                                    "ticker": str(ticker),
                                    "weight": float(weight),
                                }
                            )

                low_quantile = "Q1"
                high_quantile = f"Q{q}"
                high_return = quantile_returns[high_quantile]
                low_return = quantile_returns[low_quantile]
                high_minus_low = (
                    high_return - low_return
                    if pd.notna(high_return) and pd.notna(low_return)
                    else float("nan")
                )
                monthly_return_rows.append(
                    {
                        "signal_date": signal_date,
                        "return_date": return_date,
                        "factor": factor_name,
                        "weighting": weighting,
                        "portfolio": "high_minus_low",
                        "return": high_minus_low,
                        "constituent_count": quantile_counts[low_quantile] + quantile_counts[high_quantile],
                    }
                )
                preferred_minus_avoided = (
                    high_minus_low
                    if direction is FactorDirection.HIGH
                    else (
                        low_return - high_return
                        if pd.notna(high_return) and pd.notna(low_return)
                        else float("nan")
                    )
                )
                monthly_return_rows.append(
                    {
                        "signal_date": signal_date,
                        "return_date": return_date,
                        "factor": factor_name,
                        "weighting": weighting,
                        "portfolio": "preferred_minus_avoided",
                        "return": preferred_minus_avoided,
                        "constituent_count": quantile_counts[low_quantile] + quantile_counts[high_quantile],
                    }
                )

            rank_ic = _spearman_rank_ic(eligible_signals, next_returns)
            rank_ic_rows.append(
                {
                    "signal_date": signal_date,
                    "return_date": return_date,
                    "factor": factor_name,
                    "rank_ic": rank_ic,
                    "directional_rank_ic": rank_ic if direction is FactorDirection.HIGH else -rank_ic,
                    "n_obs": len(eligible_names),
                }
            )

    if not monthly_return_rows:
        raise Emp008FactorQuantilesUnavailableError(
            f"no factor quantile observations in requested range {start_ts.date().isoformat()} to {end_ts.date().isoformat()}"
        )

    monthly_returns = pd.DataFrame(monthly_return_rows).sort_values(
        ["signal_date", "return_date", "factor", "weighting", "portfolio"],
        kind="mergesort",
    ).reset_index(drop=True)
    portfolio_weights = pd.DataFrame(weight_rows).sort_values(
        ["signal_date", "return_date", "factor", "weighting", "quantile", "ticker"],
        kind="mergesort",
    ).reset_index(drop=True)
    rank_ic = pd.DataFrame(rank_ic_rows).sort_values(
        ["signal_date", "return_date", "factor"],
        kind="mergesort",
    ).reset_index(drop=True)
    cumulative_returns = _build_cumulative_returns(monthly_returns)
    daily_cumulative_returns = _build_daily_cumulative_returns(
        close=aligned_close,
        monthly_returns=monthly_returns,
        portfolio_weights=portfolio_weights,
        directions=normalized_directions,
        q=q,
    )
    summary = _build_summary(
        monthly_returns=monthly_returns,
        portfolio_weights=portfolio_weights,
        rank_ic=rank_ic,
        directions=normalized_directions,
        q=q,
    )

    return Emp008FactorQuantileResult(
        monthly_returns=monthly_returns,
        portfolio_weights=portfolio_weights,
        rank_ic=rank_ic,
        cumulative_returns=cumulative_returns,
        daily_cumulative_returns=daily_cumulative_returns,
        summary=summary,
    )


def summarize_monthly_returns(returns: pd.Series) -> dict[str, float | int]:
    clean_returns = pd.Series(returns, dtype=float).dropna()
    observations = int(len(clean_returns))
    if observations == 0:
        return {
            "observations": 0,
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "positive_month_rate": float("nan"),
            "mean_monthly_return": float("nan"),
        }

    mean_monthly_return = float(clean_returns.mean())
    monthly_volatility = float(clean_returns.std(ddof=0))
    annualized_volatility = float(monthly_volatility * np.sqrt(12.0))
    compounded_growth = float((1.0 + clean_returns).prod())
    if compounded_growth <= 0.0:
        annualized_return = -1.0
    else:
        annualized_return = float(compounded_growth ** (12.0 / observations) - 1.0)
    if np.isclose(monthly_volatility, 0.0) or not np.isfinite(monthly_volatility):
        sharpe = 0.0
    else:
        sharpe = float((mean_monthly_return / monthly_volatility) * np.sqrt(12.0))

    equity_curve = (1.0 + clean_returns).cumprod()
    equity_with_start = pd.concat([pd.Series([1.0], dtype=float), equity_curve], ignore_index=True)
    running_peak = equity_with_start.cummax().clip(lower=1.0)
    drawdown = equity_with_start.divide(running_peak).sub(1.0)
    max_drawdown = float(drawdown.min())

    return {
        "observations": observations,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "positive_month_rate": float(clean_returns.gt(0.0).mean()),
        "mean_monthly_return": mean_monthly_return,
    }


def _build_quantile_memberships(signals: pd.Series, *, q: int) -> dict[str, list[Any]]:
    ordered = pd.DataFrame(
        {
            "signal": signals.astype(float),
            "ticker_key": signals.index.map(str),
            "position_key": np.arange(len(signals)),
        },
        index=signals.index,
    ).sort_values(["signal", "ticker_key", "position_key"], kind="mergesort")
    ordered_names = ordered.index.tolist()
    if not ordered_names:
        return {f"Q{bucket_index}": [] for bucket_index in range(1, q + 1)}
    buckets = np.array_split(np.array(ordered_names, dtype=object), q)
    return {
        f"Q{bucket_index}": bucket.tolist()
        for bucket_index, bucket in enumerate(buckets, start=1)
    }


def _quantile_weights(
    names: list[Any],
    *,
    weighting: QuantileWeighting,
    signal_market_cap: pd.Series,
) -> pd.Series:
    index = pd.Index(names, dtype=object)
    if len(names) == 0:
        return pd.Series(dtype=float, index=index)
    if weighting is QuantileWeighting.EQUAL:
        return pd.Series(1.0 / len(names), index=index, dtype=float)
    if weighting is QuantileWeighting.MARKET_CAP:
        values = signal_market_cap.loc[names].astype(float)
        total = float(values.sum())
        return values.divide(total)
    raise ValueError(f"unsupported weighting: {weighting}")


def _validate_frame_axes(
    *,
    close: pd.DataFrame,
    market_cap: pd.DataFrame,
    universe: pd.DataFrame,
    factors: Mapping[str, pd.DataFrame],
) -> None:
    frames: list[tuple[str, pd.DataFrame]] = [
        ("close", close),
        ("market_cap", market_cap),
        ("universe", universe),
    ]
    frames.extend((f"factor '{name}'", frame) for name, frame in factors.items())
    for frame_name, frame in frames:
        if not frame.columns.is_unique:
            duplicates = ", ".join(sorted({str(label) for label in frame.columns[frame.columns.duplicated()]}))
            raise ValueError(f"duplicate ticker labels in {frame_name}: {duplicates}")

        string_map: dict[str, list[Any]] = {}
        for label in frame.columns.tolist():
            string_map.setdefault(str(label), []).append(label)
        ambiguous = {
            ticker_text: labels
            for ticker_text, labels in string_map.items()
            if len(labels) > 1
        }
        if ambiguous:
            details = ", ".join(
                f"{ticker_text} <- {labels!r}"
                for ticker_text, labels in sorted(ambiguous.items(), key=lambda item: item[0])
            )
            raise ValueError(f"ambiguous ticker labels in {frame_name}: {details}")


def _spearman_rank_ic(signals: pd.Series, next_returns: pd.Series) -> float:
    if len(signals) < 2 or signals.nunique(dropna=True) < 2 or next_returns.nunique(dropna=True) < 2:
        return float("nan")
    return float(signals.corr(next_returns, method="spearman"))


def _build_cumulative_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    cumulative = monthly_returns.loc[:, ["signal_date", "return_date", "factor", "weighting", "portfolio", "return"]].copy()
    cumulative["cumulative_return"] = cumulative.groupby(
        ["factor", "weighting", "portfolio"],
        observed=True,
        sort=False,
    )["return"].transform(lambda values: (1.0 + values).cumprod() - 1.0)
    high_minus_low = cumulative[cumulative["portfolio"] == "high_minus_low"].loc[
        :, ["signal_date", "return_date", "factor", "weighting", "return", "cumulative_return"]
    ].rename(
        columns={
            "return": "high_minus_low_return",
            "cumulative_return": "high_minus_low_cumulative_return",
        }
    )
    preferred_mask = cumulative["portfolio"] == "preferred_minus_avoided"
    preferred = cumulative.loc[preferred_mask].merge(
        high_minus_low,
        on=["signal_date", "return_date", "factor", "weighting"],
        how="left",
        validate="one_to_one",
    )
    same_direction = np.isclose(
        preferred["return"].to_numpy(dtype=float),
        preferred["high_minus_low_return"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    )
    reversed_direction = np.isclose(
        preferred["return"].to_numpy(dtype=float),
        preferred["high_minus_low_return"].mul(-1.0).to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    )
    preferred.loc[same_direction, "cumulative_return"] = preferred.loc[
        same_direction, "high_minus_low_cumulative_return"
    ].to_numpy(dtype=float)
    preferred.loc[reversed_direction, "cumulative_return"] = preferred.loc[
        reversed_direction, "high_minus_low_cumulative_return"
    ].mul(-1.0).to_numpy(dtype=float)
    cumulative.loc[preferred_mask, "cumulative_return"] = preferred["cumulative_return"].to_numpy(dtype=float)
    cumulative = cumulative.drop(columns="return")
    return cumulative.sort_values(
        ["factor", "weighting", "portfolio", "return_date", "signal_date"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_daily_cumulative_returns(
    *,
    close: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    directions: Mapping[str, FactorDirection],
    q: int,
) -> pd.DataFrame:
    if portfolio_weights.empty:
        return _empty_daily_cumulative_returns_frame()

    ordered_close = close.sort_index()
    close_ticker_labels = {str(label): label for label in ordered_close.columns}
    ordered_weights = portfolio_weights.sort_values(
        ["factor", "weighting", "quantile", "signal_date", "return_date", "ticker"],
        kind="mergesort",
    ).reset_index(drop=True)
    period_navs: dict[tuple[str, str, str, pd.Timestamp, pd.Timestamp], pd.Series] = {}
    quantile_rows: list[dict[str, object]] = []
    prior_quantile_wealth: dict[tuple[str, str, str], float] = {}

    for group_key, group in ordered_weights.groupby(
        ["factor", "weighting", "quantile", "signal_date", "return_date"],
        observed=True,
        sort=False,
    ):
        factor_name, weighting, quantile, signal_date, return_date = group_key
        weights = group.set_index("ticker")["weight"].astype(float)
        price_columns = [close_ticker_labels[str(ticker)] for ticker in weights.index]
        period_dates = ordered_close.index[(ordered_close.index >= signal_date) & (ordered_close.index <= return_date)]
        period_prices = ordered_close.loc[period_dates, price_columns].astype(float).ffill()
        period_prices.columns = weights.index
        signal_prices = period_prices.loc[signal_date].astype(float)
        period_nav = period_prices.divide(signal_prices).mul(weights, axis=1).sum(axis=1)
        period_navs[(factor_name, str(weighting), str(quantile), signal_date, return_date)] = period_nav

        wealth_key = (factor_name, str(weighting), str(quantile))
        prior_wealth = prior_quantile_wealth.get(wealth_key, 1.0)
        emit_nav = period_nav if wealth_key not in prior_quantile_wealth else period_nav.iloc[1:]
        for date, nav in emit_nav.items():
            quantile_rows.append(
                {
                    "signal_date": signal_date,
                    "date": pd.Timestamp(date),
                    "factor": factor_name,
                    "weighting": weighting,
                    "portfolio": str(quantile),
                    "cumulative_return": float((prior_wealth * nav) - 1.0),
                }
            )
        prior_quantile_wealth[wealth_key] = float(prior_wealth * period_nav.iloc[-1])

    high_minus_low_rows: list[dict[str, object]] = []
    prior_spread_wealth: dict[tuple[str, str, str], float] = {}
    period_coverage = (
        monthly_returns.loc[:, ["factor", "weighting", "signal_date", "return_date"]]
        .drop_duplicates()
        .sort_values(["factor", "weighting", "signal_date", "return_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    for factor_name, weighting, signal_date, return_date in period_coverage.itertuples(index=False, name=None):
        low_nav = period_navs.get((factor_name, str(weighting), "Q1", signal_date, return_date))
        high_nav = period_navs.get((factor_name, str(weighting), f"Q{q}", signal_date, return_date))
        if low_nav is None or high_nav is None:
            continue
        high_minus_low = high_nav.sub(low_nav)
        wealth_key = (factor_name, str(weighting), "high_minus_low")
        prior_wealth = prior_spread_wealth.get(wealth_key, 1.0)
        emit_spread = high_minus_low if wealth_key not in prior_spread_wealth else high_minus_low.iloc[1:]
        for date, spread_value in emit_spread.items():
            high_minus_low_rows.append(
                {
                    "signal_date": signal_date,
                    "date": pd.Timestamp(date),
                    "factor": factor_name,
                    "weighting": weighting,
                    "portfolio": "high_minus_low",
                    "cumulative_return": float((prior_wealth * (1.0 + spread_value)) - 1.0),
                }
            )
        prior_spread_wealth[wealth_key] = float(prior_wealth * (1.0 + high_minus_low.iloc[-1]))

    spread_rows = list(high_minus_low_rows)
    for row in high_minus_low_rows:
        direction = directions[str(row["factor"])]
        spread_rows.append(
            {
                **row,
                "portfolio": "preferred_minus_avoided",
                "cumulative_return": (
                    float(row["cumulative_return"])
                    if direction is FactorDirection.HIGH
                    else float(-float(row["cumulative_return"]))
                ),
            }
        )

    daily_cumulative_returns = pd.DataFrame(
        quantile_rows + spread_rows,
        columns=_DAILY_CUMULATIVE_RETURNS_COLUMNS,
    ).sort_values(
        ["factor", "weighting", "portfolio", "date", "signal_date"],
        kind="mergesort",
    ).reset_index(drop=True)

    if daily_cumulative_returns.duplicated(["date", "factor", "weighting", "portfolio"]).any():
        raise ValueError("daily_cumulative_returns must have unique date/factor/weighting/portfolio rows")
    if not daily_cumulative_returns["cumulative_return"].map(np.isfinite).all():
        raise ValueError("daily_cumulative_returns cumulative_return values must be finite")
    return daily_cumulative_returns


def _build_summary(
    *,
    monthly_returns: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    rank_ic: pd.DataFrame,
    directions: Mapping[str, FactorDirection],
    q: int,
) -> pd.DataFrame:
    turnover = _calculate_one_way_turnover(portfolio_weights, directions=directions, q=q)
    ic_stats = _calculate_ic_stats(rank_ic)
    monotonicity = _calculate_quantile_monotonicity(monthly_returns, directions=directions)

    summary_rows: list[dict[str, object]] = []
    for group_key, group in monthly_returns.groupby(["factor", "weighting", "portfolio"], observed=True, sort=False):
        factor_name, weighting, portfolio = group_key
        performance = summarize_monthly_returns(group["return"])
        turnover_value = turnover.get((factor_name, str(weighting), portfolio), float("nan"))
        ic_row = ic_stats.loc[factor_name] if factor_name in ic_stats.index else None
        monotonicity_value = monotonicity.get((factor_name, str(weighting)), float("nan"))
        summary_rows.append(
            {
                "factor": factor_name,
                "weighting": weighting,
                "portfolio": portfolio,
                **performance,
                "average_constituent_count": float(group["constituent_count"].mean()) if not group.empty else float("nan"),
                "average_one_way_turnover": float(turnover_value),
                "mean_rank_ic": float(ic_row["mean_rank_ic"]) if ic_row is not None else float("nan"),
                "directional_mean_rank_ic": float(ic_row["directional_mean_rank_ic"]) if ic_row is not None else float("nan"),
                "ic_information_ratio": float(ic_row["ic_information_ratio"]) if ic_row is not None else float("nan"),
                "ic_positive_rate": float(ic_row["ic_positive_rate"]) if ic_row is not None else float("nan"),
                "quantile_monotonicity": float(monotonicity_value),
            }
        )

    return pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS).sort_values(
        ["factor", "weighting", "portfolio"],
        kind="mergesort",
    ).reset_index(drop=True)


def _calculate_one_way_turnover(
    portfolio_weights: pd.DataFrame,
    *,
    directions: Mapping[str, FactorDirection],
    q: int,
) -> dict[tuple[str, str, str], float]:
    base_vectors = _collect_weight_vectors(portfolio_weights)
    turnover: dict[tuple[str, str, str], float] = {}
    for key, dated_weights in base_vectors.items():
        turnover[(key[0], key[1], key[2])] = _average_turnover_from_vectors(dated_weights)

    for factor_name, weighting in portfolio_weights[["factor", "weighting"]].drop_duplicates().itertuples(index=False):
        low_key = (factor_name, str(weighting), "Q1")
        high_key = (factor_name, str(weighting), f"Q{q}")
        low_vectors = base_vectors.get(low_key)
        high_vectors = base_vectors.get(high_key)
        if not low_vectors or not high_vectors:
            continue
        shared_dates = sorted(set(low_vectors).intersection(high_vectors))
        if not shared_dates:
            continue
        high_minus_low_vectors = {
            date: high_vectors[date].mul(1.0, fill_value=0.0).sub(low_vectors[date], fill_value=0.0)
            for date in shared_dates
        }
        turnover[(factor_name, str(weighting), "high_minus_low")] = _average_turnover_from_vectors(high_minus_low_vectors)
        direction = directions[factor_name]
        if direction is FactorDirection.HIGH:
            preferred_vectors = high_minus_low_vectors
        else:
            preferred_vectors = {date: vector.mul(-1.0) for date, vector in high_minus_low_vectors.items()}
        turnover[(factor_name, str(weighting), "preferred_minus_avoided")] = _average_turnover_from_vectors(preferred_vectors)
    return turnover


def _collect_weight_vectors(portfolio_weights: pd.DataFrame) -> dict[tuple[str, str, str], dict[pd.Timestamp, pd.Series]]:
    vectors: dict[tuple[str, str, str], dict[pd.Timestamp, pd.Series]] = {}
    for group_key, group in portfolio_weights.groupby(["factor", "weighting", "quantile", "return_date"], observed=True, sort=False):
        factor_name, weighting, quantile, return_date = group_key
        vector = pd.Series(group["weight"].to_numpy(dtype=float), index=group["ticker"].astype(str), dtype=float).sort_index()
        vectors.setdefault((factor_name, str(weighting), quantile), {})[pd.Timestamp(return_date)] = vector
    return vectors


def _average_turnover_from_vectors(dated_weights: Mapping[pd.Timestamp, pd.Series]) -> float:
    ordered_dates = sorted(dated_weights)
    if len(ordered_dates) < 2:
        return float("nan")

    turnovers: list[float] = []
    previous = dated_weights[ordered_dates[0]]
    for date in ordered_dates[1:]:
        current = dated_weights[date]
        aligned_index = previous.index.union(current.index)
        diff = current.reindex(aligned_index, fill_value=0.0).sub(previous.reindex(aligned_index, fill_value=0.0))
        turnovers.append(float(0.5 * diff.abs().sum()))
        previous = current
    return float(np.mean(turnovers)) if turnovers else float("nan")


def _calculate_ic_stats(rank_ic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor_name, group in rank_ic.groupby("factor", observed=True, sort=False):
        directional = group["directional_rank_ic"].astype(float).dropna()
        raw = group["rank_ic"].astype(float).dropna()
        directional_mean = float(directional.mean()) if not directional.empty else float("nan")
        raw_mean = float(raw.mean()) if not raw.empty else float("nan")
        if directional.empty:
            information_ratio = float("nan")
            positive_rate = float("nan")
        else:
            directional_std = float(directional.std(ddof=0))
            if directional_std == 0.0 and np.isfinite(directional_mean):
                information_ratio = 0.0
            elif directional_std == 0.0 or not np.isfinite(directional_std):
                information_ratio = float("nan")
            else:
                information_ratio = float((directional_mean / directional_std) * np.sqrt(12.0))
            positive_rate = float(directional.gt(0.0).mean())
        rows.append(
            {
                "factor": factor_name,
                "mean_rank_ic": raw_mean,
                "directional_mean_rank_ic": directional_mean,
                "ic_information_ratio": information_ratio,
                "ic_positive_rate": positive_rate,
            }
        )
    return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame(columns=["mean_rank_ic"]).set_index(pd.Index([], name="factor"))


def _calculate_quantile_monotonicity(
    monthly_returns: pd.DataFrame,
    *,
    directions: Mapping[str, FactorDirection],
) -> dict[tuple[str, str], float]:
    quantile_rows = monthly_returns[monthly_returns["portfolio"].astype(str).str.match(r"^Q\d+$", na=False)].copy()
    if quantile_rows.empty:
        return {}
    quantile_rows["quantile_number"] = quantile_rows["portfolio"].astype(str).str[1:].astype(int)

    monotonicity: dict[tuple[str, str], float] = {}
    for (factor_name, weighting), group in quantile_rows.groupby(["factor", "weighting"], observed=True, sort=False):
        averages = group.groupby("quantile_number", observed=True, sort=True)["return"].mean()
        valid = averages.dropna()
        if len(valid) < 2 or valid.nunique(dropna=True) < 2:
            monotonicity[(factor_name, str(weighting))] = float("nan")
            continue
        ordinal = pd.Series(valid.index.to_numpy(dtype=float), index=valid.index, dtype=float)
        correlation = ordinal.corr(valid.astype(float), method="spearman")
        if pd.isna(correlation):
            monotonicity[(factor_name, str(weighting))] = float("nan")
            continue
        direction_sign = 1.0 if directions[factor_name] is FactorDirection.HIGH else -1.0
        monotonicity[(factor_name, str(weighting))] = float(direction_sign * correlation)
    return monotonicity


def _validate_result_for_output(
    result: Emp008FactorQuantileResult,
    *,
    factor_set: FactorSetId | str,
    q: int,
) -> None:
    if q < 2:
        raise ValueError("q must be at least 2")
    if result.monthly_returns.empty or result.portfolio_weights.empty or result.rank_ic.empty or result.cumulative_returns.empty or result.summary.empty:
        raise ValueError("result must be nonempty before writing artifacts")

    _require_columns(result.monthly_returns, _MONTHLY_RETURNS_COLUMNS, "monthly_returns")
    _require_columns(result.portfolio_weights, _PORTFOLIO_WEIGHTS_COLUMNS, "portfolio_weights")
    _require_columns(result.rank_ic, _RANK_IC_COLUMNS, "rank_ic")
    _require_columns(result.cumulative_returns, _CUMULATIVE_RETURNS_COLUMNS, "cumulative_returns")
    _require_columns(result.summary, _SUMMARY_COLUMNS, "summary")

    factor_definitions = factor_definitions_for_set(factor_set)
    directions = {
        definition.id.value: definition.direction
        for definition in factor_definitions
    }
    observed_factors = set(result.monthly_returns["factor"])
    if not observed_factors:
        raise ValueError("monthly_returns must contain at least one factor")
    if not observed_factors.issubset(set(directions)):
        raise ValueError("monthly_returns factors must be compatible with the requested factor set")

    quantile_weights = result.portfolio_weights.copy()
    if not quantile_weights["weight"].map(np.isfinite).all():
        raise ValueError("portfolio weight values must be finite")
    if not quantile_weights["weight"].gt(0.0).all():
        raise ValueError("portfolio weight values must be positive")
    if quantile_weights.duplicated(["signal_date", "return_date", "factor", "weighting", "quantile", "ticker"]).any():
        raise ValueError("duplicate weight rows are not allowed")
    weight_sums = quantile_weights.groupby(
        ["signal_date", "return_date", "factor", "weighting", "quantile"],
        observed=True,
        sort=False,
    )["weight"].sum()
    if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("portfolio weight groups must sum to 1 within tolerance")

    if not result.monthly_returns["return"].dropna().map(np.isfinite).all():
        raise ValueError("monthly return values must be finite when present")
    quantile_monthly = result.monthly_returns[result.monthly_returns["portfolio"].astype(str).str.match(r"^Q\d+$", na=False)]
    if not quantile_monthly["return"].dropna().map(np.isfinite).all():
        raise ValueError("quantile monthly return values must be finite when present")
    if not result.rank_ic["n_obs"].map(np.isfinite).all():
        raise ValueError("rank_ic n_obs values must be finite")
    if not result.cumulative_returns["cumulative_return"].dropna().map(np.isfinite).all():
        raise ValueError("cumulative return values must be finite when present")

    if set(result.monthly_returns["factor"]) != set(result.rank_ic["factor"]):
        raise ValueError("monthly_returns and rank_ic factors must match")
    if not set(result.summary["factor"]).issubset(set(result.monthly_returns["factor"])):
        raise ValueError("summary factors must be present in monthly_returns")

    summary_keys = set(result.summary[["factor", "weighting", "portfolio"]].itertuples(index=False, name=None))
    monthly_keys = set(result.monthly_returns[["factor", "weighting", "portfolio"]].itertuples(index=False, name=None))
    if summary_keys != monthly_keys:
        raise ValueError("summary rows must align with monthly_returns portfolios")

    cumulative_keys = set(result.cumulative_returns[["factor", "weighting", "portfolio"]].itertuples(index=False, name=None))
    if cumulative_keys != monthly_keys:
        raise ValueError("cumulative_returns rows must align with monthly_returns portfolios")

    _validate_membership_parity(result.portfolio_weights)
    _validate_rank_ic(
        result.rank_ic,
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        directions=directions,
    )
    expected_cumulative = _build_cumulative_returns(result.monthly_returns)
    _compare_derived_frame(
        actual=result.cumulative_returns,
        expected=expected_cumulative,
        frame_name="cumulative_returns",
    )
    expected_summary = _build_summary(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=result.rank_ic,
        directions={factor: directions[factor] for factor in observed_factors},
        q=q,
    )
    _compare_derived_frame(
        actual=result.summary,
        expected=expected_summary,
        frame_name="summary",
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    if tuple(frame.columns) != tuple(columns):
        raise ValueError(f"{frame_name} schema mismatch")


def _validate_membership_parity(portfolio_weights: pd.DataFrame) -> None:
    equal = portfolio_weights[portfolio_weights["weighting"] == QuantileWeighting.EQUAL]
    cap = portfolio_weights[portfolio_weights["weighting"] == QuantileWeighting.MARKET_CAP]
    equal_membership = (
        equal.groupby(["factor", "signal_date", "return_date", "quantile"], observed=True, sort=False)["ticker"]
        .apply(lambda tickers: tuple(sorted(str(ticker) for ticker in tickers)))
        .sort_index()
    )
    cap_membership = (
        cap.groupby(["factor", "signal_date", "return_date", "quantile"], observed=True, sort=False)["ticker"]
        .apply(lambda tickers: tuple(sorted(str(ticker) for ticker in tickers)))
        .sort_index()
    )
    if not equal_membership.index.equals(cap_membership.index) or not equal_membership.equals(cap_membership):
        raise ValueError("equal and market-cap membership must match for each available factor/date/quantile")


def _validate_rank_ic(
    rank_ic: pd.DataFrame,
    *,
    monthly_returns: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    directions: Mapping[str, FactorDirection],
) -> None:
    coverage = (
        monthly_returns.loc[:, ["signal_date", "return_date", "factor"]]
        .drop_duplicates()
        .sort_values(["signal_date", "return_date", "factor"], kind="mergesort")
        .reset_index(drop=True)
    )
    actual = rank_ic.loc[:, ["signal_date", "return_date", "factor"]].sort_values(
        ["signal_date", "return_date", "factor"],
        kind="mergesort",
    ).reset_index(drop=True)
    _compare_derived_frame(actual=actual, expected=coverage, frame_name="rank_ic coverage", check_dtype=False)

    n_obs = rank_ic["n_obs"]
    if not n_obs.map(lambda value: float(value).is_integer() if np.isfinite(value) else False).all():
        raise ValueError("rank_ic n_obs must be integer valued")
    if not n_obs.ge(0).all():
        raise ValueError("rank_ic n_obs must be non-negative")

    expected_n_obs = _expected_rank_ic_n_obs(portfolio_weights)
    actual_n_obs = rank_ic.loc[:, ["signal_date", "return_date", "factor", "n_obs"]].sort_values(
        ["signal_date", "return_date", "factor"],
        kind="mergesort",
    ).reset_index(drop=True)
    expected_n_obs_frame = expected_n_obs.sort_values(
        ["signal_date", "return_date", "factor"],
        kind="mergesort",
    ).reset_index(drop=True)
    _compare_derived_frame(
        actual=actual_n_obs,
        expected=expected_n_obs_frame,
        frame_name="rank_ic n_obs",
        check_dtype=False,
    )

    rank_ic_values = rank_ic["rank_ic"].astype(float)
    directional_values = rank_ic["directional_rank_ic"].astype(float)
    valid_rank_ic = rank_ic_values.dropna()
    valid_directional = directional_values.dropna()
    if not valid_rank_ic.map(np.isfinite).all() or not valid_directional.map(np.isfinite).all():
        raise ValueError("rank_ic values must be finite when present")
    if not valid_rank_ic.between(-1.0 - 1e-12, 1.0 + 1e-12).all():
        raise ValueError("rank_ic values must stay within [-1, 1]")
    if not valid_directional.between(-1.0 - 1e-12, 1.0 + 1e-12).all():
        raise ValueError("directional_rank_ic values must stay within [-1, 1]")
    insufficient_obs = n_obs.lt(2)
    insufficient_ic = rank_ic.loc[insufficient_obs, ["rank_ic", "directional_rank_ic"]]
    if not insufficient_ic.isna().all().all():
        raise ValueError("rank_ic with n_obs < 2 must be NaN")

    expected_directional = rank_ic["rank_ic"].astype(float).copy()
    low_mask = rank_ic["factor"].map(lambda factor: directions[factor] is FactorDirection.LOW)
    expected_directional.loc[low_mask] = expected_directional.loc[low_mask] * -1.0
    equal_mask = (
        rank_ic["directional_rank_ic"].isna() & expected_directional.isna()
    ) | np.isclose(
        rank_ic["directional_rank_ic"].astype(float),
        expected_directional,
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    )
    if not bool(np.all(equal_mask)):
        raise ValueError("rank_ic directional_rank_ic does not match factor directions")


def _expected_rank_ic_n_obs(portfolio_weights: pd.DataFrame) -> pd.DataFrame:
    equal_weight_rows = portfolio_weights[portfolio_weights["weighting"] == QuantileWeighting.EQUAL]
    quantile_rows = equal_weight_rows[equal_weight_rows["quantile"].astype(str).str.match(r"^Q\d+$", na=False)]
    grouped = (
        quantile_rows.groupby(["signal_date", "return_date", "factor"], observed=True, sort=False)["ticker"]
        .nunique()
        .reset_index(name="n_obs")
    )
    return grouped


def _compare_derived_frame(
    *,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    frame_name: str,
    check_dtype: bool = True,
) -> None:
    try:
        assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=check_dtype,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )
    except AssertionError as exc:
        raise ValueError(f"{frame_name} does not match canonical derivation") from exc


def _build_manifest(
    result: Emp008FactorQuantileResult,
    *,
    factor_set: FactorSetId | str,
    q: int,
    output_dir: Path,
) -> dict[str, object]:
    definitions = factor_definitions_for_set(factor_set)
    plotted_factor_count = int(
        result.cumulative_returns.loc[:, "factor"].drop_duplicates().shape[0]
        if not result.cumulative_returns.empty
        else 0
    )
    artifacts = {
        "monthly_returns.csv": {"path": str(output_dir / "monthly_returns.csv"), "rows": int(len(result.monthly_returns))},
        "monthly_returns.parquet": {"path": str(output_dir / "monthly_returns.parquet"), "rows": int(len(result.monthly_returns))},
        "portfolio_weights.parquet": {"path": str(output_dir / "portfolio_weights.parquet"), "rows": int(len(result.portfolio_weights))},
        "rank_ic.csv": {"path": str(output_dir / "rank_ic.csv"), "rows": int(len(result.rank_ic))},
        "rank_ic.parquet": {"path": str(output_dir / "rank_ic.parquet"), "rows": int(len(result.rank_ic))},
        "cumulative_returns.csv": {"path": str(output_dir / "cumulative_returns.csv"), "rows": int(len(result.cumulative_returns))},
        "cumulative_quintiles_equal_weight.png": {
            "path": str(output_dir / "cumulative_quintiles_equal_weight.png"),
            "rows": plotted_factor_count,
        },
        "cumulative_quintiles_market_cap_weight.png": {
            "path": str(output_dir / "cumulative_quintiles_market_cap_weight.png"),
            "rows": plotted_factor_count,
        },
        "summary.csv": {"path": str(output_dir / "summary.csv"), "rows": int(len(result.summary))},
        "summary.json": {"path": str(output_dir / "summary.json"), "rows": int(len(result.summary))},
        "manifest.json": {"path": str(output_dir / "manifest.json"), "rows": 1},
    }
    return {
        "factor_set": str(factor_set),
        "selected_factors": [definition.id.value for definition in definitions],
        "directions": {definition.id.value: definition.direction.value for definition in definitions},
        "weighting_modes": [QuantileWeighting.EQUAL.value, QuantileWeighting.MARKET_CAP.value],
        "q": int(q),
        "min_signal_date": pd.Timestamp(result.monthly_returns["signal_date"].min()).date().isoformat(),
        "max_return_date": pd.Timestamp(result.monthly_returns["return_date"].max()).date().isoformat(),
        "timing": "month_end_t_to_next_month_end",
        "market_cap_field": "market_cap",
        "artifacts": artifacts,
    }


def _publish_artifacts_atomically(
    *,
    artifact_paths: Sequence[Path],
    destination: Path,
    staging_root: Path,
) -> None:
    backup_dir = staging_root / "backup"
    backup_dir.mkdir()
    published_targets: list[Path] = []
    backed_up_targets: list[Path] = []
    try:
        for filename in _ARTIFACT_FILENAMES:
            target = destination / filename
            if target.exists():
                target.replace(backup_dir / filename)
                backed_up_targets.append(target)
        for path in artifact_paths:
            target = destination / path.name
            path.replace(target)
            published_targets.append(target)
    except Exception:
        for target in reversed(published_targets):
            if target.exists():
                target.unlink()
        for target in backed_up_targets:
            backup_path = backup_dir / target.name
            if backup_path.exists():
                backup_path.replace(target)
        raise


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        converted: dict[str, object] = {}
        for key, value in row.items():
            converted[key] = _json_safe_value(value)
        records.append(converted)
    return records


def _json_safe_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_cumulative_quintile_plot(
    *,
    path: Path,
    cumulative_returns: pd.DataFrame,
    directions: Mapping[str, FactorDirection],
    weighting: QuantileWeighting,
    q: int,
) -> None:
    figure = _build_cumulative_quintile_figure(
        cumulative_returns=cumulative_returns,
        directions=directions,
        weighting=weighting,
        q=q,
    )
    try:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)


def _build_cumulative_quintile_figure(
    *,
    cumulative_returns: pd.DataFrame,
    directions: Mapping[str, FactorDirection],
    weighting: QuantileWeighting,
    q: int,
):
    filtered = cumulative_returns[cumulative_returns["weighting"] == weighting].copy()
    if filtered.empty:
        raise ValueError(f"no cumulative returns available for weighting '{weighting.value}'")

    filtered["return_date"] = pd.to_datetime(filtered["return_date"])
    available_factors = set(filtered["factor"].astype(str))
    factor_names = [factor_name for factor_name in directions if factor_name in available_factors]
    if not factor_names:
        raise ValueError(f"no cumulative returns available for requested factors under '{weighting.value}'")

    ncols = 2 if len(factor_names) <= 4 else 3
    nrows = math.ceil(len(factor_names) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.0 * ncols, 3.8 * nrows),
        sharex=False,
        sharey=False,
    )
    axes_array = np.atleast_1d(axes).ravel()
    quantile_portfolios = [f"Q{bucket}" for bucket in range(1, q + 1)]
    palette = plt.get_cmap("tab10")

    for index, factor_name in enumerate(factor_names):
        axis = axes_array[index]
        factor_rows = filtered[filtered["factor"] == factor_name]
        for color_index, portfolio in enumerate(quantile_portfolios):
            portfolio_rows = factor_rows[factor_rows["portfolio"] == portfolio].sort_values("return_date")
            if portfolio_rows.empty:
                continue
            axis.plot(
                portfolio_rows["return_date"],
                portfolio_rows["cumulative_return"] * 100.0,
                color=palette(color_index % 10),
                linewidth=1.8,
                label=portfolio,
            )
        spread_rows = factor_rows[factor_rows["portfolio"] == "preferred_minus_avoided"].sort_values("return_date")
        if not spread_rows.empty:
            axis.plot(
                spread_rows["return_date"],
                spread_rows["cumulative_return"] * 100.0,
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="preferred_minus_avoided",
            )
        axis.axhline(0.0, color="#9aa0a6", linewidth=0.8)
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.set_title(factor_name)
        axis.set_ylabel("Cumulative return (%)")
        axis.tick_params(axis="x", rotation=30)

    for axis in axes_array[len(factor_names) :]:
        axis.set_visible(False)

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=min(len(labels), 6),
            frameon=False,
        )
    fig.suptitle(f"Cumulative quintiles: {weighting.value}", x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return fig


def _empty_cumulative_returns_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="float64" if column == "cumulative_return" else "object") for column in _CUMULATIVE_RETURNS_COLUMNS}).astype(
        {"signal_date": "datetime64[ns]", "return_date": "datetime64[ns]", "cumulative_return": "float64"}
    )


def _empty_daily_cumulative_returns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(dtype="float64" if column == "cumulative_return" else "object")
            for column in _DAILY_CUMULATIVE_RETURNS_COLUMNS
        }
    ).astype({"signal_date": "datetime64[ns]", "date": "datetime64[ns]", "cumulative_return": "float64"})


def _empty_summary_frame() -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in _SUMMARY_COLUMNS if column not in {"factor", "weighting", "portfolio"}})
    frame.insert(0, "portfolio", pd.Series(dtype="object"))
    frame.insert(0, "weighting", pd.Series(dtype="object"))
    frame.insert(0, "factor", pd.Series(dtype="object"))
    frame["observations"] = frame["observations"].astype("int64")
    return frame.loc[:, _SUMMARY_COLUMNS]


__all__ = [
    "Emp008FactorQuantileResult",
    "QuantileWeighting",
    "evaluate_factor_quantiles",
    "run_emp008_factor_quantiles",
    "summarize_monthly_returns",
]
