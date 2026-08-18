from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting.data import MarketData
from backtesting.strategies.emp008.factor_pipeline import PreparedEmp008Factors
from backtesting.strategies.emp008.factor_quantiles import (
    Emp008FactorQuantilesUnavailableError,
    Emp008FactorQuantileResult,
    QuantileWeighting,
    _build_cumulative_quintile_figure,
    evaluate_factor_quantiles,
    run_emp008_factor_quantiles,
    summarize_monthly_returns,
)
from backtesting.strategies.emp008.factor_registry import (
    FactorDirection,
    get_factor_set_definition,
)
from backtesting.strategies.emp008.data import Emp008Config


def _frame(
    dates: pd.DatetimeIndex,
    columns: list[str],
    rows: list[list[object]],
) -> pd.DataFrame:
    return pd.DataFrame(rows, index=dates, columns=columns)


def _core_inputs() -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tuple[pd.Timestamp, ...],
]:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"])
    columns = ["A", "B", "C", "D", "E", "F", "X", "Y", "Z"]
    close = _frame(
        dates,
        columns,
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 1.0],
            [12.1, 13.2, 14.3, 15.4, 16.5, 17.6, 18.7, 19.8, 1.1],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 1.2],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [10.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 10.0],
            [1000.0, 1.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 10.0],
            [11.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0, 1.0, 10.0],
            [12.0, 32.0, 42.0, 52.0, 62.0, 72.0, 82.0, 1.0, 10.0],
        ],
    )
    universe = _frame(
        dates,
        columns,
        [
            [True, True, True, True, True, True, False, True, True],
            [True, True, True, True, True, True, False, True, True],
            [True, True, True, True, True, True, False, True, True],
            [True, True, True, True, True, True, False, True, True],
        ],
    ).astype(bool)
    factors = {
        "high_factor": _frame(
            dates,
            columns,
            [
                [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
        ),
        "low_factor": _frame(
            dates,
            columns,
            [
                [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
        ),
        "sparse_factor": _frame(
            dates,
            columns,
            [
                [None, None, None, None, None, None, None, None, None],
                [10.0, 20.0, 30.0, None, None, None, None, None, None],
                [11.0, 21.0, 31.0, None, None, None, None, None, None],
                [12.0, 22.0, 32.0, None, None, None, None, None, None],
            ],
        ),
    }
    monthly_dates = tuple(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"]))
    return factors, close, market_cap, universe, monthly_dates


def _empty_result() -> Emp008FactorQuantileResult:
    return Emp008FactorQuantileResult(
        monthly_returns=pd.DataFrame(),
        portfolio_weights=pd.DataFrame(),
        rank_ic=pd.DataFrame(),
        cumulative_returns=pd.DataFrame(),
        daily_cumulative_returns=pd.DataFrame(),
        summary=pd.DataFrame(),
    )


def _sample_prepared() -> PreparedEmp008Factors:
    dates = pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29"])
    monthly_dates = tuple(pd.to_datetime(["2024-01-31", "2024-02-29"]))
    columns = ["A", "B"]
    close = _frame(
        dates,
        columns,
        [
            [9.0, 19.0],
            [10.0, 20.0],
            [10.5, 20.5],
            [11.0, 21.0],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [90.0, 190.0],
            [100.0, 300.0],
            [110.0, 310.0],
            [120.0, 320.0],
        ],
    )
    universe = _frame(
        dates,
        columns,
        [
            [True, True],
            [True, True],
            [True, True],
            [True, True],
        ],
    ).astype(bool)
    factor_names = [factor_id.value for factor_id in get_factor_set_definition("mfbt").factors]
    alpha_factors = {
        factor_name: _frame(
            dates,
            columns,
            [
                [0.1, -0.1],
                [0.2, -0.2],
                [0.3, -0.3],
                [0.4, -0.4],
            ],
        )
        for factor_name in factor_names
    }
    market = MarketData(
        frames={
            "close": close,
            "market_cap": market_cap,
            "float_market_cap": market_cap,
            "k200_yn": universe,
            "sector_neutral_big": _frame(dates, columns, [["Tech", "Tech"]] * len(dates)),
            "bm_weights": _frame(dates, columns, [[0.25, 0.75]] * len(dates)),
        },
        universe=None,
        benchmark=None,
    )
    return PreparedEmp008Factors(
        config=Emp008Config(),
        market=market,
        factor_set_definition=get_factor_set_definition("mfbt"),
        raw_factors=dict(alpha_factors),
        alpha_factors=alpha_factors,
        sector_factors={},
        close=close,
        market_cap=market_cap,
        float_market_cap=market_cap,
        universe=universe,
        sector=_frame(dates, columns, [["Tech", "Tech"]] * len(dates)),
        benchmark_weights=_frame(dates, columns, [[0.25, 0.75]] * len(dates)),
        monthly_dates=monthly_dates,
    )


def _summary_row(
    summary: pd.DataFrame,
    *,
    factor: str,
    weighting: QuantileWeighting,
    portfolio: str,
) -> pd.Series:
    rows = summary[
        (summary["factor"] == factor)
        & (summary["weighting"] == weighting)
        & (summary["portfolio"] == portfolio)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_evaluate_factor_quantiles_reuses_membership_for_both_weightings() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    result = evaluate_factor_quantiles(
        factors={"high_factor": factors["high_factor"]},
        directions={"high_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[:2],
        start="2024-02-29",
        end="2024-02-29",
        q=5,
    )

    assert {"signal_date", "return_date", "factor", "weighting", "quantile", "ticker", "weight"} <= set(
        result.portfolio_weights.columns
    )
    assert {"signal_date", "return_date", "factor", "weighting", "portfolio", "return", "constituent_count"} <= set(
        result.monthly_returns.columns
    )

    weights = result.portfolio_weights.sort_values(
        ["weighting", "quantile", "ticker"],
        kind="mergesort",
    ).reset_index(drop=True)
    membership_columns = ["signal_date", "return_date", "factor", "quantile", "ticker"]
    equal_membership = (
        weights.loc[weights["weighting"] == QuantileWeighting.EQUAL, membership_columns]
        .sort_values(membership_columns, kind="mergesort")
        .reset_index(drop=True)
    )
    cap_membership = (
        weights.loc[weights["weighting"] == QuantileWeighting.MARKET_CAP, membership_columns]
        .sort_values(membership_columns, kind="mergesort")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(equal_membership, cap_membership)

    equal_q1 = weights[
        (weights["weighting"] == QuantileWeighting.EQUAL)
        & (weights["quantile"] == "Q1")
        & (weights["ticker"].isin(["A", "B"]))
    ]
    cap_q1 = weights[
        (weights["weighting"] == QuantileWeighting.MARKET_CAP)
        & (weights["quantile"] == "Q1")
        & (weights["ticker"].isin(["A", "B"]))
    ]
    assert equal_q1["ticker"].tolist() == ["A", "B"]
    assert equal_q1["weight"].tolist() == pytest.approx([0.5, 0.5])
    assert cap_q1["ticker"].tolist() == ["A", "B"]
    assert cap_q1["weight"].tolist() == pytest.approx([0.25, 0.75])

    quantile_weight_sums = (
        result.portfolio_weights.groupby(["weighting", "quantile"], observed=True)["weight"].sum().sort_index()
    )
    assert quantile_weight_sums.tolist() == pytest.approx([1.0] * len(quantile_weight_sums))
    assert set(weights["ticker"]) == {"A", "B", "C", "D", "E", "F"}
    assert not set(weights["ticker"]).intersection({"X", "Y", "Z"})

    equal_rows = result.monthly_returns[
        result.monthly_returns["weighting"] == QuantileWeighting.EQUAL
    ].set_index("portfolio")
    assert equal_rows.loc["Q1", "return"] == pytest.approx(0.15)
    assert equal_rows.loc["Q5", "return"] == pytest.approx(0.6)
    assert equal_rows.loc["high_minus_low", "return"] == pytest.approx(0.45)
    assert equal_rows.loc["preferred_minus_avoided", "return"] == pytest.approx(0.45)
    assert equal_rows.loc["high_minus_low", "constituent_count"] == 3


def test_evaluate_factor_quantiles_raises_unavailable_for_insufficient_monthly_dates() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    with pytest.raises(Emp008FactorQuantilesUnavailableError, match="at least two monthly dates"):
        evaluate_factor_quantiles(
            factors={"high_factor": factors["high_factor"]},
            directions={"high_factor": FactorDirection.HIGH},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=monthly_dates[:1],
            start="2024-01-31",
            end="2024-01-31",
            q=5,
        )


def test_evaluate_factor_quantiles_raises_unavailable_when_no_quantile_observations_exist() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    with pytest.raises(Emp008FactorQuantilesUnavailableError, match="no factor quantile observations"):
        evaluate_factor_quantiles(
            factors={"high_factor": factors["high_factor"]},
            directions={"high_factor": FactorDirection.HIGH},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=monthly_dates,
            start="2025-01-31",
            end="2025-02-28",
            q=5,
        )


def test_summarize_monthly_returns_computes_compounded_metrics() -> None:
    summary = summarize_monthly_returns(pd.Series([0.01] * 12, dtype=float))

    assert summary["observations"] == 12
    assert summary["annualized_return"] == pytest.approx((1.01**12) - 1.0)
    assert summary["annualized_volatility"] == pytest.approx(0.0)
    assert summary["sharpe"] == pytest.approx(0.0)
    assert summary["max_drawdown"] == pytest.approx(0.0)
    assert summary["positive_month_rate"] == pytest.approx(1.0)
    assert summary["mean_monthly_return"] == pytest.approx(0.01)


def test_summarize_monthly_returns_handles_drawdown_volatility_and_empty_inputs() -> None:
    first_loss = summarize_monthly_returns(pd.Series([-0.10, 0.05], dtype=float))
    assert first_loss["max_drawdown"] == pytest.approx(-0.10)

    varying = summarize_monthly_returns(pd.Series([0.02, -0.01, 0.03], dtype=float))
    expected_vol = float(np.std([0.02, -0.01, 0.03], ddof=0) * np.sqrt(12.0))
    expected_sharpe = float((np.mean([0.02, -0.01, 0.03]) / np.std([0.02, -0.01, 0.03], ddof=0)) * np.sqrt(12.0))
    assert varying["annualized_volatility"] == pytest.approx(expected_vol)
    assert varying["sharpe"] == pytest.approx(expected_sharpe)

    constant = summarize_monthly_returns(pd.Series([0.03, 0.03], dtype=float))
    assert constant["sharpe"] == pytest.approx(0.0)

    empty = summarize_monthly_returns(pd.Series([np.nan, None], dtype=float))
    assert empty["observations"] == 0
    for key, value in empty.items():
        if key != "observations":
            assert pd.isna(value)


def test_evaluate_factor_quantiles_populates_cumulative_returns_and_summary_metrics() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    result = evaluate_factor_quantiles(
        factors={
            "high_factor": factors["high_factor"],
            "low_factor": factors["low_factor"],
        },
        directions={
            "high_factor": FactorDirection.HIGH,
            "low_factor": FactorDirection.LOW,
        },
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[:4],
        start="2024-02-29",
        end="2024-04-30",
        q=2,
    )

    cumulative = result.cumulative_returns
    assert {"factor", "weighting", "portfolio", "return_date", "cumulative_return"} <= set(cumulative.columns)
    q1_path = cumulative[
        (cumulative["factor"] == "high_factor")
        & (cumulative["weighting"] == QuantileWeighting.EQUAL)
        & (cumulative["portfolio"] == "Q1")
    ].sort_values("return_date", kind="mergesort")
    assert q1_path["cumulative_return"].tolist() == pytest.approx([0.2, 0.32, 0.3935564435564436])

    high_q1 = _summary_row(
        result.summary,
        factor="high_factor",
        weighting=QuantileWeighting.EQUAL,
        portfolio="Q1",
    )
    high_ic = result.rank_ic[result.rank_ic["factor"] == "high_factor"]
    assert high_q1["observations"] == 3
    assert high_q1["average_constituent_count"] == pytest.approx(11.0 / 3.0)
    assert high_q1["average_one_way_turnover"] == pytest.approx(0.125)
    assert high_q1["mean_rank_ic"] == pytest.approx(high_ic["rank_ic"].dropna().mean())
    assert high_q1["directional_mean_rank_ic"] == pytest.approx(high_ic["directional_rank_ic"].dropna().mean())
    expected_high_ir = (
        high_ic["directional_rank_ic"].dropna().mean()
        / high_ic["directional_rank_ic"].dropna().std(ddof=0)
        * np.sqrt(12.0)
    )
    assert high_q1["ic_information_ratio"] == pytest.approx(expected_high_ir)
    assert high_q1["ic_positive_rate"] == pytest.approx(high_ic["directional_rank_ic"].dropna().gt(0.0).mean())
    assert high_q1["quantile_monotonicity"] == pytest.approx(1.0)

    low_q1 = _summary_row(
        result.summary,
        factor="low_factor",
        weighting=QuantileWeighting.EQUAL,
        portfolio="Q1",
    )
    low_ic = result.rank_ic[result.rank_ic["factor"] == "low_factor"]
    assert low_q1["directional_mean_rank_ic"] == pytest.approx(low_ic["directional_rank_ic"].dropna().mean())
    assert low_q1["ic_positive_rate"] == pytest.approx(low_ic["directional_rank_ic"].dropna().gt(0.0).mean())
    assert low_q1["quantile_monotonicity"] == pytest.approx(-1.0)


def test_evaluate_factor_quantiles_builds_daily_cumulative_returns_from_fixed_share_nav() -> None:
    dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-15",
            "2024-02-28",
            "2024-02-29",
            "2024-03-15",
            "2024-03-29",
        ]
    )
    columns = ["A", "B"]
    factor = _frame(
        dates,
        columns,
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ],
    )
    close = _frame(
        dates,
        columns,
        [
            [10.0, 10.0],
            [12.0, 8.0],
            [np.nan, 9.0],
            [11.0, 20.0],
            [16.5, 15.0],
            [22.0, 10.0],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
        ],
    )
    universe = _frame(dates, columns, [[True, True]] * len(dates)).astype(bool)

    result = evaluate_factor_quantiles(
        factors={"daily_factor": factor},
        directions={"daily_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=tuple(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])),
        start="2024-02-29",
        end="2024-03-29",
        q=2,
    )

    daily = result.daily_cumulative_returns.sort_values(
        ["factor", "weighting", "portfolio", "date", "signal_date"],
        kind="mergesort",
    ).reset_index(drop=True)
    assert tuple(daily.columns) == (
        "signal_date",
        "date",
        "factor",
        "weighting",
        "portfolio",
        "cumulative_return",
    )
    assert not daily.duplicated(["date", "factor", "weighting", "portfolio"]).any()
    assert daily["cumulative_return"].map(np.isfinite).all()

    equal_daily = daily[
        (daily["factor"] == "daily_factor")
        & (daily["weighting"] == QuantileWeighting.EQUAL)
    ]
    q1 = equal_daily[equal_daily["portfolio"] == "Q1"].set_index("date").sort_index()
    q2 = equal_daily[equal_daily["portfolio"] == "Q2"].set_index("date").sort_index()
    spread = equal_daily[equal_daily["portfolio"] == "high_minus_low"].set_index("date").sort_index()
    preferred = equal_daily[equal_daily["portfolio"] == "preferred_minus_avoided"].set_index("date").sort_index()

    assert q1.index.tolist() == pd.to_datetime(
        ["2024-01-31", "2024-02-15", "2024-02-28", "2024-02-29", "2024-03-15", "2024-03-29"]
    ).tolist()
    assert q1.loc[pd.Timestamp("2024-01-31"), "cumulative_return"] == pytest.approx(0.0)
    assert q1.loc[pd.Timestamp("2024-02-15"), "cumulative_return"] == pytest.approx(0.2)
    assert q1.loc[pd.Timestamp("2024-02-28"), "cumulative_return"] == pytest.approx(0.2)
    assert q1.loc[pd.Timestamp("2024-02-29"), "cumulative_return"] == pytest.approx(0.1)
    assert q1.loc[pd.Timestamp("2024-03-15"), "cumulative_return"] == pytest.approx(0.65)
    assert q1.loc[pd.Timestamp("2024-03-29"), "cumulative_return"] == pytest.approx(1.2)

    assert q2.loc[pd.Timestamp("2024-02-15"), "cumulative_return"] == pytest.approx(-0.2)
    assert q2.loc[pd.Timestamp("2024-02-28"), "cumulative_return"] == pytest.approx(-0.1)
    assert q2.loc[pd.Timestamp("2024-02-29"), "cumulative_return"] == pytest.approx(1.0)
    assert q2.loc[pd.Timestamp("2024-03-15"), "cumulative_return"] == pytest.approx(0.5)
    assert q2.loc[pd.Timestamp("2024-03-29"), "cumulative_return"] == pytest.approx(0.0)

    assert spread.loc[pd.Timestamp("2024-02-15"), "cumulative_return"] == pytest.approx(-0.4)
    assert spread.loc[pd.Timestamp("2024-02-28"), "cumulative_return"] == pytest.approx(-0.3)
    assert spread.loc[pd.Timestamp("2024-02-29"), "cumulative_return"] == pytest.approx(0.9)
    assert spread.loc[pd.Timestamp("2024-03-15"), "cumulative_return"] == pytest.approx(-0.525)
    assert spread.loc[pd.Timestamp("2024-03-29"), "cumulative_return"] == pytest.approx(-1.95)
    pd.testing.assert_series_equal(
        spread["cumulative_return"],
        preferred["cumulative_return"],
        check_names=False,
    )

    monthly_endpoints = (
        daily[daily["date"].isin(result.cumulative_returns["return_date"])]
        .sort_values(["factor", "weighting", "portfolio", "date"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, ["factor", "weighting", "portfolio", "date", "cumulative_return"]]
        .rename(columns={"date": "return_date"})
    )
    expected_endpoints = result.cumulative_returns.sort_values(
        ["factor", "weighting", "portfolio", "return_date"],
        kind="mergesort",
    ).reset_index(drop=True).loc[:, ["factor", "weighting", "portfolio", "return_date", "cumulative_return"]]
    pd.testing.assert_frame_equal(monthly_endpoints, expected_endpoints)

    boundary_rows = daily[daily["date"] == pd.Timestamp("2024-02-29")]
    portfolios_per_weighting = result.monthly_returns["portfolio"].nunique()
    assert len(boundary_rows) == len(list(QuantileWeighting)) * portfolios_per_weighting


def test_evaluate_factor_quantiles_builds_low_direction_daily_spreads_with_reversed_preference() -> None:
    dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-15",
            "2024-02-28",
            "2024-02-29",
            "2024-03-15",
            "2024-03-29",
        ]
    )
    columns = ["A", "B"]
    factor = _frame(
        dates,
        columns,
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ],
    )
    close = _frame(
        dates,
        columns,
        [
            [10.0, 10.0],
            [12.0, 8.0],
            [np.nan, 9.0],
            [11.0, 20.0],
            [16.5, 15.0],
            [22.0, 10.0],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
        ],
    )
    universe = _frame(dates, columns, [[True, True]] * len(dates)).astype(bool)

    result = evaluate_factor_quantiles(
        factors={"daily_factor_low": factor},
        directions={"daily_factor_low": FactorDirection.LOW},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=tuple(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])),
        start="2024-02-29",
        end="2024-03-29",
        q=2,
    )

    equal_daily = result.daily_cumulative_returns[
        (result.daily_cumulative_returns["factor"] == "daily_factor_low")
        & (result.daily_cumulative_returns["weighting"] == QuantileWeighting.EQUAL)
    ]
    spread = equal_daily[equal_daily["portfolio"] == "high_minus_low"].set_index("date").sort_index()
    preferred = equal_daily[equal_daily["portfolio"] == "preferred_minus_avoided"].set_index("date").sort_index()

    first_period_dates = pd.to_datetime(["2024-01-31", "2024-02-15", "2024-02-28", "2024-02-29"])
    second_period_dates = pd.to_datetime(["2024-02-29", "2024-03-15", "2024-03-29"])
    first_period_spread = spread.loc[first_period_dates, "cumulative_return"] - spread.loc[pd.Timestamp("2024-01-31"), "cumulative_return"]
    first_period_preferred = preferred.loc[first_period_dates, "cumulative_return"] - preferred.loc[pd.Timestamp("2024-01-31"), "cumulative_return"]
    second_period_spread = (
        (1.0 + spread.loc[second_period_dates, "cumulative_return"])
        / (1.0 + spread.loc[pd.Timestamp("2024-02-29"), "cumulative_return"])
        - 1.0
    )
    second_period_preferred = (
        (1.0 + preferred.loc[second_period_dates, "cumulative_return"])
        / (1.0 + preferred.loc[pd.Timestamp("2024-02-29"), "cumulative_return"])
        - 1.0
    )
    pd.testing.assert_series_equal(first_period_preferred, -first_period_spread, check_names=False)
    pd.testing.assert_series_equal(second_period_preferred, -second_period_spread, check_names=False)

    assert spread.loc[pd.Timestamp("2024-03-15"), "cumulative_return"] == pytest.approx(-0.525)
    assert spread.loc[pd.Timestamp("2024-03-29"), "cumulative_return"] == pytest.approx(-1.95)
    assert preferred.loc[pd.Timestamp("2024-03-15"), "cumulative_return"] == pytest.approx(-0.825)
    assert preferred.loc[pd.Timestamp("2024-03-29"), "cumulative_return"] == pytest.approx(-0.75)

    monthly_endpoints = (
        result.daily_cumulative_returns[
            (result.daily_cumulative_returns["factor"] == "daily_factor_low")
            & (result.daily_cumulative_returns["date"].isin(result.cumulative_returns["return_date"]))
        ]
        .sort_values(["weighting", "portfolio", "date"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, ["weighting", "portfolio", "date", "cumulative_return"]]
        .rename(columns={"date": "return_date"})
    )
    expected_endpoints = (
        result.cumulative_returns[result.cumulative_returns["factor"] == "daily_factor_low"]
        .sort_values(["weighting", "portfolio", "return_date"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, ["weighting", "portfolio", "return_date", "cumulative_return"]]
    )
    pd.testing.assert_frame_equal(monthly_endpoints, expected_endpoints)

    monthly_low = result.cumulative_returns[
        (result.cumulative_returns["factor"] == "daily_factor_low")
        & (result.cumulative_returns["weighting"] == QuantileWeighting.EQUAL)
        & (result.cumulative_returns["portfolio"] == "preferred_minus_avoided")
    ].sort_values("return_date", kind="mergesort")
    assert monthly_low["cumulative_return"].tolist() == pytest.approx([-0.9, -0.75])


def test_evaluate_factor_quantiles_computes_turnover_for_buckets_and_spreads() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"])
    columns = ["A", "B", "C", "D"]
    factor = _frame(
        dates,
        columns,
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 3.0, 2.0, 4.0],
            [3.0, 1.0, 4.0, 2.0],
            [2.0, 4.0, 1.0, 3.0],
        ],
    )
    close = _frame(
        dates,
        columns,
        [
            [10.0, 10.0, 10.0, 10.0],
            [11.0, 12.0, 13.0, 14.0],
            [12.1, 13.2, 14.3, 15.4],
            [13.31, 14.52, 15.73, 16.94],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [[10.0, 20.0, 30.0, 40.0]] * len(dates),
    )
    universe = _frame(dates, columns, [[True, True, True, True]] * len(dates)).astype(bool)

    result = evaluate_factor_quantiles(
        factors={"turnover_factor": factor},
        directions={"turnover_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=tuple(dates),
        start="2024-02-29",
        end="2024-04-30",
        q=2,
    )

    q1_row = _summary_row(
        result.summary,
        factor="turnover_factor",
        weighting=QuantileWeighting.EQUAL,
        portfolio="Q1",
    )
    spread_row = _summary_row(
        result.summary,
        factor="turnover_factor",
        weighting=QuantileWeighting.EQUAL,
        portfolio="high_minus_low",
    )
    assert q1_row["average_one_way_turnover"] == pytest.approx(0.75)
    assert spread_row["average_one_way_turnover"] == pytest.approx(1.5)


def test_write_outputs_creates_auditable_artifacts_and_rejects_invalid_results(tmp_path) -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()
    result = evaluate_factor_quantiles(
        factors={
            "price_to_252d_high": factors["high_factor"],
            "ln_market_cap": factors["low_factor"],
        },
        directions={
            "price_to_252d_high": FactorDirection.HIGH,
            "ln_market_cap": FactorDirection.LOW,
        },
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[:4],
        start="2024-02-29",
        end="2024-04-30",
        q=2,
    )

    payload = result.write_outputs(tmp_path, factor_set="mfbt", q=2)

    for name in [
        "monthly_returns.csv",
        "monthly_returns.parquet",
        "portfolio_weights.parquet",
        "rank_ic.csv",
        "rank_ic.parquet",
        "cumulative_returns.csv",
        "daily_cumulative_returns.csv",
        "daily_cumulative_returns.parquet",
        "cumulative_quintiles_equal_weight.png",
        "cumulative_quintiles_market_cap_weight.png",
        "summary.csv",
        "summary.json",
        "manifest.json",
    ]:
        assert (tmp_path / name).exists()

    assert {
        "monthly_returns_parquet",
        "portfolio_weights_parquet",
        "rank_ic_parquet",
        "daily_cumulative_returns_csv",
        "daily_cumulative_returns_parquet",
        "cumulative_quintiles_equal_weight_png",
        "cumulative_quintiles_market_cap_weight_png",
        "summary_json",
        "manifest_json",
    } <= set(payload)
    assert (tmp_path / "cumulative_quintiles_equal_weight.png").stat().st_size > 0
    assert (tmp_path / "cumulative_quintiles_market_cap_weight.png").stat().st_size > 0
    summary_json = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary_json, list)
    assert summary_json
    assert summary_json[0]["weighting"] in {mode.value for mode in QuantileWeighting}
    assert summary_json[0]["weighting"] == "equal_weight"
    assert summary_json[0]["portfolio"]
    assert summary_json[0]["observations"] >= 1

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["factor_set"] == "mfbt"
    assert manifest["weighting_modes"] == ["equal_weight", "market_cap_weight"]
    assert manifest["q"] == 2
    assert manifest["timing"] == "month_end_t_to_next_month_end"
    assert manifest["rebalance_frequency"] == "monthly"
    assert manifest["nav_frequency"] == "daily"
    assert manifest["market_cap_field"] == "market_cap"
    assert manifest["selected_factors"] == ["price_to_252d_high", "earnings_momentum", "dividend_yield_ttm", "retail_flow", "value", "ln_market_cap"]
    assert manifest["directions"]["ln_market_cap"] == "low"
    assert manifest["artifacts"]["summary.json"]["rows"] == len(result.summary)
    assert manifest["artifacts"]["daily_cumulative_returns.csv"]["rows"] == len(result.daily_cumulative_returns)
    assert manifest["artifacts"]["daily_cumulative_returns.parquet"]["rows"] == len(result.daily_cumulative_returns)
    assert manifest["artifacts"]["cumulative_quintiles_equal_weight.png"]["rows"] == 2
    assert manifest["artifacts"]["cumulative_quintiles_market_cap_weight.png"]["rows"] == 2

    invalid_dir = tmp_path / "invalid"
    invalid_monthly = result.monthly_returns.copy()
    invalid_monthly.loc[
        invalid_monthly["portfolio"].eq("Q1"),
        "return",
    ] = float("inf")
    invalid = Emp008FactorQuantileResult(
        monthly_returns=invalid_monthly,
        portfolio_weights=result.portfolio_weights,
        rank_ic=result.rank_ic,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    with pytest.raises(ValueError, match="finite"):
        invalid.write_outputs(invalid_dir, factor_set="mfbt", q=2)
    assert not invalid_dir.exists() or list(invalid_dir.iterdir()) == []


def test_build_cumulative_quintile_figure_creates_one_subplot_per_factor() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()
    result = evaluate_factor_quantiles(
        factors={
            "price_to_252d_high": factors["high_factor"],
            "ln_market_cap": factors["low_factor"],
            "momentum_12_1m": factors["sparse_factor"],
        },
        directions={
            "price_to_252d_high": FactorDirection.HIGH,
            "ln_market_cap": FactorDirection.LOW,
            "momentum_12_1m": FactorDirection.HIGH,
        },
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[:4],
        start="2024-02-29",
        end="2024-04-30",
        q=2,
    )

    figure = _build_cumulative_quintile_figure(
        cumulative_returns=result.daily_cumulative_returns,
        directions={
            "price_to_252d_high": FactorDirection.HIGH,
            "ln_market_cap": FactorDirection.LOW,
            "momentum_12_1m": FactorDirection.HIGH,
        },
        weighting=QuantileWeighting.EQUAL,
        q=2,
    )
    try:
        assert len(figure.axes) == 4
        titled_axes = [axis for axis in figure.axes if axis.get_title()]
        assert [axis.get_title() for axis in titled_axes] == [
            "price_to_252d_high",
            "ln_market_cap",
            "momentum_12_1m",
        ]
        first_line = titled_axes[0].lines[0]
        assert len(first_line.get_xdata()) > len(result.cumulative_returns["return_date"].unique())
    finally:
        figure.clf()


def test_write_outputs_rejects_tampered_cumulative_summary_and_rank_ic(tmp_path) -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()
    result = evaluate_factor_quantiles(
        factors={
            "price_to_252d_high": factors["high_factor"],
            "ln_market_cap": factors["low_factor"],
        },
        directions={
            "price_to_252d_high": FactorDirection.HIGH,
            "ln_market_cap": FactorDirection.LOW,
        },
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[:4],
        start="2024-02-29",
        end="2024-04-30",
        q=2,
    )

    bad_cumulative = result.cumulative_returns.copy()
    bad_cumulative.loc[0, "cumulative_return"] = 999.0
    cumulative_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=result.rank_ic,
        cumulative_returns=bad_cumulative,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    bad_cumulative_dir = tmp_path / "bad_cumulative"
    with pytest.raises(ValueError, match="cumulative_returns"):
        cumulative_result.write_outputs(bad_cumulative_dir, factor_set="mfbt", q=2)
    assert not bad_cumulative_dir.exists() or list(bad_cumulative_dir.iterdir()) == []

    bad_summary = result.summary.copy()
    bad_summary.loc[0, "annualized_return"] = 999.0
    summary_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=result.rank_ic,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=bad_summary,
    )
    bad_summary_dir = tmp_path / "bad_summary"
    with pytest.raises(ValueError, match="summary"):
        summary_result.write_outputs(bad_summary_dir, factor_set="mfbt", q=2)
    assert not bad_summary_dir.exists() or list(bad_summary_dir.iterdir()) == []

    stale_rank_ic = result.rank_ic.copy()
    stale_rank_ic.loc[0, "return_date"] = pd.Timestamp("2024-05-31")
    stale_rank_ic_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=stale_rank_ic,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    stale_rank_ic_dir = tmp_path / "stale_rank_ic"
    with pytest.raises(ValueError, match="rank_ic"):
        stale_rank_ic_result.write_outputs(stale_rank_ic_dir, factor_set="mfbt", q=2)
    assert not stale_rank_ic_dir.exists() or list(stale_rank_ic_dir.iterdir()) == []

    extra_rank_ic = pd.concat([result.rank_ic, result.rank_ic.iloc[[0]]], ignore_index=True)
    extra_rank_ic_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=extra_rank_ic,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    extra_rank_ic_dir = tmp_path / "extra_rank_ic"
    with pytest.raises(ValueError, match="rank_ic"):
        extra_rank_ic_result.write_outputs(extra_rank_ic_dir, factor_set="mfbt", q=2)
    assert not extra_rank_ic_dir.exists() or list(extra_rank_ic_dir.iterdir()) == []

    missing_rank_ic = result.rank_ic.iloc[1:].reset_index(drop=True)
    missing_rank_ic_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=missing_rank_ic,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    missing_rank_ic_dir = tmp_path / "missing_rank_ic"
    with pytest.raises(ValueError, match="rank_ic"):
        missing_rank_ic_result.write_outputs(missing_rank_ic_dir, factor_set="mfbt", q=2)
    assert not missing_rank_ic_dir.exists() or list(missing_rank_ic_dir.iterdir()) == []

    wrong_directional = result.rank_ic.copy()
    wrong_directional.loc[wrong_directional["factor"] == "ln_market_cap", "directional_rank_ic"] = wrong_directional.loc[
        wrong_directional["factor"] == "ln_market_cap",
        "rank_ic",
    ]
    wrong_directional_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=wrong_directional,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    wrong_directional_dir = tmp_path / "wrong_directional"
    with pytest.raises(ValueError, match="directional_rank_ic"):
        wrong_directional_result.write_outputs(wrong_directional_dir, factor_set="mfbt", q=2)
    assert not wrong_directional_dir.exists() or list(wrong_directional_dir.iterdir()) == []

    wrong_n_obs = result.rank_ic.copy()
    wrong_n_obs.loc[0, "n_obs"] = wrong_n_obs.loc[0, "n_obs"] + 1
    wrong_n_obs_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=wrong_n_obs,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    wrong_n_obs_dir = tmp_path / "wrong_n_obs"
    with pytest.raises(ValueError, match="n_obs"):
        wrong_n_obs_result.write_outputs(wrong_n_obs_dir, factor_set="mfbt", q=2)
    assert not wrong_n_obs_dir.exists() or list(wrong_n_obs_dir.iterdir()) == []

    out_of_range_ic = result.rank_ic.copy()
    out_of_range_ic.loc[0, "rank_ic"] = 1.5
    out_of_range_ic.loc[0, "directional_rank_ic"] = 1.5
    out_of_range_ic_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=out_of_range_ic,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    out_of_range_ic_dir = tmp_path / "out_of_range_ic"
    with pytest.raises(ValueError, match="rank_ic"):
        out_of_range_ic_result.write_outputs(out_of_range_ic_dir, factor_set="mfbt", q=2)
    assert not out_of_range_ic_dir.exists() or list(out_of_range_ic_dir.iterdir()) == []

    small_n_obs_rank_ic = result.rank_ic.copy()
    small_n_obs_rank_ic.loc[0, "n_obs"] = 1
    small_n_obs_rank_ic.loc[0, "rank_ic"] = 0.25
    small_n_obs_rank_ic.loc[0, "directional_rank_ic"] = 0.25
    small_n_obs_rank_ic_result = Emp008FactorQuantileResult(
        monthly_returns=result.monthly_returns,
        portfolio_weights=result.portfolio_weights,
        rank_ic=small_n_obs_rank_ic,
        cumulative_returns=result.cumulative_returns,
        daily_cumulative_returns=result.daily_cumulative_returns,
        summary=result.summary,
    )
    small_n_obs_rank_ic_dir = tmp_path / "small_n_obs_rank_ic"
    with pytest.raises(ValueError, match="n_obs"):
        small_n_obs_rank_ic_result.write_outputs(small_n_obs_rank_ic_dir, factor_set="mfbt", q=2)
    assert not small_n_obs_rank_ic_dir.exists() or list(small_n_obs_rank_ic_dir.iterdir()) == []


def test_write_outputs_rolls_back_on_publication_failure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()
    result = evaluate_factor_quantiles(
        factors={
            "price_to_252d_high": factors["high_factor"],
            "ln_market_cap": factors["low_factor"],
        },
        directions={
            "price_to_252d_high": FactorDirection.HIGH,
            "ln_market_cap": FactorDirection.LOW,
        },
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[:4],
        start="2024-02-29",
        end="2024-04-30",
        q=2,
    )
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("leave me", encoding="utf-8")

    original_replace = Path.replace
    calls: list[str] = []

    def failing_replace(self: Path, target: Path) -> Path:
        calls.append(target.name)
        if target.name == "rank_ic.csv":
            raise OSError("simulated publish failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated publish failure"):
        result.write_outputs(output_dir, factor_set="mfbt", q=2)

    assert calls
    assert sentinel.read_text(encoding="utf-8") == "leave me"
    assert sorted(path.name for path in output_dir.iterdir()) == ["keep.txt"]


def test_evaluate_factor_quantiles_handles_low_direction_sparse_months_and_rank_ic() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    result = evaluate_factor_quantiles(
        factors={
            "high_factor": factors["high_factor"],
            "low_factor": factors["low_factor"],
            "sparse_factor": factors["sparse_factor"],
        },
        directions={
            "high_factor": FactorDirection.HIGH,
            "low_factor": FactorDirection.LOW,
            "sparse_factor": FactorDirection.HIGH,
        },
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[:3],
        start="2024-02-29",
        end="2024-03-29",
        q=5,
    )

    low_equal = result.monthly_returns[
        (result.monthly_returns["factor"] == "low_factor")
        & (result.monthly_returns["weighting"] == QuantileWeighting.EQUAL)
        & (result.monthly_returns["return_date"] == pd.Timestamp("2024-02-29"))
    ].set_index("portfolio")
    assert low_equal.loc["high_minus_low", "return"] == pytest.approx(0.45)
    assert low_equal.loc["preferred_minus_avoided", "return"] == pytest.approx(-0.45)

    rank_ic = result.rank_ic.set_index(["factor", "return_date"]).sort_index()
    assert rank_ic.loc[("high_factor", pd.Timestamp("2024-02-29")), "rank_ic"] > 0.0
    assert rank_ic.loc[("high_factor", pd.Timestamp("2024-02-29")), "directional_rank_ic"] > 0.0
    assert rank_ic.loc[("low_factor", pd.Timestamp("2024-02-29")), "rank_ic"] > 0.0
    assert rank_ic.loc[("low_factor", pd.Timestamp("2024-02-29")), "directional_rank_ic"] < 0.0
    assert rank_ic.loc[("low_factor", pd.Timestamp("2024-02-29")), "n_obs"] == 6

    sparse_weights = result.portfolio_weights[
        (result.portfolio_weights["factor"] == "sparse_factor")
        & (result.portfolio_weights["return_date"] == pd.Timestamp("2024-03-29"))
        & (result.portfolio_weights["weighting"] == QuantileWeighting.EQUAL)
    ]
    assert set(sparse_weights["quantile"]) == {"Q1", "Q2", "Q3"}
    assert "Q5" not in set(sparse_weights["quantile"])
    sparse_returns = result.monthly_returns[
        (result.monthly_returns["factor"] == "sparse_factor")
        & (result.monthly_returns["return_date"] == pd.Timestamp("2024-03-29"))
    ].set_index(["weighting", "portfolio"]).sort_index()
    for weighting in QuantileWeighting:
        assert sparse_returns.loc[(weighting, "Q1"), "constituent_count"] == 1
        assert sparse_returns.loc[(weighting, "Q2"), "constituent_count"] == 1
        assert sparse_returns.loc[(weighting, "Q3"), "constituent_count"] == 1
        assert sparse_returns.loc[(weighting, "Q4"), "constituent_count"] == 0
        assert sparse_returns.loc[(weighting, "Q5"), "constituent_count"] == 0
        assert pd.isna(sparse_returns.loc[(weighting, "Q4"), "return"])
        assert pd.isna(sparse_returns.loc[(weighting, "Q5"), "return"])
        assert pd.isna(sparse_returns.loc[(weighting, "high_minus_low"), "return"])
        assert pd.isna(sparse_returns.loc[(weighting, "preferred_minus_avoided"), "return"])
        assert sparse_returns.loc[(weighting, "high_minus_low"), "constituent_count"] == 1
        assert sparse_returns.loc[(weighting, "preferred_minus_avoided"), "constituent_count"] == 1
    assert pd.Timestamp("2024-02-29") in set(
        result.monthly_returns.loc[result.monthly_returns["factor"] == "high_factor", "return_date"]
    )


def test_evaluate_factor_quantiles_excludes_invalid_market_cap_tickers_from_both_modes() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()
    market_cap.loc[pd.Timestamp("2024-01-31"), "Y"] = 0.0

    result = evaluate_factor_quantiles(
        factors={"high_factor": factors["high_factor"]},
        directions={"high_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[:2],
        start="2024-02-29",
        end="2024-02-29",
        q=3,
    )

    weights = result.portfolio_weights.sort_values(["weighting", "quantile", "ticker"], kind="mergesort").reset_index(drop=True)
    equal_membership = weights.loc[weights["weighting"] == QuantileWeighting.EQUAL, ["quantile", "ticker"]].reset_index(drop=True)
    cap_membership = weights.loc[weights["weighting"] == QuantileWeighting.MARKET_CAP, ["quantile", "ticker"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(equal_membership, cap_membership)
    assert "Y" not in set(weights["ticker"])
    assert set(weights["ticker"]) == {"A", "B", "C", "D", "E", "F"}

    monthly = result.monthly_returns.set_index(["weighting", "portfolio"]).sort_index()
    assert monthly.loc[(QuantileWeighting.EQUAL, "Q3"), "constituent_count"] == 2
    assert monthly.loc[(QuantileWeighting.MARKET_CAP, "Q3"), "constituent_count"] == 2


def test_evaluate_factor_quantiles_emits_all_quantile_rows_when_names_are_fewer_than_q() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    result = evaluate_factor_quantiles(
        factors={"sparse_factor": factors["sparse_factor"]},
        directions={"sparse_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[1:3],
        start="2024-03-29",
        end="2024-03-29",
        q=5,
    )

    monthly = result.monthly_returns.set_index(["weighting", "portfolio"]).sort_index()
    for weighting in QuantileWeighting:
        assert list(
            result.monthly_returns.loc[result.monthly_returns["weighting"] == weighting, "portfolio"]
        ) == ["Q1", "Q2", "Q3", "Q4", "Q5", "high_minus_low", "preferred_minus_avoided"]
        assert monthly.loc[(weighting, "Q4"), "constituent_count"] == 0
        assert monthly.loc[(weighting, "Q5"), "constituent_count"] == 0
        assert pd.isna(monthly.loc[(weighting, "Q4"), "return"])
        assert pd.isna(monthly.loc[(weighting, "Q5"), "return"])

    cumulative = result.cumulative_returns.set_index(["weighting", "portfolio"]).sort_index()
    assert pd.isna(cumulative.loc[(QuantileWeighting.EQUAL, "Q4"), "cumulative_return"])
    assert pd.isna(cumulative.loc[(QuantileWeighting.MARKET_CAP, "Q5"), "cumulative_return"])


def test_evaluate_factor_quantiles_preserves_sparse_daily_endpoint_coverage() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    result = evaluate_factor_quantiles(
        factors={"sparse_factor": factors["sparse_factor"]},
        directions={"sparse_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates[1:3],
        start="2024-03-29",
        end="2024-03-29",
        q=5,
    )

    daily_endpoint = (
        result.daily_cumulative_returns[
            (result.daily_cumulative_returns["factor"] == "sparse_factor")
            & (result.daily_cumulative_returns["date"] == pd.Timestamp("2024-03-29"))
        ]
        .sort_values(["weighting", "portfolio"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, ["weighting", "portfolio", "cumulative_return"]]
    )
    cumulative_endpoint = (
        result.cumulative_returns[
            (result.cumulative_returns["factor"] == "sparse_factor")
            & (result.cumulative_returns["return_date"] == pd.Timestamp("2024-03-29"))
        ]
        .sort_values(["weighting", "portfolio"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, ["weighting", "portfolio", "cumulative_return"]]
    )
    pd.testing.assert_frame_equal(daily_endpoint, cumulative_endpoint)

    sparse_daily = result.daily_cumulative_returns[
        (result.daily_cumulative_returns["factor"] == "sparse_factor")
        & (result.daily_cumulative_returns["weighting"] == QuantileWeighting.EQUAL)
        & (result.daily_cumulative_returns["date"] == pd.Timestamp("2024-03-29"))
    ].set_index("portfolio").sort_index()
    assert pd.isna(sparse_daily.loc["Q4", "cumulative_return"])
    assert pd.isna(sparse_daily.loc["Q5", "cumulative_return"])
    assert pd.isna(sparse_daily.loc["high_minus_low", "cumulative_return"])
    assert pd.isna(sparse_daily.loc["preferred_minus_avoided", "cumulative_return"])


def test_evaluate_factor_quantiles_filters_on_return_date_and_uses_consecutive_month_pairs() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    result = evaluate_factor_quantiles(
        factors={"high_factor": factors["high_factor"]},
        directions={"high_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=monthly_dates,
        start="2024-03-01",
        end="2024-03-29",
        q=5,
    )

    assert set(result.monthly_returns["signal_date"]) == {pd.Timestamp("2024-02-29")}
    assert set(result.monthly_returns["return_date"]) == {pd.Timestamp("2024-03-29")}
    assert set(result.portfolio_weights["signal_date"]) == {pd.Timestamp("2024-02-29")}
    assert set(result.portfolio_weights["return_date"]) == {pd.Timestamp("2024-03-29")}


def test_evaluate_factor_quantiles_excludes_nonfinite_inputs_from_both_weightings() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    columns = ["A", "B", "C", "D", "E", "F", "G"]
    factors = {
        "high_factor": _frame(
            dates,
            columns,
            [
                [1.0, 2.0, float("inf"), float("-inf"), 5.0, 6.0, 7.0],
                [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            ],
        )
    }
    close = _frame(
        dates,
        columns,
        [
            [10.0, float("inf"), 12.0, 13.0, 14.0, 15.0, 16.0],
            [11.0, 12.0, 13.0, float("inf"), 15.0, 16.0, 17.0],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [100.0, 200.0, 300.0, 400.0, float("inf"), float("-inf"), 700.0],
            [101.0, 201.0, 301.0, 401.0, 501.0, 601.0, 701.0],
        ],
    )
    universe = _frame(dates, columns, [[True] * len(columns), [True] * len(columns)]).astype(bool)

    result = evaluate_factor_quantiles(
        factors=factors,
        directions={"high_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=tuple(dates),
        start="2024-02-29",
        end="2024-02-29",
        q=2,
    )

    assert set(result.portfolio_weights["ticker"]) == {"A", "G"}
    for weighting in QuantileWeighting:
        weighted = result.portfolio_weights[result.portfolio_weights["weighting"] == weighting]
        assert weighted["weight"].map(np.isfinite).all()
        returns = result.monthly_returns[result.monthly_returns["weighting"] == weighting]
        assert returns["return"].dropna().map(np.isfinite).all()
    assert result.portfolio_weights[result.portfolio_weights["weighting"] == QuantileWeighting.EQUAL][
        "ticker"
    ].tolist() == ["A", "G"]
    assert result.portfolio_weights[result.portfolio_weights["weighting"] == QuantileWeighting.MARKET_CAP][
        "ticker"
    ].tolist() == ["A", "G"]


def test_evaluate_factor_quantiles_supports_unique_integer_tickers() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    columns = [101, 102, 103, 104]
    factors = {
        "high_factor": _frame(
            dates,
            columns,
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.5, 2.5, 3.5, 4.5],
            ],
        )
    }
    close = _frame(
        dates,
        columns,
        [
            [10.0, 10.0, 10.0, 10.0],
            [11.0, 12.0, 13.0, 14.0],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
    )
    universe = _frame(dates, columns, [[True] * len(columns), [True] * len(columns)]).astype(bool)

    result = evaluate_factor_quantiles(
        factors=factors,
        directions={"high_factor": FactorDirection.HIGH},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=tuple(dates),
        start="2024-02-29",
        end="2024-02-29",
        q=2,
    )

    weights = result.portfolio_weights.sort_values(["weighting", "quantile", "ticker"], kind="mergesort").reset_index(drop=True)
    assert set(weights["ticker"]) == {"101", "102", "103", "104"}
    equal_q1 = weights[
        (weights["weighting"] == QuantileWeighting.EQUAL)
        & (weights["quantile"] == "Q1")
    ]
    cap_q1 = weights[
        (weights["weighting"] == QuantileWeighting.MARKET_CAP)
        & (weights["quantile"] == "Q1")
    ]
    assert equal_q1["ticker"].tolist() == ["101", "102"]
    assert equal_q1["weight"].tolist() == pytest.approx([0.5, 0.5])
    assert cap_q1["weight"].tolist() == pytest.approx([0.25, 0.75])
    monthly = result.monthly_returns.set_index(["weighting", "portfolio"]).sort_index()
    assert monthly.loc[(QuantileWeighting.EQUAL, "Q1"), "return"] == pytest.approx(0.15)
    assert monthly.loc[(QuantileWeighting.MARKET_CAP, "Q1"), "return"] == pytest.approx(0.175)


def test_evaluate_factor_quantiles_rejects_duplicate_or_ambiguous_ticker_labels() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    duplicate_columns = ["A", "A", "B"]
    close_duplicate = _frame(
        dates,
        duplicate_columns,
        [
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0],
        ],
    )
    market_cap_duplicate = _frame(
        dates,
        duplicate_columns,
        [
            [100.0, 110.0, 120.0],
            [101.0, 111.0, 121.0],
        ],
    )
    universe_duplicate = _frame(dates, duplicate_columns, [[True, True, True], [True, True, True]]).astype(bool)
    factor_duplicate = _frame(
        dates,
        duplicate_columns,
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
        ],
    )

    with pytest.raises(ValueError, match="duplicate.*ticker"):
        evaluate_factor_quantiles(
            factors={"high_factor": factor_duplicate},
            directions={"high_factor": FactorDirection.HIGH},
            close=close_duplicate,
            market_cap=market_cap_duplicate,
            universe=universe_duplicate,
            monthly_dates=tuple(dates),
            start="2024-02-29",
            end="2024-02-29",
            q=2,
        )

    mixed_columns = [101, "101", 202]
    close_mixed = _frame(
        dates,
        mixed_columns,
        [
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0],
        ],
    )
    market_cap_mixed = _frame(
        dates,
        mixed_columns,
        [
            [100.0, 110.0, 120.0],
            [101.0, 111.0, 121.0],
        ],
    )
    universe_mixed = _frame(dates, mixed_columns, [[True, True, True], [True, True, True]]).astype(bool)
    factor_mixed = _frame(
        dates,
        mixed_columns,
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
        ],
    )

    with pytest.raises(ValueError, match="ambiguous.*101"):
        evaluate_factor_quantiles(
            factors={"high_factor": factor_mixed},
            directions={"high_factor": FactorDirection.HIGH},
            close=close_mixed,
            market_cap=market_cap_mixed,
            universe=universe_mixed,
            monthly_dates=tuple(dates),
            start="2024-02-29",
            end="2024-02-29",
            q=2,
        )


def test_evaluate_factor_quantiles_validates_inputs_and_reports_empty_results() -> None:
    factors, close, market_cap, universe, monthly_dates = _core_inputs()

    with pytest.raises(ValueError, match="q must be at least 2"):
        evaluate_factor_quantiles(
            factors={"high_factor": factors["high_factor"]},
            directions={"high_factor": FactorDirection.HIGH},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=monthly_dates[:2],
            start="2024-02-29",
            end="2024-02-29",
            q=1,
        )

    with pytest.raises(ValueError, match="missing directions.*high_factor"):
        evaluate_factor_quantiles(
            factors={"high_factor": factors["high_factor"]},
            directions={},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=monthly_dates[:2],
            start="2024-02-29",
            end="2024-02-29",
            q=5,
        )

    with pytest.raises(ValueError, match="at least two monthly dates"):
        evaluate_factor_quantiles(
            factors={"high_factor": factors["high_factor"]},
            directions={"high_factor": FactorDirection.HIGH},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=monthly_dates[:1],
            start="2024-01-31",
            end="2024-01-31",
            q=5,
        )

    with pytest.raises(ValueError, match="start.*must be on or before.*end"):
        evaluate_factor_quantiles(
            factors={"high_factor": factors["high_factor"]},
            directions={"high_factor": FactorDirection.HIGH},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=monthly_dates[:2],
            start="2024-03-01",
            end="2024-02-29",
            q=5,
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_factor_quantiles(
            factors={"high_factor": factors["high_factor"]},
            directions={"high_factor": FactorDirection.HIGH},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=(monthly_dates[1], monthly_dates[0]),
            start="2024-02-29",
            end="2024-02-29",
            q=5,
        )

    with pytest.raises(ValueError, match="duplicate.*monthly dates"):
        evaluate_factor_quantiles(
            factors={"high_factor": factors["high_factor"]},
            directions={"high_factor": FactorDirection.HIGH},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=(monthly_dates[0], monthly_dates[0]),
            start="2024-01-31",
            end="2024-01-31",
            q=5,
        )

    empty_factor = factors["high_factor"] * float("nan")
    with pytest.raises(ValueError, match="no factor quantile observations"):
        evaluate_factor_quantiles(
            factors={"high_factor": empty_factor},
            directions={"high_factor": FactorDirection.HIGH},
            close=close,
            market_cap=market_cap,
            universe=universe,
            monthly_dates=monthly_dates[:2],
            start="2024-02-29",
            end="2024-02-29",
            q=5,
        )


def test_evaluate_factor_quantiles_emits_nan_rank_ic_for_insufficient_or_constant_cases() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    columns = ["A", "B", "C"]
    factors = {
        "valid_factor": _frame(
            dates,
            columns,
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
            ],
        ),
        "one_obs_factor": _frame(
            dates,
            columns,
            [
                [1.0, None, None],
                [2.0, None, None],
                [3.0, None, None],
            ],
        ),
        "constant_signal_factor": _frame(
            dates,
            columns,
            [
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0],
            ],
        ),
        "constant_return_factor": _frame(
            dates,
            columns,
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
            ],
        ),
    }
    close = _frame(
        dates,
        columns,
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [21.0, 22.0, 23.0],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [100.0, 200.0, 300.0],
            [110.0, 210.0, 310.0],
            [120.0, 220.0, 320.0],
        ],
    )
    universe = _frame(dates, columns, [[True, True, True]] * len(dates)).astype(bool)

    result = evaluate_factor_quantiles(
        factors=factors,
        directions={name: FactorDirection.HIGH for name in factors},
        close=close,
        market_cap=market_cap,
        universe=universe,
        monthly_dates=tuple(dates),
        start="2024-02-29",
        end="2024-03-29",
        q=3,
    )

    rank_ic = result.rank_ic.set_index(["factor", "return_date"]).sort_index()
    for factor_name, return_date, n_obs in [
        ("one_obs_factor", pd.Timestamp("2024-02-29"), 1),
        ("one_obs_factor", pd.Timestamp("2024-03-29"), 1),
        ("constant_signal_factor", pd.Timestamp("2024-02-29"), 3),
        ("constant_signal_factor", pd.Timestamp("2024-03-29"), 3),
        ("constant_return_factor", pd.Timestamp("2024-02-29"), 3),
    ]:
        row = rank_ic.loc[(factor_name, return_date)]
        assert pd.isna(row["rank_ic"])
        assert pd.isna(row["directional_rank_ic"])
        assert row["n_obs"] == n_obs

    valid_row = rank_ic.loc[("valid_factor", pd.Timestamp("2024-03-29"))]
    assert valid_row["rank_ic"] > 0.0


def test_run_emp008_factor_quantiles_uses_raw_factor_values_and_prepared_market_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _sample_prepared()
    captured: dict[str, object] = {}

    def fake_evaluate_factor_quantiles(**kwargs: object) -> Emp008FactorQuantileResult:
        captured.update(kwargs)
        return _empty_result()

    monkeypatch.setattr(
        "backtesting.strategies.emp008.factor_quantiles.evaluate_factor_quantiles",
        fake_evaluate_factor_quantiles,
    )

    result = run_emp008_factor_quantiles(
        prepared=prepared,
        start="2024-02-01",
        end="2024-02-29",
        q=7,
    )

    assert result.monthly_returns.empty
    assert result.portfolio_weights.empty
    assert result.rank_ic.empty
    assert result.cumulative_returns.empty
    assert result.summary.empty
    assert captured["factors"] is prepared.raw_factors
    assert captured["close"] is prepared.close
    assert captured["market_cap"] is prepared.market_cap
    assert captured["universe"] is prepared.universe
    assert captured["monthly_dates"] == prepared.monthly_dates
    assert captured["start"] == "2024-02-01"
    assert captured["end"] == "2024-02-29"
    assert captured["q"] == 7
    assert captured["directions"] == {
        "price_to_252d_high": FactorDirection.HIGH,
        "earnings_momentum": FactorDirection.HIGH,
        "dividend_yield_ttm": FactorDirection.HIGH,
        "retail_flow": FactorDirection.HIGH,
        "value": FactorDirection.HIGH,
        "ln_market_cap": FactorDirection.LOW,
    }
