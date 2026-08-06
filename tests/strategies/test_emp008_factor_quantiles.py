from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.data import MarketData
from backtesting.strategies.emp008.mfbt_emp008_factor_pipeline import PreparedEmp008Factors
from backtesting.strategies.emp008.mfbt_emp008_factor_quantiles import (
    Emp008FactorQuantileResult,
    QuantileWeighting,
    evaluate_factor_quantiles,
    run_emp008_factor_quantiles,
)
from backtesting.strategies.emp008.mfbt_emp008_factor_registry import (
    FactorDirection,
    get_factor_set_definition,
)
from backtesting.strategies.emp008.mfbt_emp008_data import MfbtEmp008Config


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
        config=MfbtEmp008Config(),
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
    ]
    assert "high_minus_low" not in set(sparse_returns["portfolio"])
    assert pd.Timestamp("2024-02-29") in set(
        result.monthly_returns.loc[result.monthly_returns["factor"] == "high_factor", "return_date"]
    )


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
        assert returns["return"].map(np.isfinite).all()
        assert weighted["ticker"].tolist() == ["A", "G"]


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

    empty_factor = factors["high_factor"] * float("nan")
    with pytest.raises(ValueError, match="high_factor"):
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


def test_run_emp008_factor_quantiles_uses_prepared_monthly_dates_and_total_market_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _sample_prepared()
    captured: dict[str, object] = {}

    def fake_evaluate_factor_quantiles(**kwargs: object) -> Emp008FactorQuantileResult:
        captured.update(kwargs)
        return _empty_result()

    monkeypatch.setattr(
        "backtesting.strategies.emp008.mfbt_emp008_factor_quantiles.evaluate_factor_quantiles",
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
    assert captured["factors"] is prepared.alpha_factors
    assert captured["close"] is prepared.close
    assert captured["market_cap"] is prepared.market_cap
    assert captured["universe"] is prepared.universe
    assert captured["monthly_dates"] == prepared.monthly_dates
    assert captured["start"] == "2024-02-01"
    assert captured["end"] == "2024-02-29"
    assert captured["q"] == 7
    assert captured["directions"] == {
        "price_momentum": FactorDirection.HIGH,
        "earnings_momentum": FactorDirection.HIGH,
        "dividend_yield": FactorDirection.HIGH,
        "retail_flow": FactorDirection.HIGH,
        "value": FactorDirection.HIGH,
        "ln_market_cap": FactorDirection.LOW,
    }
