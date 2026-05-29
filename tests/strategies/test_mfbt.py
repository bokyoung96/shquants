import pandas as pd
from pandas.testing import assert_frame_equal

from backtesting.data import MarketData
from backtesting.strategies import build_strategy
from backtesting.strategies.mfbt import (
    DividendYieldFactor,
    EarningsMomentumFactor,
    PriceMomentumFactor,
    RetailFlowFactor,
    ValueFactor,
    _quarter_lagged_financials,
)


FACTOR_BUILDERS = {
    "price_momentum": PriceMomentumFactor(),
    "earnings_momentum": EarningsMomentumFactor(),
    "dividend_yield": DividendYieldFactor(),
    "retail_flow": RetailFlowFactor(),
    "value": ValueFactor(),
}


def _mfbt_market(
    *,
    close: pd.DataFrame | None = None,
    op_fwd_12m: pd.DataFrame | None = None,
    dps_ttm: pd.DataFrame | None = None,
    dividend_cash_ttm: pd.DataFrame | None = None,
    retail_flow: pd.DataFrame | None = None,
    sector_big: pd.DataFrame | None = None,
    market_cap: pd.DataFrame | None = None,
    free_cash_flow: pd.DataFrame | None = None,
    interest_bearing_liability: pd.DataFrame | None = None,
    quick_asset: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
) -> MarketData:
    frames: dict[str, pd.DataFrame] = {}
    if close is not None:
        frames["close"] = close
        if op_fwd_12m is None:
            op_fwd_12m = pd.DataFrame(200_000_000_000.0, index=close.index, columns=close.columns)
        if dps_ttm is None:
            dps_ttm = pd.DataFrame(1000.0, index=close.index, columns=close.columns)
        if dividend_cash_ttm is None:
            dividend_cash_ttm = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        if retail_flow is None:
            retail_flow = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        if sector_big is None:
            sector_big = pd.DataFrame("G10", index=close.index, columns=close.columns)
        if market_cap is None:
            market_cap = pd.DataFrame(100_000_000_000.0, index=close.index, columns=close.columns)
        if free_cash_flow is None:
            free_cash_flow = pd.DataFrame(10_000_000_000.0, index=close.index, columns=close.columns)
        if interest_bearing_liability is None:
            interest_bearing_liability = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        if quick_asset is None:
            quick_asset = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if op_fwd_12m is not None:
        frames["op_fwd_12m"] = op_fwd_12m
    if dps_ttm is not None:
        frames["dps_ttm"] = dps_ttm
    if dividend_cash_ttm is not None:
        frames["dividend_cash_ttm"] = dividend_cash_ttm
    if retail_flow is not None:
        frames["retail_flow"] = retail_flow
    if sector_big is not None:
        frames["sector_big"] = sector_big
    if market_cap is not None:
        frames["market_cap"] = market_cap
    if free_cash_flow is not None:
        frames["free_cash_flow"] = free_cash_flow
    if interest_bearing_liability is not None:
        frames["interest_bearing_liability"] = interest_bearing_liability
    if quick_asset is not None:
        frames["quick_asset"] = quick_asset
    return MarketData(frames=frames, universe=universe, benchmark=None)


def _factor_score(name: str, market: MarketData) -> pd.DataFrame:
    return FACTOR_BUILDERS[name].build(market)


def test_mfbt_price_momentum_emits_month_end_binary_signal_from_252_day_close_high() -> None:
    index = pd.to_datetime(["2024-01-30", "2024-01-31", *pd.date_range("2024-02-01", periods=252, freq="D")])
    close = pd.DataFrame(
        {
            "A": [100.0] * 253 + [81.0],
            "B": [100.0] * 253 + [80.0],
            "C": [100.0] * 253 + [79.0],
        },
        index=index,
    )
    market = _mfbt_market(close=close)

    signal = _factor_score("price_momentum", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=close.columns)
    expected.loc[index[-1], "A"] = 1.0
    expected.loc[index[-1], ["B", "C"]] = 0.0
    assert_frame_equal(signal, expected)


def test_mfbt_builds_equal_weight_plan_for_price_momentum_names() -> None:
    index = pd.bdate_range("2023-01-02", periods=360)
    close = pd.DataFrame(
        {
            "A": [100.0] * (len(index) - 1) + [90.0],
            "B": [100.0] * (len(index) - 1) + [85.0],
            "C": [100.0] * (len(index) - 1) + [75.0],
        },
        index=index,
    )
    market = _mfbt_market(close=close)

    plan = build_strategy("mfbt", top_n=2).build_plan(market)
    last = plan.target_weights.iloc[-1]

    assert last["A"] == 0.5
    assert last["B"] == 0.5
    assert last["C"] == 0.0


def test_mfbt_lists_earnings_momentum_datasets() -> None:
    strategy = build_strategy("mfbt")

    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert "qw_adj_c" in dataset_values
    assert "qw_op_fwd_12m" in dataset_values
    assert "qw_dps_ttm" in dataset_values
    assert "qw_dividend_cash_ttm" in dataset_values
    assert "qw_dividend_cash" not in dataset_values
    assert "qw_retail" in dataset_values
    assert "qw_wi_sec_26_big" in dataset_values
    assert "qw_wi_sec_26" not in dataset_values
    assert "qw_wics_sec_big" not in dataset_values
    assert "qw_mktcap" in dataset_values
    assert "qw_fcf" in dataset_values
    assert "qw_int_bearing_liab_nfq0" in dataset_values
    assert "qw_quick_assets_nfq0" in dataset_values
    assert "qw_tangible_assets_nfq0" not in dataset_values


def test_mfbt_price_momentum_masks_non_universe_names_in_factor_meta() -> None:
    index = pd.date_range("2024-01-02", periods=253, freq="D")
    close = pd.DataFrame(
        {
            "A": [100.0] * 252 + [90.0],
            "B": [100.0] * 252 + [90.0],
        },
        index=index,
    )
    universe = pd.DataFrame({"A": True, "B": False}, index=index)
    market = _mfbt_market(close=close, universe=universe)

    score = _factor_score("price_momentum", market)

    assert score.loc[index[-1], "A"] == 1.0
    assert pd.isna(score.loc[index[-1], "B"])


def test_mfbt_final_factor_meta_uses_common_dates_without_forcing_common_tickers() -> None:
    index = pd.bdate_range("2023-01-02", periods=360)
    columns = ["A", "B"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    free_cash_flow = pd.DataFrame(10_000_000_000.0, index=index, columns=columns)
    free_cash_flow["B"] = None
    market = _mfbt_market(close=close, free_cash_flow=free_cash_flow)

    bundle = build_strategy("mfbt").signal_producer.build(market)

    assert bundle.meta["price_momentum"].loc[index[20]].isna().all()
    for frame in bundle.meta.values():
        assert pd.notna(frame.loc[index[-1], "A"])
    assert pd.notna(bundle.meta["price_momentum"].loc[index[-1], "B"])
    assert pd.notna(bundle.meta["earnings_momentum"].loc[index[-1], "B"])
    assert pd.notna(bundle.meta["dividend_yield"].loc[index[-1], "B"])
    assert pd.notna(bundle.meta["retail_flow"].loc[index[-1], "B"])
    assert pd.isna(bundle.meta["value"].loc[index[-1], "B"])


def test_mfbt_factor_frames_share_close_index_and_columns() -> None:
    close_index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    op_index = pd.date_range("2024-01-31", "2024-02-29", freq="D")
    columns = ["A", "B"]
    close = pd.DataFrame(100.0, index=close_index, columns=columns)
    op_fwd_12m = pd.DataFrame(200_000_000_000.0, index=op_index, columns=columns)
    market = _mfbt_market(close=close, op_fwd_12m=op_fwd_12m)

    bundle = build_strategy("mfbt").signal_producer.build(market)

    for frame in bundle.meta.values():
        assert frame.index.equals(close.index)
        assert frame.columns.equals(close.columns)


def test_mfbt_earnings_momentum_scores_month_end_growth_quantiles() -> None:
    index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-27", "2024-02-29"])
    columns = ["A", "B", "C", "D", "E"]
    op_fwd_12m = pd.DataFrame(
        [
            [180_000_000_000.0] * 5,
            [200_000_000_000.0] * 5,
            [190_000_000_000.0, 200_000_000_000.0, 210_000_000_000.0, 240_000_000_000.0, 280_000_000_000.0],
            [180_000_000_000.0, 200_000_000_000.0, 220_000_000_000.0, 260_000_000_000.0, 320_000_000_000.0],
        ],
        index=index,
        columns=columns,
    )
    close = pd.DataFrame(100.0, index=index, columns=columns)
    market = _mfbt_market(close=close, op_fwd_12m=op_fwd_12m)

    score = _factor_score("earnings_momentum", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc["2024-02-29"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert_frame_equal(score, expected)


def test_mfbt_earnings_momentum_scores_quantiles_inside_universe_only() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    columns = ["A", "B", "C", "D", "E", "F"]
    op_fwd_12m = pd.DataFrame(
        [
            [100_000_000_000.0] * 6,
            [
                90_000_000_000.0,
                100_000_000_000.0,
                110_000_000_000.0,
                120_000_000_000.0,
                130_000_000_000.0,
                1_000_000_000_000.0,
            ],
        ],
        index=index,
        columns=columns,
    )
    close = pd.DataFrame(100.0, index=index, columns=columns)
    universe = pd.DataFrame(True, index=index, columns=columns)
    universe["F"] = False
    market = _mfbt_market(close=close, op_fwd_12m=op_fwd_12m, universe=universe)

    score = _factor_score("earnings_momentum", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc["2024-02-29", ["A", "B", "C", "D", "E"]] = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert_frame_equal(score, expected)


def test_mfbt_earnings_momentum_requires_current_and_previous_month_consensus() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    op_fwd_12m = pd.DataFrame(
        {
            "A": [200_000_000_000.0, 240_000_000_000.0],
            "B": [None, 110.0],
            "C": [200_000_000_000.0, None],
            "D": [0.0, 110_000_000_000.0],
            "E": [-200_000_000_000.0, -180_000_000_000.0],
        },
        index=index,
    )
    close = pd.DataFrame(100.0, index=index, columns=op_fwd_12m.columns)
    market = _mfbt_market(close=close, op_fwd_12m=op_fwd_12m)

    score = _factor_score("earnings_momentum", market)

    assert pd.isna(score.loc["2024-02-29", "B"])
    assert pd.isna(score.loc["2024-02-29", "C"])
    assert pd.isna(score.loc["2024-02-29", "D"])
    assert score.loc["2024-02-29", "A"] > 0.0
    assert score.loc["2024-02-29", "E"] == 0.0


def test_mfbt_earnings_momentum_filters_low_op_extreme_growth_to_zero_before_scoring() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    columns = ["A", "B", "C", "D", "E"]
    op_fwd_12m = pd.DataFrame(
        {
            "A": [200_000_000_000.0, 180_000_000_000.0],
            "B": [200_000_000_000.0, 200_000_000_000.0],
            "C": [200_000_000_000.0, 220_000_000_000.0],
            "D": [200_000_000_000.0, 240_000_000_000.0],
            "E": [60_000_000_000.0, 96_000_000_000.0],
        },
        index=index,
    )
    close = pd.DataFrame(100.0, index=index, columns=columns)
    market = _mfbt_market(close=close, op_fwd_12m=op_fwd_12m)

    score = _factor_score("earnings_momentum", market)

    assert score.loc["2024-02-29", "E"] == score.loc["2024-02-29", "B"]
    assert score.loc["2024-02-29", "E"] < score.loc["2024-02-29", "D"]


def test_mfbt_earnings_momentum_does_not_filter_exact_100bn_op_growth() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    columns = ["A", "B", "C", "D", "E"]
    op_fwd_12m = pd.DataFrame(
        {
            "A": [200_000_000_000.0, 180_000_000_000.0],
            "B": [200_000_000_000.0, 200_000_000_000.0],
            "C": [200_000_000_000.0, 220_000_000_000.0],
            "D": [200_000_000_000.0, 240_000_000_000.0],
            "E": [62_500_000_000.0, 100_000_000_000.0],
        },
        index=index,
    )
    close = pd.DataFrame(100.0, index=index, columns=columns)
    market = _mfbt_market(close=close, op_fwd_12m=op_fwd_12m)

    score = _factor_score("earnings_momentum", market)

    assert score.loc["2024-02-29", "E"] > score.loc["2024-02-29", "D"]


def test_mfbt_dividend_yield_scores_month_end_yield_quantiles() -> None:
    index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-27", "2024-02-29"])
    columns = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    dps_ttm = pd.DataFrame(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 3.0, 5.0, 7.0, 9.0],
        ],
        index=index,
        columns=columns,
    )
    market = _mfbt_market(close=close, dps_ttm=dps_ttm)

    score = _factor_score("dividend_yield", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc["2024-01-31"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    expected.loc["2024-02-29"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert_frame_equal(score, expected)


def test_mfbt_dividend_yield_scores_quantiles_inside_universe_only() -> None:
    index = pd.to_datetime(["2024-01-31"])
    columns = ["A", "B", "C", "D", "E", "F"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    dps_ttm = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]], index=index, columns=columns)
    universe = pd.DataFrame(True, index=index, columns=columns)
    universe["F"] = False
    market = _mfbt_market(close=close, dps_ttm=dps_ttm, universe=universe)

    score = _factor_score("dividend_yield", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc["2024-01-31", ["A", "B", "C", "D", "E"]] = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert_frame_equal(score, expected)


def test_mfbt_dividend_yield_adds_bonus_for_same_month_three_year_ttm_increase() -> None:
    index = pd.to_datetime(["2022-01-31", "2023-01-31", "2024-01-30", "2024-01-31"])
    columns = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    dps_ttm = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        index=index,
        columns=columns,
    )
    dividend_cash = pd.DataFrame(0.0, index=index, columns=columns)
    dividend_cash.loc["2022-01-31", "E"] = 10.0
    dividend_cash.loc["2023-01-31", "E"] = 20.0
    dividend_cash.loc["2024-01-31", "E"] = 30.0
    dividend_cash.loc["2022-01-31", "D"] = 30.0
    dividend_cash.loc["2023-01-31", "D"] = 20.0
    dividend_cash.loc["2024-01-31", "D"] = 10.0
    market = _mfbt_market(close=close, dps_ttm=dps_ttm, dividend_cash_ttm=dividend_cash)

    score = _factor_score("dividend_yield", market)

    assert score.loc["2024-01-30"].isna().all()
    assert score.loc["2024-01-31", "D"] == 3.0
    assert score.loc["2024-01-31", "E"] == 5.0


def test_mfbt_dividend_yield_bonus_uses_same_month_ttm_not_year_end_cash() -> None:
    index = pd.to_datetime(
        [
            "2022-01-31",
            "2022-12-30",
            "2023-01-31",
            "2023-12-29",
            "2024-01-31",
        ]
    )
    columns = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    dps_ttm = pd.DataFrame(
        {
            "A": [1.0] * len(index),
            "B": [2.0] * len(index),
            "C": [3.0] * len(index),
            "D": [4.0] * len(index),
            "E": [5.0] * len(index),
        },
        index=index,
    )
    dividend_cash = pd.DataFrame(0.0, index=index, columns=columns)
    dividend_cash.loc[["2022-01-31", "2023-01-31", "2024-01-31"], "E"] = [10.0, 20.0, 30.0]
    dividend_cash.loc[["2022-12-30", "2023-12-29"], "E"] = [100.0, 10.0]
    market = _mfbt_market(close=close, dps_ttm=dps_ttm, dividend_cash_ttm=dividend_cash)

    score = _factor_score("dividend_yield", market)

    assert score.loc["2024-01-31", "E"] == 5.0


def test_mfbt_retail_flow_scores_sector_net_selling_inside_universe_only() -> None:
    index = pd.bdate_range("2023-02-14", periods=252)
    columns = ["A", "B", "C", "D", "E", "F", "G"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    retail_flow = pd.DataFrame(
        {
            "A": [-10.0] * len(index),
            "B": [-20.0] * len(index),
            "C": [-5.0] * len(index),
            "D": [0.0] * len(index),
            "E": [5.0] * len(index),
            "F": [10.0] * len(index),
            "G": [-10_000.0] * len(index),
        },
        index=index,
    )
    sector_big = pd.DataFrame(
        {
            "A": ["G10"] * len(index),
            "B": ["G10"] * len(index),
            "C": ["G15"] * len(index),
            "D": ["G20"] * len(index),
            "E": ["G25"] * len(index),
            "F": ["G30"] * len(index),
            "G": ["G10"] * len(index),
        },
        index=index,
    )
    universe = pd.DataFrame(True, index=index, columns=columns)
    universe["G"] = False
    market = _mfbt_market(
        close=close,
        retail_flow=retail_flow,
        sector_big=sector_big,
        universe=universe,
    )

    score = _factor_score("retail_flow", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc[index[-1], ["A", "B", "C", "D", "E", "F"]] = [4.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    assert_frame_equal(score, expected)


def test_mfbt_retail_flow_scores_sector_average_not_sector_sum() -> None:
    index = pd.bdate_range("2023-02-14", periods=252)
    columns = ["A", "B", "C", "D", "E", "F"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    retail_flow = pd.DataFrame(
        {
            "A": [-10.0] * len(index),
            "B": [-10.0] * len(index),
            "C": [-15.0] * len(index),
            "D": [0.0] * len(index),
            "E": [5.0] * len(index),
            "F": [10.0] * len(index),
        },
        index=index,
    )
    sector_big = pd.DataFrame(
        {
            "A": ["G10"] * len(index),
            "B": ["G10"] * len(index),
            "C": ["G15"] * len(index),
            "D": ["G20"] * len(index),
            "E": ["G25"] * len(index),
            "F": ["G30"] * len(index),
        },
        index=index,
    )
    universe = pd.DataFrame(True, index=index, columns=columns)
    market = _mfbt_market(close=close, retail_flow=retail_flow, sector_big=sector_big, universe=universe)

    score = _factor_score("retail_flow", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc[index[-1], ["A", "B", "C", "D", "E", "F"]] = [3.0, 3.0, 4.0, 2.0, 1.0, 0.0]
    assert_frame_equal(score, expected)


def test_mfbt_value_scores_lagged_fcf_to_tev_quantiles() -> None:
    index = pd.to_datetime(["2024-03-31", "2024-04-30", "2024-05-31", "2024-06-30"])
    columns = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    market_cap = pd.DataFrame(100.0, index=index, columns=columns)
    free_cash_flow = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [None, None, None, None, None],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [None, None, None, None, None],
        ],
        index=index,
        columns=columns,
    )
    interest_bearing_liability = pd.DataFrame(0.0, index=index, columns=columns)
    quick_asset = pd.DataFrame(0.0, index=index, columns=columns)
    market = _mfbt_market(
        close=close,
        market_cap=market_cap,
        free_cash_flow=free_cash_flow,
        interest_bearing_liability=interest_bearing_liability,
        quick_asset=quick_asset,
    )

    score = _factor_score("value", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc["2024-04-30"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    expected.loc["2024-05-31"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    expected.loc["2024-06-30"] = [4.0, 3.0, 2.0, 1.0, 0.0]
    assert_frame_equal(score, expected)


def test_mfbt_value_financial_lag_uses_fixed_source_month_windows() -> None:
    index = pd.to_datetime(
        [
            "2024-03-31",
            "2024-04-30",
            "2024-05-31",
            "2024-06-30",
            "2024-08-30",
            "2024-09-30",
            "2024-11-29",
            "2024-12-30",
            "2025-01-31",
            "2025-03-31",
        ]
    )
    frame = pd.DataFrame({"A": [3.0, None, 5.0, None, 8.0, None, 11.0, None, None, None]}, index=index)

    lagged = _quarter_lagged_financials(frame, pd.DatetimeIndex(index[1:]))

    expected = pd.DataFrame(
        {"A": [3.0, 3.0, 5.0, 5.0, 8.0, 8.0, 11.0, 11.0, 11.0]},
        index=index[1:],
    )
    assert_frame_equal(lagged, expected)


def test_mfbt_value_keeps_missing_inputs_nan_and_scores_non_positive_tev_low() -> None:
    index = pd.to_datetime(["2024-03-31", "2024-04-30"])
    columns = ["A", "B", "C", "D", "E", "F"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    market_cap = pd.DataFrame(100.0, index=index, columns=columns)
    free_cash_flow = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0, -100.0, None], [None, None, None, None, None, None]],
        index=index,
        columns=columns,
    )
    interest_bearing_liability = pd.DataFrame(0.0, index=index, columns=columns)
    quick_asset = pd.DataFrame(
        [[0.0, 0.0, 0.0, 0.0, 101.0, 0.0], [0.0, 0.0, 0.0, 0.0, 101.0, 0.0]],
        index=index,
        columns=columns,
    )
    universe = pd.DataFrame(True, index=index, columns=columns)
    market = _mfbt_market(
        close=close,
        market_cap=market_cap,
        free_cash_flow=free_cash_flow,
        interest_bearing_liability=interest_bearing_liability,
        quick_asset=quick_asset,
        universe=universe,
    )

    score = _factor_score("value", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc["2024-04-30", ["A", "B", "C", "D", "E"]] = [1.0, 2.0, 3.0, 4.0, 0.0]
    assert_frame_equal(score, expected)


def test_mfbt_value_scores_all_non_positive_tev_names_as_zero() -> None:
    index = pd.to_datetime(["2024-03-31", "2024-04-30"])
    columns = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    market_cap = pd.DataFrame(100.0, index=index, columns=columns)
    free_cash_flow = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0], [None] * 5], index=index, columns=columns)
    interest_bearing_liability = pd.DataFrame(0.0, index=index, columns=columns)
    quick_asset = pd.DataFrame(101.0, index=index, columns=columns)
    market = _mfbt_market(
        close=close,
        market_cap=market_cap,
        free_cash_flow=free_cash_flow,
        interest_bearing_liability=interest_bearing_liability,
        quick_asset=quick_asset,
    )

    score = _factor_score("value", market)

    expected = pd.DataFrame(float("nan"), index=index, columns=columns)
    expected.loc["2024-04-30"] = 0.0
    assert_frame_equal(score, expected)
