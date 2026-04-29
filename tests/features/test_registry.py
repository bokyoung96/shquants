import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtesting.catalog import DatasetId
from backtesting.data import MarketData
from backtesting.features import build_features, feature_dataset_ids, feature_warmup_days, get_feature


@pytest.fixture
def market_data() -> MarketData:
    index = pd.date_range("2024-01-01", periods=80, freq="D")
    close = pd.DataFrame(
        {
            "A": range(100, 180),
            "B": range(200, 280),
        },
        index=index,
        dtype=float,
    )
    volume = pd.DataFrame(
        {
            "A": range(1000, 1080),
            "B": range(2000, 2080),
        },
        index=index,
        dtype=float,
    )
    market_cap = pd.DataFrame(
        {
            "A": range(10000, 10080),
            "B": range(20000, 20080),
        },
        index=index,
        dtype=float,
    )
    float_market_cap = market_cap * 0.8
    foreign_ratio = pd.DataFrame(
        {
            "A": [0.1] * len(index),
            "B": [0.2] * len(index),
        },
        index=index,
        dtype=float,
    )
    inst_flow = pd.DataFrame(
        {
            "A": range(10, 90),
            "B": range(20, 100),
        },
        index=index,
        dtype=float,
    )
    retail_flow = pd.DataFrame(
        {
            "A": range(-5, 75),
            "B": range(5, 85),
        },
        index=index,
        dtype=float,
    )
    open_frame = close - 1.0

    return MarketData(
        frames={
            "close": close,
            "open": open_frame,
            "volume": volume,
            "market_cap": market_cap,
            "float_market_cap": float_market_cap,
            "foreign_ratio": foreign_ratio,
            "inst_flow": inst_flow,
            "retail_flow": retail_flow,
        },
        universe=None,
        benchmark=None,
    )


def test_feature_dataset_ids_include_required_datasets() -> None:
    dataset_ids = feature_dataset_ids(["momentum_60d", "market_cap", "avg_trading_value_20d"])

    assert DatasetId.QW_ADJ_C in dataset_ids
    assert DatasetId.QW_MKTCAP in dataset_ids
    assert DatasetId.QW_V in dataset_ids


def test_feature_dataset_ids_preserve_order_while_deduplicating() -> None:
    assert feature_dataset_ids(["market_cap", "momentum_20d", "avg_trading_value_20d", "market_cap"]) == (
        DatasetId.QW_MKTCAP,
        DatasetId.QW_ADJ_C,
        DatasetId.QW_V,
    )


def test_feature_warmup_days_returns_maximum_lookback() -> None:
    assert feature_warmup_days(["momentum_60d", "momentum_20d"]) == 60


def test_build_features_returns_registered_frames(market_data: MarketData) -> None:
    features = build_features(
        market_data,
        [
            "market_cap",
            "float_market_cap",
            "momentum_20d",
            "avg_trading_value_20d",
            "foreign_ratio",
            "institution_flow_20d",
            "retail_flow_20d",
            "market_cap",
        ],
    )

    assert list(features) == [
        "market_cap",
        "float_market_cap",
        "momentum_20d",
        "avg_trading_value_20d",
        "foreign_ratio",
        "institution_flow_20d",
        "retail_flow_20d",
    ]
    assert_frame_equal(features["market_cap"], market_data.frames["market_cap"])
    assert_frame_equal(features["float_market_cap"], market_data.frames["float_market_cap"])
    assert_frame_equal(features["foreign_ratio"], market_data.frames["foreign_ratio"])
    assert features["momentum_20d"].shape == market_data.frames["close"].shape
    assert features["avg_trading_value_20d"].shape == market_data.frames["close"].shape
    assert features["institution_flow_20d"].shape == market_data.frames["close"].shape
    assert features["retail_flow_20d"].shape == market_data.frames["close"].shape

    expected_momentum = market_data.frames["close"].pct_change(20, fill_method=None)
    expected_trading_value = (market_data.frames["close"] * market_data.frames["volume"]).rolling(20, min_periods=20).mean()
    assert_frame_equal(features["momentum_20d"], expected_momentum)
    assert_frame_equal(features["avg_trading_value_20d"], expected_trading_value)


def test_get_feature_raises_for_unknown_field() -> None:
    with pytest.raises(KeyError, match="unknown feature field"):
        get_feature("not_a_feature")


def test_build_features_raises_for_unknown_field(market_data: MarketData) -> None:
    with pytest.raises(KeyError, match="unknown feature field"):
        build_features(market_data, ["market_cap", "not_a_feature"])


def test_build_features_raises_for_missing_required_market_frame() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    close = pd.DataFrame({"A": range(30)}, index=index, dtype=float)
    market = MarketData(frames={"close": close}, universe=None, benchmark=None)

    with pytest.raises(KeyError, match="volume"):
        build_features(market, ["avg_trading_value_20d"])
