from pathlib import Path

import pandas as pd

from assetallocation.feature.builder import ASSETALLOCATION_TICKERS, FeatureBuilder


def _write_ohlc(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow")


def _sample_ohlc(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    close = pd.DataFrame(
        {
            "USYC2Y10 Index": [50, 55, 52, 48, 45, 40],
            "USYC3M2Y Index": [80, 82, 78, 70, 65, 60],
            "USDJPY Curncy": [100, 101, 102, 103, 104, 105],
            "USGG10YR Index": [4.0, 4.1, 4.0, 3.9, 3.8, 3.7],
            "GC1 Comdty": [1900, 1910, 1920, 1930, 1940, 1950],
            "CL1 Comdty": [70, 71, 69, 72, 73, 74],
            "HG1 Comdty": [400, 402, 405, 407, 409, 411],
            "SPX Index": [1000, 1010, 1020, 1030, 1040, 1050],
            "INDU Index": [30000, 30100, 30200, 30300, 30400, 30500],
            "RTY Index": [2000, 2010, 2020, 2030, 2040, 2050],
            "SPY US Equity": [400, 404, 408, 412, 416, 420],
            "IEF US Equity": [100, 101, 102, 103, 104, 105],
        },
        index=index,
    )
    open_ = close * 0.99
    high = close * 1.01
    low = close * 0.98
    for frame in (open_, high, low, close):
        frame.index.name = "date"
    return {"open": open_, "high": high, "low": low, "close": close}


def test_feature_builder_writes_feature_and_target_parquets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "feature"
    index = pd.date_range("2024-01-01", periods=6, freq="D", name="date")
    for stem, frame in _sample_ohlc(index).items():
        _write_ohlc(data_dir / f"{stem}.parquet", frame)

    result = FeatureBuilder(
        data_dir=data_dir,
        output_dir=output_dir,
        horizons=(2, 3),
        volatility_windows=(2,),
        zscore_window=3,
    ).build()

    features = pd.read_parquet(output_dir / "features.parquet", engine="pyarrow")
    targets = pd.read_parquet(output_dir / "targets.parquet", engine="pyarrow")

    assert result.feature_path == output_dir / "features.parquet"
    assert result.target_path == output_dir / "targets.parquet"
    assert result.rows == 6
    assert result.feature_columns == len(features.columns)
    assert result.target_columns == len(targets.columns)
    assert "spx_mom_2d" in features.columns
    assert "us10y_chg_2d_bp" in features.columns
    assert "curve_2y10_inverted" in features.columns
    assert "spy_vs_ief_mom_2d" in features.columns
    assert "target_spy_excess_ief_fwd_2d" in targets.columns

    expected_ief_forward = 104 / 102 - 1.0
    expected_spy_forward = 416 / 408 - 1.0
    actual = targets.loc[pd.Timestamp("2024-01-02"), "target_spy_excess_ief_fwd_2d"]
    assert round(actual, 10) == round(expected_spy_forward - expected_ief_forward, 10)


def test_feature_builder_rejects_missing_required_ticker(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    index = pd.date_range("2024-01-01", periods=6, freq="D", name="date")
    frames = _sample_ohlc(index)
    frames["close"] = frames["close"].drop(columns=["SPY US Equity"])
    for stem, frame in frames.items():
        _write_ohlc(data_dir / f"{stem}.parquet", frame)

    try:
        FeatureBuilder(data_dir=data_dir, output_dir=tmp_path / "feature").build()
    except ValueError as exc:
        assert "missing required assetallocation tickers" in str(exc)
    else:
        raise AssertionError("expected missing ticker validation")


def test_asset_ticker_map_matches_expected_bloomberg_columns() -> None:
    assert set(ASSETALLOCATION_TICKERS) == {
        "USYC2Y10 Index",
        "USYC3M2Y Index",
        "USDJPY Curncy",
        "USGG10YR Index",
        "GC1 Comdty",
        "CL1 Comdty",
        "HG1 Comdty",
        "SPX Index",
        "INDU Index",
        "RTY Index",
        "SPY US Equity",
        "IEF US Equity",
    }
