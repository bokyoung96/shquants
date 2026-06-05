import pandas as pd

from backtesting.strategies.emp008.mfbt_emp008_preprocess import (
    build_sector_active_exposures,
    combine_exposures,
    preprocess_factor_frame,
)


def test_preprocess_fills_missing_with_float_mktcap_mean_and_centers() -> None:
    date = pd.Timestamp("2024-01-31")
    raw = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [None]}, index=[date])
    float_mktcap = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [1.0]}, index=[date])
    universe = pd.DataFrame({"A": [True], "B": [True], "C": [True]}, index=[date])

    result = preprocess_factor_frame(raw, float_mktcap, universe)

    weights = float_mktcap.loc[date] / float_mktcap.loc[date].sum()
    assert abs((result.loc[date] * weights).sum()) < 1e-12
    assert abs((result.loc[date].pow(2) * weights).sum() - 1.0) < 1e-12
    assert result.loc[date, "C"] == 0.0
    assert result.loc[date].notna().all()


def test_preprocess_sets_outside_universe_to_zero() -> None:
    date = pd.Timestamp("2024-01-31")
    raw = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [5.0]}, index=[date])
    float_mktcap = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [1.0]}, index=[date])
    universe = pd.DataFrame({"A": [True], "B": [True], "C": [False]}, index=[date])

    result = preprocess_factor_frame(raw, float_mktcap, universe)

    assert result.loc[date, "C"] == 0.0


def test_sector_active_exposures_are_dummy_minus_float_weight() -> None:
    date = pd.Timestamp("2024-01-31")
    sector = pd.DataFrame({"A": ["G10"], "B": ["G10"], "C": ["G20"]}, index=[date])
    float_mktcap = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [1.0]}, index=[date])
    universe = pd.DataFrame({"A": [True], "B": [True], "C": [True]}, index=[date])

    exposures = build_sector_active_exposures(sector, float_mktcap, universe)

    assert exposures["G10"].loc[date, "A"] == 1.0 - 0.8
    assert exposures["G10"].loc[date, "C"] == 0.0 - 0.8
    assert exposures["G20"].loc[date, "C"] == 1.0 - 0.2


def test_combine_exposures_returns_ticker_by_factor_frame() -> None:
    date = pd.Timestamp("2024-01-31")
    alpha = {
        "price_momentum": pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=[date]),
    }
    sector = {
        "G10": pd.DataFrame({"A": [0.2], "B": [-0.8]}, index=[date]),
    }

    result = combine_exposures(alpha, sector, date)

    assert result.index.tolist() == ["A", "B"]
    assert result.columns.tolist() == ["price_momentum", "G10"]
    assert result.loc["A", "G10"] == 0.2
