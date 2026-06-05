import pandas as pd

from backtesting.strategies.emp008.mfbt_emp008_preprocess import (
    build_sector_active_exposures,
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
    assert result.loc[date].notna().all()


def test_sector_active_exposures_are_dummy_minus_float_weight() -> None:
    date = pd.Timestamp("2024-01-31")
    sector = pd.DataFrame({"A": ["G10"], "B": ["G10"], "C": ["G20"]}, index=[date])
    float_mktcap = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [1.0]}, index=[date])
    universe = pd.DataFrame({"A": [True], "B": [True], "C": [True]}, index=[date])

    exposures = build_sector_active_exposures(sector, float_mktcap, universe)

    assert exposures["G10"].loc[date, "A"] == 1.0 - 0.8
    assert exposures["G10"].loc[date, "C"] == 0.0 - 0.8
    assert exposures["G20"].loc[date, "C"] == 1.0 - 0.2
