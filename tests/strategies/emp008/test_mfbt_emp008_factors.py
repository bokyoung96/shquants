import pandas as pd

from backtesting.strategies.emp008.mfbt_emp008_data import MfbtEmp008Config
from backtesting.strategies.emp008.mfbt_emp008_factors import build_raw_mfbt_factors


def test_build_raw_mfbt_factors_emits_continuous_month_end_exposures(mfbt_emp008_market) -> None:
    factors = build_raw_mfbt_factors(mfbt_emp008_market, MfbtEmp008Config())
    latest = factors["price_momentum"].dropna(how="all").index[-1]

    assert set(factors) == {
        "price_momentum",
        "earnings_momentum",
        "dividend_yield",
        "retail_flow",
        "value",
    }
    assert factors["price_momentum"].loc[latest, "A"] == 0.9
    assert factors["price_momentum"].loc[latest, "B"] == 0.8
    assert factors["dividend_yield"].loc[latest, "A"] == 5.0 / 90.0
    assert factors["retail_flow"].loc[latest, "A"] == factors["retail_flow"].loc[latest, "B"]
    assert factors["retail_flow"].loc[latest, "A"] > factors["retail_flow"].loc[latest, "C"]
    assert factors["value"].loc[latest, "A"] == 10.0 / 100.0


def test_value_factor_treats_non_positive_tev_as_missing(mfbt_emp008_market) -> None:
    mfbt_emp008_market.frames["quick_asset"].iloc[-1] = 101.0

    factors = build_raw_mfbt_factors(mfbt_emp008_market, MfbtEmp008Config())
    latest = factors["price_momentum"].dropna(how="all").index[-1]

    assert pd.isna(factors["value"].loc[latest, "A"])
    assert pd.isna(factors["value"].loc[latest, "B"])
    assert pd.isna(factors["value"].loc[latest, "C"])
