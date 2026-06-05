import pandas as pd
import pytest

from backtesting.data import MarketData


@pytest.fixture
def mfbt_emp008_market() -> MarketData:
    index = pd.bdate_range("2024-01-02", periods=260)
    columns = ["A", "B", "C"]
    close = pd.DataFrame(
        {
            "A": [100.0] * 259 + [90.0],
            "B": [100.0] * 259 + [80.0],
            "C": [100.0] * 259 + [50.0],
        },
        index=index,
    )
    op_fwd_12m = pd.DataFrame(200_000_000_000.0, index=index, columns=columns)
    op_fwd_12m.loc[index[-1], ["A", "B", "C"]] = [
        220_000_000_000.0,
        200_000_000_000.0,
        180_000_000_000.0,
    ]
    dps_ttm = pd.DataFrame({"A": 5.0, "B": 2.0, "C": 1.0}, index=index)
    retail_flow = pd.DataFrame({"A": -10.0, "B": -20.0, "C": 5.0}, index=index)
    sector_big = pd.DataFrame({"A": "G10", "B": "G10", "C": "G20"}, index=index)
    market_cap = pd.DataFrame(100.0, index=index, columns=columns)
    free_cash_flow = pd.DataFrame({"A": 10.0, "B": 5.0, "C": -5.0}, index=index)
    debt = pd.DataFrame(0.0, index=index, columns=columns)
    quick_asset = pd.DataFrame(0.0, index=index, columns=columns)
    universe = pd.DataFrame(True, index=index, columns=columns)
    return MarketData(
        frames={
            "close": close,
            "op_fwd_12m": op_fwd_12m,
            "dps_ttm": dps_ttm,
            "retail_flow": retail_flow,
            "sector_big": sector_big,
            "market_cap": market_cap,
            "free_cash_flow": free_cash_flow,
            "interest_bearing_liability": debt,
            "quick_asset": quick_asset,
            "k200_yn": universe.astype(int),
        },
        universe=universe,
        benchmark=None,
    )
