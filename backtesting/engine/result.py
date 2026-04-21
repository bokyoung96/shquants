from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    qty: pd.DataFrame
    turnover: pd.Series
