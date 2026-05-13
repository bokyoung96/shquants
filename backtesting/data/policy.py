import pandas as pd
from pandas.api.types import is_integer_dtype


def expand_monthly_frame(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    validity: str,
) -> pd.DataFrame:
    if validity != "month_only":
        raise ValueError(f"unsupported validity: {validity}")

    monthly = frame.copy()
    for column in monthly.columns:
        if is_integer_dtype(monthly[column].dtype):
            monthly[column] = monthly[column].astype("Int64")

    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M")
    monthly = monthly.loc[~monthly.index.duplicated(keep="last")]

    expanded = monthly.reindex(calendar.to_period("M"))
    expanded.index = calendar
    return expanded
