import pandas as pd
from pandas.api.types import is_integer_dtype


def expand_monthly_frame(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    validity: str,
) -> pd.DataFrame:
    if validity != "month_only":
        raise ValueError(f"unsupported validity: {validity}")

    out = frame.copy()
    for column in out.columns:
        if is_integer_dtype(out[column].dtype):
            out[column] = out[column].astype("Int64")

    out = out.reindex(calendar).copy()
    for ts, row in frame.iterrows():
        month_mask = (calendar.year == ts.year) & (calendar.month == ts.month)
        for column in out.columns:
            out.loc[month_mask, column] = row[column]
    return out
