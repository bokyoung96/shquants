import pandas as pd


def fill_prices(
    close: pd.DataFrame,
    open_: pd.DataFrame | None,
    fill_mode: str,
) -> pd.DataFrame:
    if fill_mode == "close":
        return close
    if fill_mode == "next_open":
        if open_ is None:
            raise ValueError("open prices required for next_open")
        return open_.shift(-1).iloc[:-1]
    raise ValueError(f"unsupported fill_mode: {fill_mode}")
