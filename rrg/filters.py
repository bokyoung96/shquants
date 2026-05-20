from __future__ import annotations

import pandas as pd


def smooth_frame(frame: pd.DataFrame, *, method: str = "none", window: int = 5) -> pd.DataFrame:
    """Smooth a frame with deterministic methods suitable for tests and reports."""
    if method == "none":
        return frame.copy()
    if window <= 0:
        raise ValueError("window must be positive")
    if method == "rolling":
        return frame.rolling(window, min_periods=1).mean()
    if method == "ewm":
        return frame.ewm(span=window, adjust=False, min_periods=1).mean()
    raise ValueError(f"unsupported smoothing method: {method}")


def rolling_zscore(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("window must be greater than 1")
    mean = frame.rolling(window, min_periods=max(2, window // 2)).mean()
    std = frame.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    return frame.sub(mean).divide(std.replace(0.0, pd.NA))
