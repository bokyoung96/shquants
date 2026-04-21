from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class SplitConfig:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


@dataclass(slots=True)
class SplitResult:
    is_frame: pd.DataFrame
    oos_frame: pd.DataFrame


def split_frame(frame: pd.DataFrame, config: SplitConfig) -> SplitResult:
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame.index must be monotonic increasing")
    if config.is_start > config.is_end:
        raise ValueError("is_start must be <= is_end")
    if config.oos_start > config.oos_end:
        raise ValueError("oos_start must be <= oos_end")
    if config.is_end >= config.oos_start:
        raise ValueError("is_end must be < oos_start")

    frame_start = frame.index.min()
    frame_end = frame.index.max()
    if config.is_start < frame_start or config.is_end > frame_end:
        raise ValueError("IS window must be within frame bounds")
    if config.oos_start < frame_start or config.oos_end > frame_end:
        raise ValueError("OOS window must be within frame bounds")

    is_frame = frame.loc[config.is_start : config.is_end].copy()
    if is_frame.empty:
        raise ValueError("IS window must overlap frame")

    oos_frame = frame.loc[config.oos_start : config.oos_end].copy()
    if oos_frame.empty:
        raise ValueError("OOS window must overlap frame")

    return SplitResult(
        is_frame=is_frame,
        oos_frame=oos_frame,
    )
