from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .filters import rolling_zscore, smooth_frame


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    name: str
    periods: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("horizon name is required")
        if self.periods <= 0:
            raise ValueError("horizon periods must be positive")


@dataclass(frozen=True, slots=True)
class RrgConfig:
    horizons: tuple[HorizonSpec, ...] = (
        HorizonSpec("short", 5),
        HorizonSpec("medium", 20),
        HorizonSpec("long", 60),
    )
    trend_window: int = 60
    smoothing_method: str = "none"
    smoothing_window: int = 5
    acceleration_z_window: int = 60
    turning_threshold: float = 1.0

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("at least one horizon is required")
        if self.trend_window <= 0:
            raise ValueError("trend_window must be positive")
        if self.acceleration_z_window <= 1:
            raise ValueError("acceleration_z_window must be greater than 1")
        if self.turning_threshold < 0.0:
            raise ValueError("turning_threshold must be nonnegative")


def compute_multi_horizon_rrg(
    *,
    sector_prices: pd.DataFrame,
    benchmark: pd.Series,
    confidence: pd.DataFrame | None = None,
    config: RrgConfig | None = None,
) -> pd.DataFrame:
    cfg = config or RrgConfig()
    frames = [
        compute_horizon_rrg(
            sector_prices=sector_prices,
            benchmark=benchmark,
            horizon=horizon,
            confidence=confidence,
            config=cfg,
        )
        for horizon in cfg.horizons
    ]
    if not frames:
        return _empty_rrg_frame()
    return pd.concat(frames, ignore_index=True)


def compute_horizon_rrg(
    *,
    sector_prices: pd.DataFrame,
    benchmark: pd.Series,
    horizon: HorizonSpec,
    confidence: pd.DataFrame | None = None,
    config: RrgConfig | None = None,
) -> pd.DataFrame:
    cfg = config or RrgConfig(horizons=(horizon,))
    prices = sector_prices.astype(float).sort_index()
    aligned_benchmark = benchmark.reindex(prices.index).ffill().astype(float)
    relative_strength = prices.divide(aligned_benchmark.replace(0.0, np.nan), axis=0)
    log_rs = np.log(relative_strength.replace(0.0, np.nan))
    log_rs = smooth_frame(log_rs, method=cfg.smoothing_method, window=cfg.smoothing_window)

    trend = log_rs.rolling(cfg.trend_window, min_periods=1).mean()
    rs_centered = log_rs - trend
    mom = log_rs - log_rs.shift(horizon.periods)
    acc = mom - mom.shift(horizon.periods)
    acc_z = rolling_zscore(acc, window=cfg.acceleration_z_window)

    tidy = _to_tidy(
        frames={
            "rs": relative_strength,
            "log_rs": log_rs,
            "rs_centered": rs_centered,
            "mom": mom,
            "acc": acc,
            "acc_z": acc_z,
        },
        horizon=horizon.name,
    )
    if confidence is None:
        tidy["confidence"] = 1.0
    else:
        conf_tidy = (
            confidence.reindex(index=prices.index, columns=prices.columns)
            .stack(future_stack=True)
            .rename("confidence")
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "sector"})
        )
        tidy = tidy.merge(conf_tidy, on=["date", "sector"], how="left")
        tidy["confidence"] = tidy["confidence"].fillna(1.0).astype(float)

    tidy["state"] = [
        classify_rrg_state(rs_value, mom_value)
        for rs_value, mom_value in zip(tidy["rs_centered"], tidy["mom"], strict=True)
    ]
    tidy["turning_label"] = [
        classify_turning_point(mom=mom_value, acc_z=acc_value, threshold=cfg.turning_threshold)
        for mom_value, acc_value in zip(tidy["mom"], tidy["acc_z"], strict=True)
    ]
    return add_state_persistence(tidy)


def classify_rrg_state(rs_centered: float, mom: float) -> str:
    if pd.isna(rs_centered) or pd.isna(mom):
        return "Unclassified"
    if rs_centered >= 0.0 and mom >= 0.0:
        return "Leading"
    if rs_centered < 0.0 and mom >= 0.0:
        return "Improving"
    if rs_centered < 0.0 and mom < 0.0:
        return "Lagging"
    return "Weakening"


def classify_turning_point(*, mom: float, acc_z: float, threshold: float = 1.0) -> str:
    if pd.isna(mom) or pd.isna(acc_z):
        return "Unclassified"
    if abs(acc_z) < threshold:
        return "Neutral"
    if mom > 0.0 and acc_z > 0.0:
        return "Trend strengthening"
    if mom > 0.0 and acc_z < 0.0:
        return "Exhaustion risk"
    if mom < 0.0 and acc_z > 0.0:
        return "Recovery candidate"
    if mom < 0.0 and acc_z < 0.0:
        return "Breakdown pressure"
    return "Neutral"


def add_state_persistence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        result["persistence"] = pd.Series(dtype="int64")
        return result
    result = frame.sort_values(["horizon", "sector", "date"]).copy()
    persistence = pd.Series(0, index=result.index, dtype="int64")
    for (_horizon, _sector), group in result.groupby(["horizon", "sector"], sort=False):
        count = 0
        previous = None
        for idx, state in group["state"].items():
            if state == "Unclassified":
                count = 0
                previous = state
            elif state == previous:
                count += 1
            else:
                count = 1
                previous = state
            persistence.loc[idx] = count
    result["persistence"] = persistence
    return result.sort_index()


def _to_tidy(*, frames: dict[str, pd.DataFrame], horizon: str) -> pd.DataFrame:
    pieces = []
    for name, frame in frames.items():
        piece = (
            frame.stack(future_stack=True)
            .rename(name)
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "sector"})
        )
        pieces.append(piece)
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=["date", "sector"], how="outer")
    result.insert(2, "horizon", horizon)
    return result


def _empty_rrg_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "sector",
            "horizon",
            "rs",
            "log_rs",
            "rs_centered",
            "mom",
            "acc",
            "acc_z",
            "state",
            "turning_label",
            "persistence",
            "confidence",
        ]
    )
