from __future__ import annotations

import pandas as pd


def preprocess_factor_frame(
    raw: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    rank_transform: bool = False,
    winsor_quantile: float | None = None,
    zscore_cap: float | None = None,
) -> pd.DataFrame:
    raw = raw.reindex(index=float_mktcap.index, columns=float_mktcap.columns).astype(float)
    universe = universe.reindex(index=raw.index, columns=raw.columns).fillna(False).astype(bool)
    if winsor_quantile is not None:
        raw = _winsorize_by_row(raw, universe, winsor_quantile)
    weights = _normalized_weights(float_mktcap.reindex(index=raw.index, columns=raw.columns), universe)
    masked = raw.where(universe)
    observed_weights = weights.where(masked.notna(), 0.0)
    observed_weights = observed_weights.div(observed_weights.sum(axis=1).replace(0.0, float("nan")), axis=0).fillna(0.0)
    mean = (masked * observed_weights).sum(axis=1)
    filled = masked.T.fillna(mean).T.where(universe)
    if rank_transform:
        filled = filled.rank(axis=1, method="min", ascending=True).where(universe)
    centered = filled.sub((filled * weights).sum(axis=1), axis=0)
    std = centered.pow(2).mul(weights).sum(axis=1).pow(0.5).replace(0.0, float("nan"))
    zscore = centered.div(std, axis=0).fillna(0.0)
    if zscore_cap is not None:
        if zscore_cap <= 0.0:
            raise ValueError("zscore_cap must be positive")
        zscore = zscore.clip(lower=-zscore_cap, upper=zscore_cap)
    return zscore.where(universe, 0.0).astype(float)


def build_sector_active_exposures(
    sector: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    universe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    sector = sector.reindex(index=float_mktcap.index, columns=float_mktcap.columns).ffill()
    universe = universe.reindex(index=float_mktcap.index, columns=float_mktcap.columns).fillna(False).astype(bool)
    weights = _normalized_weights(float_mktcap, universe)
    sector_labels = sorted(sector.where(universe).stack().dropna().unique(), key=str)
    exposures: dict[str, pd.DataFrame] = {}
    for label in sector_labels:
        name = str(label)
        dummy = sector.eq(label).where(universe, False).astype(float)
        sector_weight = (dummy * weights).sum(axis=1)
        exposures[name] = dummy.sub(sector_weight, axis=0).where(universe, 0.0).astype(float)
    return exposures


def combine_exposures(
    alpha_factors: dict[str, pd.DataFrame],
    sector_factors: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> pd.DataFrame:
    frames = [frame.loc[date].rename(name) for name, frame in alpha_factors.items()]
    frames.extend(frame.loc[date].rename(name) for name, frame in sector_factors.items())
    return pd.concat(frames, axis=1).astype(float)


def _normalized_weights(float_mktcap: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    values = float_mktcap.astype(float).where(universe).clip(lower=0.0)
    total = values.sum(axis=1).replace(0.0, float("nan"))
    return values.div(total, axis=0).fillna(0.0)


def _winsorize_by_row(raw: pd.DataFrame, universe: pd.DataFrame, quantile: float) -> pd.DataFrame:
    if quantile <= 0.0 or quantile >= 0.5:
        raise ValueError("winsor_quantile must be between 0 and 0.5")
    masked = raw.where(universe)
    lower = masked.quantile(quantile, axis=1)
    upper = masked.quantile(1.0 - quantile, axis=1)
    return raw.clip(lower=lower, upper=upper, axis=0)
