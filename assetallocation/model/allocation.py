import numpy as np
import pandas as pd


def score_to_two_asset_weights(
    score: pd.Series,
    *,
    scale: float = 10.0,
    min_spy_weight: float = 0.0,
    max_spy_weight: float = 1.0,
    neutral_spy_weight: float = 0.5,
) -> pd.DataFrame:
    if min_spy_weight > max_spy_weight:
        raise ValueError("min_spy_weight must be less than or equal to max_spy_weight")

    spy_weight = neutral_spy_weight + score.fillna(0.0).astype(float) * float(scale)
    spy_weight = spy_weight.clip(lower=min_spy_weight, upper=max_spy_weight)
    spy_weight = spy_weight.where(score.notna(), neutral_spy_weight)
    weights = pd.DataFrame(
        {
            "SPY US Equity": spy_weight,
            "IEF US Equity": 1.0 - spy_weight,
        },
        index=score.index,
        dtype=float,
    )
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(
        {"SPY US Equity": neutral_spy_weight, "IEF US Equity": 1.0 - neutral_spy_weight}
    )
    return weights.round(12)


def rebalance_weekly(weights: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    if weights.empty:
        return weights.copy()

    weekly_updates = weights.resample(rule).last().dropna(how="all")
    combined = pd.concat([weights.iloc[[0]], weekly_updates]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    rebalanced = combined.reindex(weights.index.union(combined.index)).ffill().reindex(weights.index)
    return rebalanced.loc[:, weights.columns]
