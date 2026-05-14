from __future__ import annotations

import pandas as pd

from .models import ExecutionSpec


def shorting_fields(spec: ExecutionSpec) -> tuple[str, ...]:
    if spec.shorting.shortable_field is not None:
        return (spec.shorting.shortable_field,)
    return ()


def apply_shorting(
    spec: ExecutionSpec,
    *,
    base_weights: pd.DataFrame,
    selection: pd.DataFrame,
    features: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    has_short_targets = bool(base_weights.lt(0.0).any().any())
    if has_short_targets and not spec.shorting.enabled:
        raise ValueError("negative target weights require shorting.enabled = true")

    if not spec.shorting.enabled or spec.shorting.shortable_field is None:
        return base_weights, selection

    shortable = (
        features[spec.shorting.shortable_field]
        .reindex(index=base_weights.index, columns=base_weights.columns)
        .fillna(False)
        .astype(bool)
    )
    blocked_short = base_weights.lt(0.0) & ~shortable
    if not bool(blocked_short.any().any()):
        return base_weights, selection

    adjusted_weights = base_weights.mask(blocked_short, 0.0).astype(float)
    adjusted_selection = selection.where(~blocked_short, False).astype(bool)
    return adjusted_weights, adjusted_selection
