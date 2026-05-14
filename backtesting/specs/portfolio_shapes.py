from __future__ import annotations

import pandas as pd

from backtesting.construction import LongShortTopBottom, SectorNeutralTopBottom
from backtesting.signals.base import SignalBundle

from .models import PortfolioShapeSpec, SelectionSpec


def portfolio_shape_fields(portfolio_shape: PortfolioShapeSpec | None) -> tuple[str, ...]:
    if portfolio_shape is not None and portfolio_shape.kind == "sector_neutral":
        return (portfolio_shape.group_field,)
    return ()


def build_portfolio_shape_construction(
    *,
    selection_spec: SelectionSpec,
    portfolio_shape: PortfolioShapeSpec,
    features: dict[str, pd.DataFrame],
):
    if selection_spec.kind != "rank_top_bottom":
        raise ValueError(f"portfolio_shape kind '{portfolio_shape.kind}' requires selection kind 'rank_top_bottom'")
    if selection_spec.field is None:
        raise ValueError("selection kind 'rank_top_bottom' requires field")
    if selection_spec.top_n is None or selection_spec.bottom_n is None:
        raise ValueError("selection kind 'rank_top_bottom' requires top_n and bottom_n")

    alpha = features[selection_spec.field]
    bundle = SignalBundle(alpha=alpha, context={})
    if portfolio_shape.kind == "long_short":
        return LongShortTopBottom(
            top_n=selection_spec.top_n,
            bottom_n=selection_spec.bottom_n,
            gross_long=portfolio_shape.gross_long,
            gross_short=portfolio_shape.gross_short,
        ).build(bundle)
    if portfolio_shape.kind == "sector_neutral":
        bundle = SignalBundle(
            alpha=alpha,
            context={"sector": features[portfolio_shape.group_field]},
        )
        return SectorNeutralTopBottom(
            top_n=selection_spec.top_n,
            bottom_n=selection_spec.bottom_n,
            group_budget=portfolio_shape.group_budget,
        ).build(bundle)
    raise ValueError(f"unknown portfolio_shape kind: {portfolio_shape.kind}")
