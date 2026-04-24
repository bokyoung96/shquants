from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.policy.pass_through import PassThroughPolicy
from backtesting.signals.base import SignalBundle

from .models import ResolvedExecutionSpec


@dataclass(frozen=True, slots=True)
class HookRegistration:
    hook_id: str
    required_datasets: tuple[DatasetId, ...]
    build_plan: Callable[..., "HookPlan"]


@dataclass(slots=True)
class HookPlan:
    position_plan: object
    schedule: pd.Series | None = None
    tradable: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)


_HOOKS: dict[str, HookRegistration] = {}


def register_hook(registration: HookRegistration) -> None:
    if registration.hook_id in _HOOKS:
        raise ValueError(f"hook already registered: {registration.hook_id}")
    _HOOKS[registration.hook_id] = registration


def get_hook(hook_id: str) -> HookRegistration:
    try:
        return _HOOKS[hook_id]
    except KeyError as exc:
        raise KeyError(f"unknown hook_id: {hook_id}") from exc


def second_thursday_flags(index: pd.DatetimeIndex) -> pd.Series:
    flags = pd.Series(False, index=index, dtype=bool)
    months = sorted({(ts.year, ts.month) for ts in index if ts.month in (6, 12)})
    normalized_index = index.normalize()
    for year, month in months:
        month_days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
        month_days = month_days[month_days.month == month]
        thursdays = month_days[month_days.weekday == 3]
        if len(thursdays) >= 2:
            flags.loc[normalized_index == thursdays[1]] = True
    return flags


def build_floatcap_weights(float_mcap: pd.DataFrame, universe: pd.DataFrame, schedule: pd.Series) -> pd.DataFrame:
    valid_caps = float_mcap.where(universe.astype(bool))
    denom = valid_caps.sum(axis=1).replace(0.0, pd.NA)
    target = valid_caps.div(denom, axis=0).fillna(0.0)
    weights = pd.DataFrame(0.0, index=target.index, columns=target.columns, dtype=float)

    last = pd.Series(0.0, index=target.columns, dtype=float)
    for ts in target.index:
        if bool(schedule.loc[ts]):
            last = target.loc[ts].fillna(0.0).astype(float)
        weights.loc[ts] = last
    return weights


def _semiannual_floatcap_plan(*, market, resolved_spec: ResolvedExecutionSpec, universe_spec=None) -> HookPlan:
    close = market.frames["close"]
    basis_key = "float_market_cap" if resolved_spec.execution.data_policy.resolved_weight_basis == "float_market_cap" else "market_cap"
    basis_frame = market.frames[basis_key].reindex_like(close)
    membership_key = "k200_yn"
    if universe_spec is not None and universe_spec.membership_dataset is not None and universe_spec.membership_dataset.value != DatasetId.QW_K200_YN.value:
        membership_key = "universe_membership"
    universe = market.frames[membership_key].fillna(0).astype(bool).reindex_like(close).fillna(False)
    schedule = second_thursday_flags(close.index)
    weights = build_floatcap_weights(basis_frame, universe, schedule)
    construction = ConstructionResult(
        base_target_weights=weights,
        selection_mask=weights.ne(0.0),
        group_long_budget=None,
        group_short_budget=None,
        meta={},
    )
    plan = PassThroughPolicy().apply(
        construction=construction,
        market=market,
        bundle=SignalBundle(alpha=weights, context={"tradable": weights.ne(0.0)}),
    )
    tradable = close.notna() & universe & basis_frame.notna()
    rebalance_dates = [ts.date().isoformat() for ts in schedule[schedule].index]
    return HookPlan(
        position_plan=plan,
        schedule=schedule,
        tradable=tradable,
        metadata={
            "requested_weight_basis": resolved_spec.execution.data_policy.requested_weight_basis,
            "resolved_weight_basis": resolved_spec.execution.data_policy.resolved_weight_basis,
            "rebalance_dates": rebalance_dates,
        },
    )


register_hook(
    HookRegistration(
        hook_id="kospi200_semiannual_floatcap",
        required_datasets=(DatasetId.QW_K200_YN,),
        build_plan=_semiannual_floatcap_plan,
    )
)
