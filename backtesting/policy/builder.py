from __future__ import annotations

import pandas as pd

from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle
from backtesting.specs.models import PositionPolicySpec, PositionRuleSpec

from .base import PositionPlan
from .pass_through import PassThroughPolicy
from .staged import BudgetPreservingStagedPolicy, BucketDefinition, StagedRuleSet


def build_position_plan_from_spec(
    spec: PositionPolicySpec,
    base_target_weights: pd.DataFrame,
    selection_mask: pd.DataFrame,
    market: MarketData,
) -> PositionPlan:
    aligned_base = base_target_weights.copy().astype(float)
    aligned_selection = selection_mask.reindex(index=aligned_base.index, columns=aligned_base.columns).fillna(False)
    aligned_selection = aligned_selection.astype(bool)

    construction = ConstructionResult(
        base_target_weights=aligned_base,
        selection_mask=aligned_selection,
        group_long_budget=None,
        group_short_budget=None,
        meta={},
    )

    if spec.kind == "pass_through":
        return PassThroughPolicy().apply(
            construction,
            market,
            SignalBundle(alpha=aligned_base, context={}),
        )
    if spec.kind == "staged":
        return _build_staged(spec, construction=construction, market=market)
    if spec.kind == "hook":
        raise ValueError(
            "unsupported position policy kind 'hook' in builder; use weight_source.kind == 'hook' for full-plan hooks"
        )
    raise ValueError(f"unknown position policy kind: {spec.kind}")



def _build_staged(
    spec: PositionPolicySpec,
    *,
    construction: ConstructionResult,
    market: MarketData,
) -> PositionPlan:
    if not spec.buckets:
        raise ValueError("staged position policy requires at least one bucket")
    if spec.entry is None or spec.exit is None:
        raise ValueError("staged position policy requires entry and exit rules")

    expected_adds = max(len(spec.buckets) - 1, 0)
    if len(spec.adds) != expected_adds:
        raise ValueError("staged position policy requires one add rule for each bucket after the first")

    buckets = tuple(BucketDefinition(bucket.id, bucket.fraction) for bucket in spec.buckets)

    entry_key, entry_frame = _rule_context("entry", spec.entry, construction.selection_mask)
    add_pairs = tuple(
        _rule_context(f"add_{index}", rule, construction.selection_mask)
        for index, rule in enumerate(spec.adds, start=1)
    )
    exit_key, exit_frame = _rule_context("exit", spec.exit, construction.selection_mask)

    bundle = SignalBundle(
        alpha=construction.base_target_weights,
        context={
            entry_key: entry_frame,
            **{key: frame for key, frame in add_pairs},
            exit_key: exit_frame,
        },
    )

    return BudgetPreservingStagedPolicy(
        buckets=buckets,
        rules=StagedRuleSet(
            entry_key=entry_key,
            add_keys=tuple(key for key, _ in add_pairs),
            exit_key=exit_key,
        ),
    ).apply(construction, market, bundle)



def _rule_key(stage: str, rule: PositionRuleSpec) -> str:
    return f"position_policy.{stage}.{rule.kind}.{rule.count}"



def _rule_context(
    stage: str,
    rule: PositionRuleSpec | None,
    selection_mask: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    if rule is None:
        raise ValueError(f"missing position rule for stage: {stage}")

    aligned_selection = selection_mask.fillna(False).astype(bool)
    key = _rule_key(stage, rule)

    if rule.kind == "selection_passes":
        return key, aligned_selection
    if rule.kind == "selection_fails":
        return key, ~aligned_selection
    if rule.kind == "still_passes_after_rebalances":
        if rule.count < 0:
            raise ValueError("still_passes_after_rebalances count must be non-negative")
        shifted = aligned_selection.shift(rule.count, fill_value=False).astype(bool)
        return key, aligned_selection & shifted

    raise ValueError(f"unsupported position rule kind: {rule.kind}")
