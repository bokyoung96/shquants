from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from backtesting.specs.models import ConditionSpec, SelectionSpec

SelectionHook: TypeAlias = Callable[[SelectionSpec, Mapping[str, pd.DataFrame]], pd.DataFrame]

_SELECTION_HOOKS: dict[str, SelectionHook] = {}
_SUPPORTED_OPERATORS = {">", ">=", "<", "<=", "==", "!=", "notna", "isna"}


def register_selection_hook(hook_id: str, hook: SelectionHook) -> None:
    if hook_id in _SELECTION_HOOKS:
        raise ValueError(f"selection hook already registered: {hook_id}")
    _SELECTION_HOOKS[hook_id] = hook


def selection_fields(spec: SelectionSpec) -> tuple[str, ...]:
    ordered: list[str] = []
    for field in (spec.field, *(condition.field for condition in spec.conditions), *(condition.other_field for condition in spec.conditions)):
        if field is None or field in ordered:
            continue
        ordered.append(field)
    return tuple(ordered)


def build_selection(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if spec.kind == "filter":
        return _build_filter(spec, feature_frames)
    if spec.kind == "rank_top_n":
        return _build_rank_top_n(spec, feature_frames)
    if spec.kind == "score_threshold":
        return _build_score_threshold(spec, feature_frames)
    if spec.kind == "event":
        return _build_event(spec, feature_frames)
    if spec.kind == "explicit":
        return _build_explicit(spec, feature_frames)
    if spec.kind == "hook":
        return _build_hook(spec, feature_frames)
    raise ValueError(f"unknown selection kind: {spec.kind}")


def _build_filter(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    anchor = _selection_anchor(spec, feature_frames)
    mask = pd.DataFrame(True, index=anchor.index, columns=anchor.columns, dtype=bool)
    for condition in spec.conditions:
        evaluated = _evaluate_condition(condition, feature_frames).reindex(index=anchor.index, columns=anchor.columns, fill_value=False)
        mask &= evaluated
    return mask.astype(bool)


def _build_rank_top_n(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    field = _require_field(spec)
    n = _require_positive_n(spec)
    scores = _frame_for_field(feature_frames, field)
    ranks = scores.rank(axis=1, ascending=spec.ascending, method="first", na_option="bottom")
    selected = ranks.le(n) & scores.notna()
    return selected.astype(bool)


def _build_score_threshold(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    field = _require_field(spec)
    threshold = _require_threshold(spec)
    return _frame_for_field(feature_frames, field).ge(threshold).astype(bool)


def _build_event(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    field = _require_field(spec)
    if spec.hold_days < 0:
        raise ValueError("event selection requires hold_days >= 0")
    events = _frame_for_field(feature_frames, field).fillna(0).astype(bool)
    held = events.astype(int).rolling(window=spec.hold_days + 1, min_periods=1).max().astype(bool)
    return held.astype(bool)


def _build_explicit(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    path = _require_path(spec)
    explicit = pd.read_csv(Path(path), index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")
    anchor = _selection_anchor(spec, feature_frames, fallback=explicit)
    aligned = explicit.reindex(index=anchor.index, columns=anchor.columns, fill_value=0.0)
    return aligned.fillna(0.0).astype(bool)


def _build_hook(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    hook_id = _require_hook_id(spec)
    try:
        hook = _SELECTION_HOOKS[hook_id]
    except KeyError as exc:
        raise KeyError(f"unknown selection hook_id: {hook_id}") from exc
    selection = hook(spec, feature_frames)
    anchor = _selection_anchor(spec, feature_frames, fallback=selection)
    return selection.reindex(index=anchor.index, columns=anchor.columns, fill_value=False).fillna(False).astype(bool)


def _selection_anchor(
    spec: SelectionSpec,
    feature_frames: Mapping[str, pd.DataFrame],
    *,
    fallback: pd.DataFrame | None = None,
) -> pd.DataFrame:
    for field in selection_fields(spec):
        if field in feature_frames:
            return feature_frames[field]
    try:
        return next(iter(feature_frames.values()))
    except StopIteration:
        if fallback is not None:
            return fallback
        raise ValueError("selection requires at least one feature frame for index/column alignment")


def _evaluate_condition(condition: ConditionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    left = _frame_for_field(feature_frames, condition.field)
    if condition.op == "notna":
        return left.notna()
    if condition.op == "isna":
        return left.isna()
    right: pd.DataFrame | object
    if condition.other_field is not None:
        right = _frame_for_field(feature_frames, condition.other_field).reindex_like(left)
    else:
        right = condition.value
    if condition.op == ">":
        return left.gt(right)
    if condition.op == ">=":
        return left.ge(right)
    if condition.op == "<":
        return left.lt(right)
    if condition.op == "<=":
        return left.le(right)
    if condition.op == "==":
        return left.eq(right)
    if condition.op == "!=":
        return left.ne(right)
    raise ValueError(
        f"unsupported condition operator: {condition.op}; supported operators: {sorted(_SUPPORTED_OPERATORS)}"
    )


def _frame_for_field(feature_frames: Mapping[str, pd.DataFrame], field: str) -> pd.DataFrame:
    try:
        return feature_frames[field]
    except KeyError as exc:
        raise KeyError(f"unknown selection field: {field}") from exc


def _require_field(spec: SelectionSpec) -> str:
    if spec.field is None:
        raise ValueError(f"selection kind '{spec.kind}' requires field")
    return spec.field


def _require_positive_n(spec: SelectionSpec) -> int:
    if spec.n is None:
        raise ValueError("selection kind 'rank_top_n' requires n")
    if spec.n <= 0:
        raise ValueError("selection kind 'rank_top_n' requires n > 0")
    return spec.n


def _require_threshold(spec: SelectionSpec) -> float:
    if spec.threshold is None:
        raise ValueError("selection kind 'score_threshold' requires threshold")
    return spec.threshold


def _require_path(spec: SelectionSpec) -> str:
    if spec.path is None:
        raise ValueError("selection kind 'explicit' requires path")
    return spec.path


def _require_hook_id(spec: SelectionSpec) -> str:
    if spec.hook_id is None:
        raise ValueError("selection kind 'hook' requires hook_id")
    return spec.hook_id
