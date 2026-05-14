from __future__ import annotations

import csv
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


def unregister_selection_hook(hook_id: str) -> None:
    try:
        del _SELECTION_HOOKS[hook_id]
    except KeyError as exc:
        raise KeyError(f"unknown selection hook_id: {hook_id}") from exc


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
    if spec.kind == "rank_top_bottom":
        return _build_rank_top_bottom(spec, feature_frames)
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


def _build_rank_top_bottom(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    field = _require_field(spec)
    top_n, bottom_n = _require_top_bottom(spec)
    scores = _frame_for_field(feature_frames, field)
    valid = scores.notna()
    long_rank = scores.rank(axis=1, ascending=False, method="first", na_option="bottom")
    selected_long = long_rank.le(top_n) & valid

    short_rank = scores.rank(axis=1, ascending=True, method="first", na_option="bottom")
    selected_short = short_rank.le(bottom_n) & valid & ~selected_long
    return (selected_long | selected_short).astype(bool)


def _build_score_threshold(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    field = _require_field(spec)
    threshold = _require_threshold(spec)
    return _frame_for_field(feature_frames, field).ge(threshold).astype(bool)


def _build_event(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    field = _require_field(spec)
    hold_days = _require_non_negative_int(spec.hold_days, "event", "hold_days")
    events = _frame_for_field(feature_frames, field).fillna(0).astype(bool)
    held = events.astype(int).rolling(window=hold_days + 1, min_periods=1).max().astype(bool)
    return held.astype(bool)


def _build_explicit(spec: SelectionSpec, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    path = _require_path(spec)
    explicit = _read_explicit_selection(Path(path))
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


def _read_explicit_selection(path: Path) -> pd.DataFrame:
    _validate_explicit_header(path)
    explicit = pd.read_csv(path, index_col=0, keep_default_na=False)
    explicit.index = _normalize_explicit_index(explicit.index)
    explicit.columns = _normalize_explicit_columns(explicit.columns)
    numeric = explicit.apply(pd.to_numeric, errors="coerce")
    _validate_explicit_values(explicit, numeric)
    return numeric


def _validate_explicit_values(raw: pd.DataFrame, numeric: pd.DataFrame) -> None:
    non_empty = raw.map(lambda value: not _is_blank_cell(value))
    invalid_numeric = non_empty & numeric.isna()
    invalid_binary = non_empty & numeric.notna() & ~numeric.isin((0.0, 1.0))
    invalid = invalid_numeric | invalid_binary
    if not invalid.to_numpy().any():
        return
    locations = [f"{row_label}/{column_label}" for row_label, column_label in invalid.stack()[lambda mask: mask].index]
    raise ValueError(f"explicit selection contains invalid values at: {', '.join(locations)}")


def _is_blank_cell(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _validate_explicit_header(path: Path) -> None:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("explicit selection file must include a header row") from exc
    if len(header) < 2:
        raise ValueError("explicit selection file must include at least one symbol column")
    _normalize_explicit_columns(pd.Index(header[1:]))


def _normalize_explicit_index(index: pd.Index) -> pd.DatetimeIndex:
    raw_index = pd.Index(index)
    if raw_index.hasnans:
        raise ValueError("explicit selection index must contain valid unique dates in ISO format (YYYY-MM-DD)")
    normalized_labels: list[str] = []
    for value in raw_index:
        if not isinstance(value, str):
            raise ValueError("explicit selection index must contain valid unique dates in ISO format (YYYY-MM-DD)")
        stripped = value.strip()
        if len(stripped) != 10 or stripped[4] != "-" or stripped[7] != "-":
            raise ValueError("explicit selection index must contain valid unique dates in ISO format (YYYY-MM-DD)")
        normalized_labels.append(stripped)
    normalized = pd.to_datetime(pd.Index(normalized_labels), format="%Y-%m-%d", errors="coerce")
    if normalized.isna().any():
        raise ValueError("explicit selection index must contain valid unique dates in ISO format (YYYY-MM-DD)")
    if normalized.has_duplicates:
        raise ValueError("explicit selection index must contain unique dates")
    return pd.DatetimeIndex(normalized)


def _normalize_explicit_columns(columns: pd.Index) -> pd.Index:
    normalized = pd.Index(columns)
    if normalized.hasnans:
        raise ValueError("explicit selection columns must not contain missing labels")
    stripped = normalized.map(lambda label: label.strip() if isinstance(label, str) else label)
    if any(isinstance(label, str) and not label for label in stripped):
        raise ValueError("explicit selection columns must not contain blank labels")
    normalized = pd.Index(stripped)
    if normalized.has_duplicates:
        raise ValueError("explicit selection columns must contain unique labels")
    return normalized


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
    value = _require_non_negative_int(spec.n, "rank_top_n", "n")
    if value <= 0:
        raise ValueError("selection kind 'rank_top_n' requires n > 0")
    return value


def _require_top_bottom(spec: SelectionSpec) -> tuple[int, int]:
    top_n = _require_non_negative_int(spec.top_n, "rank_top_bottom", "top_n")
    bottom_n = _require_non_negative_int(spec.bottom_n, "rank_top_bottom", "bottom_n")
    if top_n <= 0:
        raise ValueError("selection kind 'rank_top_bottom' requires top_n > 0")
    if bottom_n <= 0:
        raise ValueError("selection kind 'rank_top_bottom' requires bottom_n > 0")
    return top_n, bottom_n


def _require_threshold(spec: SelectionSpec) -> float:
    return _require_float(spec.threshold, "score_threshold", "threshold")




def _require_non_negative_int(value: object, kind: str, field_name: str) -> int:
    if value is None:
        raise ValueError(f"selection kind '{kind}' requires {field_name}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"selection kind '{kind}' requires integer {field_name}")
    if value < 0:
        raise ValueError(f"selection kind '{kind}' requires {field_name} >= 0")
    return value


def _require_float(value: object, kind: str, field_name: str) -> float:
    if value is None:
        raise ValueError(f"selection kind '{kind}' requires {field_name}")
    if isinstance(value, bool):
        raise ValueError(f"selection kind '{kind}' requires numeric {field_name}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"selection kind '{kind}' requires numeric {field_name}") from exc


def _require_path(spec: SelectionSpec) -> str:
    if spec.path is None:
        raise ValueError("selection kind 'explicit' requires path")
    return spec.path


def _require_hook_id(spec: SelectionSpec) -> str:
    if spec.hook_id is None:
        raise ValueError("selection kind 'hook' requires hook_id")
    return spec.hook_id
