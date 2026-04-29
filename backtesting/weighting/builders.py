from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from backtesting.specs.models import WeightingSpec

WeightingHook: TypeAlias = Callable[[WeightingSpec, pd.DataFrame, Mapping[str, pd.DataFrame]], pd.DataFrame]

_WEIGHTING_HOOKS: dict[str, WeightingHook] = {}


def register_weighting_hook(hook_id: str, hook: WeightingHook) -> None:
    if hook_id in _WEIGHTING_HOOKS:
        raise ValueError(f"weighting hook already registered: {hook_id}")
    _WEIGHTING_HOOKS[hook_id] = hook



def unregister_weighting_hook(hook_id: str) -> None:
    try:
        del _WEIGHTING_HOOKS[hook_id]
    except KeyError as exc:
        raise KeyError(f"unknown weighting hook_id: {hook_id}") from exc



def weighting_fields(spec: WeightingSpec) -> tuple[str, ...]:
    if spec.kind == "equal_weight":
        return ()
    if spec.kind == "market_cap":
        return ("market_cap",)
    if spec.kind == "float_market_cap":
        return ("float_market_cap",)
    if spec.kind == "score":
        return (_require_field(spec),)
    if spec.kind == "inverse_vol":
        return ("close",)
    if spec.kind == "explicit":
        return ()
    if spec.kind == "hook":
        return _optional_fields(spec.params.get("fields"))
    raise ValueError(f"unknown weighting kind: {spec.kind}")



def build_weights(spec: WeightingSpec, selection: pd.DataFrame, feature_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if spec.kind == "equal_weight":
        values = selection.astype(float)
    elif spec.kind == "market_cap":
        values = _weighted_by_feature(spec, selection, feature_frames, field="market_cap")
    elif spec.kind == "float_market_cap":
        values = _weighted_by_feature(spec, selection, feature_frames, field="float_market_cap")
    elif spec.kind == "score":
        values = _weighted_by_feature(spec, selection, feature_frames, field=_require_field(spec))
    elif spec.kind == "inverse_vol":
        close = _frame_for_field(feature_frames, "close").reindex(index=selection.index, columns=selection.columns)
        returns = close.pct_change(fill_method=None)
        vol = returns.rolling(20, min_periods=20).std()
        values = selection.astype(float) * _sanitize_positive(1.0 / vol)
    elif spec.kind == "explicit":
        explicit = _read_explicit_weights(Path(_require_path(spec)))
        values = explicit.reindex(index=selection.index, columns=selection.columns, fill_value=0.0)
    elif spec.kind == "hook":
        hook_id = _require_hook_id(spec)
        try:
            hook = _WEIGHTING_HOOKS[hook_id]
        except KeyError as exc:
            raise KeyError(f"unknown weighting hook_id: {hook_id}") from exc
        values = hook(spec, selection.copy(), feature_frames)
    else:
        raise ValueError(f"unknown weighting kind: {spec.kind}")

    aligned = pd.DataFrame(values, index=selection.index, columns=selection.columns, dtype=float)
    return _normalize(aligned, selection)



def _normalize(values: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    aligned = values.reindex(index=selection.index, columns=selection.columns, fill_value=0.0).astype(float)
    selected = selection.reindex(index=aligned.index, columns=aligned.columns, fill_value=False).astype(bool)
    masked = aligned.where(selected, 0.0)
    positive = _sanitize_positive(masked)
    row_sums = positive.sum(axis=1)
    normalized = positive.div(row_sums.where(row_sums.gt(0.0), other=1.0), axis=0)
    return normalized.where(row_sums.gt(0.0), 0.0).astype(float)



def _weighted_by_feature(
    spec: WeightingSpec,
    selection: pd.DataFrame,
    feature_frames: Mapping[str, pd.DataFrame],
    *,
    field: str,
) -> pd.DataFrame:
    frame = _frame_for_field(feature_frames, field).reindex(index=selection.index, columns=selection.columns)
    return selection.astype(float) * _sanitize_positive(frame)



def _frame_for_field(feature_frames: Mapping[str, pd.DataFrame], field: str) -> pd.DataFrame:
    try:
        return feature_frames[field]
    except KeyError as exc:
        raise KeyError(f"unknown weighting field: {field}") from exc



def _sanitize_positive(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.astype(float).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return sanitized.where(sanitized.gt(0.0), 0.0)



def _read_explicit_weights(path: Path) -> pd.DataFrame:
    _validate_explicit_header(path)
    explicit = pd.read_csv(path, index_col=0, keep_default_na=False)
    explicit.index = _normalize_explicit_index(explicit.index)
    explicit.columns = _normalize_explicit_columns(explicit.columns)
    numeric = explicit.apply(pd.to_numeric, errors="coerce")
    _validate_explicit_values(explicit, numeric)
    return numeric.fillna(0.0).astype(float)



def _validate_explicit_values(raw: pd.DataFrame, numeric: pd.DataFrame) -> None:
    non_empty = raw.map(lambda value: not _is_blank_cell(value))
    invalid = non_empty & numeric.isna()
    if not invalid.to_numpy().any():
        return
    locations = [f"{row_label}/{column_label}" for row_label, column_label in invalid.stack()[lambda mask: mask].index]
    raise ValueError(f"explicit weights contain invalid values at: {', '.join(locations)}")



def _is_blank_cell(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())



def _validate_explicit_header(path: Path) -> None:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("explicit weights file must include a header row") from exc
    if len(header) < 2:
        raise ValueError("explicit weights file must include at least one symbol column")
    _normalize_explicit_columns(pd.Index(header[1:]))



def _normalize_explicit_index(index: pd.Index) -> pd.DatetimeIndex:
    raw_index = pd.Index(index)
    if raw_index.hasnans:
        raise ValueError("explicit weights index must contain valid unique dates in ISO format (YYYY-MM-DD)")
    normalized_labels: list[str] = []
    for value in raw_index:
        if not isinstance(value, str):
            raise ValueError("explicit weights index must contain valid unique dates in ISO format (YYYY-MM-DD)")
        stripped = value.strip()
        if len(stripped) != 10 or stripped[4] != "-" or stripped[7] != "-":
            raise ValueError("explicit weights index must contain valid unique dates in ISO format (YYYY-MM-DD)")
        normalized_labels.append(stripped)
    normalized = pd.to_datetime(pd.Index(normalized_labels), format="%Y-%m-%d", errors="coerce")
    if normalized.isna().any():
        raise ValueError("explicit weights index must contain valid unique dates in ISO format (YYYY-MM-DD)")
    if normalized.has_duplicates:
        raise ValueError("explicit weights index must contain unique dates")
    return pd.DatetimeIndex(normalized)



def _normalize_explicit_columns(columns: pd.Index) -> pd.Index:
    normalized = pd.Index(columns)
    if normalized.hasnans:
        raise ValueError("explicit weights columns must not contain missing labels")
    stripped = normalized.map(lambda label: label.strip() if isinstance(label, str) else label)
    if any(isinstance(label, str) and not label for label in stripped):
        raise ValueError("explicit weights columns must not contain blank labels")
    normalized = pd.Index(stripped)
    if normalized.has_duplicates:
        raise ValueError("explicit weights columns must contain unique labels")
    return normalized



def _optional_fields(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("weighting kind 'hook' requires params.fields to be a list of field names")
    fields: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("weighting kind 'hook' requires params.fields to be a list of field names")
        field = item.strip()
        if field in fields:
            continue
        fields.append(field)
    return tuple(fields)



def _require_field(spec: WeightingSpec) -> str:
    if spec.field is None or not isinstance(spec.field, str) or not spec.field.strip():
        raise ValueError("weighting kind 'score' requires field")
    return spec.field.strip()



def _require_path(spec: WeightingSpec) -> str:
    if spec.path is None or not isinstance(spec.path, str) or not spec.path.strip():
        raise ValueError("weighting kind 'explicit' requires path")
    return spec.path



def _require_hook_id(spec: WeightingSpec) -> str:
    if spec.hook_id is None or not isinstance(spec.hook_id, str) or not spec.hook_id.strip():
        raise ValueError("weighting kind 'hook' requires hook_id")
    return spec.hook_id.strip()
