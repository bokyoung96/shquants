from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from backtesting.data import MarketData
from backtesting.features import build_features
from backtesting.policy import build_position_plan_from_spec
from backtesting.policy.base import PositionPlan

from .models import ExecutionSpec, PositionPolicySpec


def build_position_plan_from_target_weights(
    spec: ExecutionSpec,
    market: MarketData,
) -> tuple[PositionPlan, dict[str, object]]:
    target_spec = spec.target_weights
    if target_spec is None:
        raise ValueError("target_weights plan requires target_weights spec")
    if target_spec.kind != "file":
        raise ValueError(f"unsupported target_weights kind: {target_spec.kind}")
    if target_spec.path is None or not target_spec.path.strip():
        raise ValueError("target_weights kind 'file' requires path")

    close = market.frames["close"]
    raw_weights = read_target_weights_csv(Path(target_spec.path))
    _reject_nonzero_unknown_symbols(raw_weights, close)
    weights = raw_weights.reindex(index=close.index, columns=close.columns, fill_value=0.0)
    weights = weights.fillna(0.0).astype(float)

    _validate_shorting_enabled(spec, weights)
    _validate_tradable_targets(weights, close, market)
    _validate_shortable_targets(spec, weights, market)

    selection = weights.ne(0.0)
    plan = build_position_plan_from_spec(
        PositionPolicySpec(kind="pass_through"),
        base_target_weights=weights,
        selection_mask=selection,
        market=market,
    )
    return plan, _exposure_metadata(weights)


def read_target_weights_csv(path: Path) -> pd.DataFrame:
    _validate_explicit_header(path)
    explicit = pd.read_csv(path, index_col=0, keep_default_na=False)
    explicit.index = _normalize_explicit_index(explicit.index)
    explicit.columns = _normalize_explicit_columns(explicit.columns)
    numeric = explicit.apply(pd.to_numeric, errors="coerce")
    _validate_explicit_values(explicit, numeric)
    return numeric.fillna(0.0).astype(float)


def _reject_nonzero_unknown_symbols(weights: pd.DataFrame, close: pd.DataFrame) -> None:
    unknown = [column for column in weights.columns if column not in close.columns]
    if not unknown:
        return
    unknown_values = weights.loc[:, unknown].fillna(0.0).astype(float)
    nonzero = unknown_values.ne(0.0)
    if not bool(nonzero.any().any()):
        return
    locations = _locations(nonzero)
    raise ValueError(f"target weights contain nonzero symbols absent from market data: {', '.join(locations)}")


def _validate_shorting_enabled(spec: ExecutionSpec, weights: pd.DataFrame) -> None:
    if bool(weights.lt(0.0).any().any()) and not spec.shorting.enabled:
        raise ValueError("negative target weights require shorting.enabled = true")


def _validate_tradable_targets(weights: pd.DataFrame, close: pd.DataFrame, market: MarketData) -> None:
    tradable = close.notna()
    if market.universe is not None:
        universe = market.universe.reindex(index=weights.index, columns=weights.columns)
        tradable = tradable & universe.fillna(False).astype(bool)
    blocked = weights.ne(0.0) & ~tradable.reindex(index=weights.index, columns=weights.columns, fill_value=False)
    if not bool(blocked.any().any()):
        return
    raise ValueError(f"target weights contain untradable target weights: {', '.join(_locations(blocked))}")


def _validate_shortable_targets(spec: ExecutionSpec, weights: pd.DataFrame, market: MarketData) -> None:
    if spec.shorting.shortable_field is None:
        return
    if not bool(weights.lt(0.0).any().any()):
        return
    shortable = build_features(market, (spec.shorting.shortable_field,))[spec.shorting.shortable_field]
    aligned = shortable.reindex(index=weights.index, columns=weights.columns).fillna(False).astype(bool)
    blocked = weights.lt(0.0) & ~aligned
    if not bool(blocked.any().any()):
        return
    raise ValueError(f"target weights contain unshortable short targets: {', '.join(_locations(blocked))}")


def _exposure_metadata(weights: pd.DataFrame) -> dict[str, object]:
    gross = weights.abs().sum(axis=1).astype(float)
    net = weights.sum(axis=1).astype(float)
    return {
        "plan_source": "target_weights",
        "avg_gross_exposure": float(gross.mean()) if not gross.empty else 0.0,
        "max_gross_exposure": float(gross.max()) if not gross.empty else 0.0,
        "avg_net_exposure": float(net.mean()) if not net.empty else 0.0,
        "min_net_exposure": float(net.min()) if not net.empty else 0.0,
        "max_net_exposure": float(net.max()) if not net.empty else 0.0,
    }


def _validate_explicit_values(raw: pd.DataFrame, numeric: pd.DataFrame) -> None:
    non_empty = raw.map(lambda value: not _is_blank_cell(value))
    finite = numeric.map(lambda value: pd.notna(value) and value not in {float("inf"), float("-inf")})
    invalid = non_empty & ~finite
    if not invalid.to_numpy().any():
        return
    raise ValueError(f"target weights contain invalid values at: {', '.join(_locations(invalid))}")


def _is_blank_cell(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _validate_explicit_header(path: Path) -> None:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("target weights file must include a header row") from exc
    if len(header) < 2:
        raise ValueError("target weights file must include at least one symbol column")
    _normalize_explicit_columns(pd.Index(header[1:]))


def _normalize_explicit_index(index: pd.Index) -> pd.DatetimeIndex:
    raw_index = pd.Index(index)
    if raw_index.hasnans:
        raise ValueError("target weights index must contain valid unique dates in ISO format (YYYY-MM-DD)")
    normalized_labels: list[str] = []
    for value in raw_index:
        if not isinstance(value, str):
            raise ValueError("target weights index must contain valid unique dates in ISO format (YYYY-MM-DD)")
        stripped = value.strip()
        if len(stripped) != 10 or stripped[4] != "-" or stripped[7] != "-":
            raise ValueError("target weights index must contain valid unique dates in ISO format (YYYY-MM-DD)")
        normalized_labels.append(stripped)
    normalized = pd.to_datetime(pd.Index(normalized_labels), format="%Y-%m-%d", errors="coerce")
    if normalized.isna().any():
        raise ValueError("target weights index must contain valid unique dates in ISO format (YYYY-MM-DD)")
    if normalized.has_duplicates:
        raise ValueError("target weights index must contain unique dates")
    return pd.DatetimeIndex(normalized)


def _normalize_explicit_columns(columns: pd.Index) -> pd.Index:
    normalized = pd.Index(columns)
    if normalized.hasnans:
        raise ValueError("target weights columns must not contain missing labels")
    stripped = normalized.map(lambda label: label.strip() if isinstance(label, str) else label)
    if any(isinstance(label, str) and not label for label in stripped):
        raise ValueError("target weights columns must not contain blank labels")
    normalized = pd.Index(stripped)
    if normalized.has_duplicates:
        raise ValueError("target weights columns must contain unique labels")
    return normalized


def _locations(mask: pd.DataFrame) -> list[str]:
    return [f"{row_label}/{column_label}" for row_label, column_label in mask.stack()[lambda values: values].index]
