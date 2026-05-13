from __future__ import annotations

import json
from pathlib import Path

from .models import (
    ConditionSpec,
    DataPolicySpec,
    ExecutionSpec,
    PositionBucketSpec,
    PositionPolicySpec,
    PositionRuleSpec,
    ScheduleSpec,
    SelectionSpec,
    WeightingSpec,
    WeightSourceSpec,
)


def _read_bool(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean")


def _read_optional_bool(payload: dict[str, object], key: str, default: bool, error_key: str | None = None) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    label = error_key or key
    raise ValueError(f"{label} must be a boolean")


def _read_object(payload: dict[str, object], key: str) -> dict[str, object] | None:
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, dict):
        return value
    raise ValueError(f"{key} must be an object")


def _read_required_string(payload: dict[str, object], key: str, error_key: str | None = None) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    label = error_key or key
    raise ValueError(f"{label} must be a string")


def _read_optional_string(payload: dict[str, object], key: str, error_key: str | None = None) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, str):
        return value
    label = error_key or key
    raise ValueError(f"{label} must be a string")


def _read_params(payload: dict[str, object], key: str, error_key: str | None = None) -> dict[str, object]:
    if key not in payload:
        return {}
    value = payload[key]
    if isinstance(value, dict):
        return value
    label = error_key or key
    raise ValueError(f"{label} must be an object")


def _read_conditions(raw_conditions: object) -> tuple[ConditionSpec, ...]:
    if raw_conditions is None:
        return ()
    if not isinstance(raw_conditions, list):
        raise ValueError("selection.conditions must be a list")

    conditions: list[ConditionSpec] = []
    for raw_condition in raw_conditions:
        if not isinstance(raw_condition, dict):
            raise ValueError("selection.conditions entries must be objects")
        conditions.append(
            ConditionSpec(
                field=str(raw_condition["field"]),
                op=str(raw_condition["op"]),
                value=raw_condition.get("value"),
                other_field=(
                    str(raw_condition["other_field"])
                    if raw_condition.get("other_field") is not None
                    else None
                ),
            )
        )
    return tuple(conditions)


def _read_selection(payload: dict[str, object]) -> SelectionSpec | None:
    raw = _read_object(payload, "selection")
    if raw is None:
        return None

    return SelectionSpec(
        kind=_read_required_string(raw, "kind", "selection.kind"),
        field=_read_optional_string(raw, "field", "selection.field"),
        conditions=_read_conditions(raw.get("conditions")),
        n=int(raw["n"]) if raw.get("n") is not None else None,
        ascending=_read_optional_bool(raw, "ascending", False, "selection.ascending"),
        threshold=float(raw["threshold"]) if raw.get("threshold") is not None else None,
        path=_read_optional_string(raw, "path", "selection.path"),
        hook_id=_read_optional_string(raw, "hook_id", "selection.hook_id"),
        params=_read_params(raw, "params", "selection.params"),
        hold_days=int(raw.get("hold_days", 0)),
    )


def _read_weighting(payload: dict[str, object], selection: SelectionSpec | None) -> WeightingSpec | None:
    raw = _read_object(payload, "weighting")
    if raw is None:
        return WeightingSpec(kind="equal_weight") if selection is not None else None

    return WeightingSpec(
        kind=_read_required_string(raw, "kind", "weighting.kind") if "kind" in raw else "equal_weight",
        field=_read_optional_string(raw, "field", "weighting.field"),
        path=_read_optional_string(raw, "path", "weighting.path"),
        hook_id=_read_optional_string(raw, "hook_id", "weighting.hook_id"),
        params=_read_params(raw, "params", "weighting.params"),
    )


def _read_position_rule(raw: object, default_kind: str, error_key: str) -> PositionRuleSpec:
    if raw is None:
        return PositionRuleSpec(kind=default_kind)
    if not isinstance(raw, dict):
        raise ValueError("position_policy rules must be objects")
    return PositionRuleSpec(
        kind=_read_required_string(raw, "kind", error_key) if "kind" in raw else default_kind,
        count=int(raw.get("count", 0)),
    )


def _read_position_policy(payload: dict[str, object], selection: SelectionSpec | None) -> PositionPolicySpec | None:
    raw = _read_object(payload, "position_policy")
    if raw is None:
        return PositionPolicySpec(kind="pass_through") if selection is not None else None

    raw_buckets = raw.get("buckets")
    if raw_buckets is None:
        buckets: tuple[PositionBucketSpec, ...] = ()
    else:
        if not isinstance(raw_buckets, list):
            raise ValueError("position_policy.buckets must be a list")
        parsed_buckets: list[PositionBucketSpec] = []
        for raw_bucket in raw_buckets:
            if not isinstance(raw_bucket, dict):
                raise ValueError("position_policy.buckets entries must be objects")
            parsed_buckets.append(
                PositionBucketSpec(
                    id=str(raw_bucket["id"]),
                    fraction=float(raw_bucket["fraction"]),
                )
            )
        buckets = tuple(parsed_buckets)

    raw_rules = raw.get("rules")
    if raw_rules is None:
        rules: dict[str, object] = {}
    else:
        if not isinstance(raw_rules, dict):
            raise ValueError("position_policy.rules must be an object")
        rules = raw_rules

    raw_adds = rules.get("adds")
    if raw_adds is None:
        adds: tuple[PositionRuleSpec, ...] = ()
    else:
        if not isinstance(raw_adds, list):
            raise ValueError("position_policy.rules.adds must be a list")
        parsed_adds: list[PositionRuleSpec] = []
        for raw_add in raw_adds:
            if not isinstance(raw_add, dict):
                raise ValueError("position_policy.rules.adds entries must be objects")
            parsed_adds.append(_read_position_rule(raw_add, "still_passes_after_rebalances", "position_policy.rules.adds.kind"))
        adds = tuple(parsed_adds)

    return PositionPolicySpec(
        kind=_read_required_string(raw, "kind", "position_policy.kind") if "kind" in raw else "pass_through",
        buckets=buckets,
        entry=_read_position_rule(rules.get("entry"), "selection_passes", "position_policy.rules.entry.kind"),
        adds=adds,
        exit=_read_position_rule(rules.get("exit"), "selection_fails", "position_policy.rules.exit.kind"),
        hook_id=_read_optional_string(raw, "hook_id", "position_policy.hook_id"),
        params=_read_params(raw, "params", "position_policy.params"),
    )


def load_execution_spec(path: str | Path) -> ExecutionSpec:
    spec_path = Path(path)
    suffix = spec_path.suffix.lower()
    raw = spec_path.read_text(encoding="utf-8")

    if suffix == ".json":
        payload = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        raise ValueError("YAML spec loading is not available without an approved YAML dependency")
    else:
        raise ValueError(f"unsupported spec format: {suffix or '<none>'}")

    selection = _read_selection(payload)

    return ExecutionSpec(
        start=str(payload["start"]),
        end=str(payload["end"]),
        capital=float(payload.get("capital", 100_000_000.0)),
        strategy=str(payload.get("strategy", "trend_rank")),
        name=payload.get("name"),
        description=payload.get("description"),
        top_n=int(payload.get("top_n", 20)),
        lookback=int(payload.get("lookback", 20)),
        flow_lookback=int(payload.get("flow_lookback", 20)),
        momentum_lookback=int(payload.get("momentum_lookback", 60)),
        liquidity_lookback=int(payload.get("liquidity_lookback", 20)),
        momentum_weight=float(payload.get("momentum_weight", 0.5)),
        schedule=ScheduleSpec(**payload.get("schedule", {"kind": "named", "name": "monthly"})),
        fill_mode=str(payload.get("fill_mode", "next_open")),
        fee=float(payload.get("fee", 0.0)),
        sell_tax=float(payload.get("sell_tax", 0.0)),
        slippage=float(payload.get("slippage", 0.0)),
        use_k200=_read_bool(payload, "use_k200", True),
        allow_fractional=_read_bool(payload, "allow_fractional", True),
        universe_id=payload.get("universe_id"),
        benchmark_code=payload.get("benchmark_code"),
        benchmark_name=payload.get("benchmark_name"),
        benchmark_dataset=payload.get("benchmark_dataset"),
        warmup_days=int(payload.get("warmup_days", 0)),
        weight_source=WeightSourceSpec(**payload.get("weight_source", {"kind": "strategy"})),
        data_policy=DataPolicySpec(**payload.get("data_policy", {})),
        selection=selection,
        weighting=_read_weighting(payload, selection),
        position_policy=_read_position_policy(payload, selection),
        spec_source="spec_file",
        preset_id=payload.get("preset_id"),
        notes=tuple(payload.get("notes", ())),
    )
