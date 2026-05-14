from __future__ import annotations

from datetime import date
import json
from math import isfinite
from pathlib import Path

from .models import (
    ConditionSpec,
    DataPolicySpec,
    ExecutionSpec,
    PortfolioShapeSpec,
    PositionBucketSpec,
    PositionPolicySpec,
    PositionRuleSpec,
    ScheduleEvaluationSpec,
    ScheduleSpec,
    SelectionSpec,
    ShortingSpec,
    WeightingSpec,
    WeightSourceSpec,
)


def _read_date_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a YYYY-MM-DD date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a YYYY-MM-DD date string") from exc
    if value != parsed.isoformat():
        raise ValueError(f"{key} must be a YYYY-MM-DD date string")
    return value


def _read_date_tuple(payload: dict[str, object], key: str, error_key: str) -> tuple[str, ...]:
    if key not in payload:
        return ()
    values = payload[key]
    if not isinstance(values, list):
        raise ValueError(f"{error_key} must be a list")
    parsed: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{error_key} entries must be YYYY-MM-DD date strings")
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{error_key} entries must be YYYY-MM-DD date strings") from exc
        if value != parsed_date.isoformat():
            raise ValueError(f"{error_key} entries must be YYYY-MM-DD date strings")
        parsed.append(value)
    return tuple(parsed)


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


def _read_string_choice(
    payload: dict[str, object],
    key: str,
    *,
    default: str | None = None,
    error_key: str | None = None,
    allowed: set[str],
) -> str:
    if key not in payload:
        if default is None:
            return _read_required_string(payload, key, error_key)
        value = default
    else:
        value = _read_required_string(payload, key, error_key)
    if value not in allowed:
        label = error_key or key
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{label} must be one of: {allowed_values}")
    return value


def _read_int(
    payload: dict[str, object],
    key: str,
    default: int | None = None,
    *,
    error_key: str | None = None,
    min_value: int | None = None,
) -> int | None:
    if key not in payload:
        return default
    value = payload[key]
    label = error_key or key
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if min_value is not None and value < min_value:
        raise ValueError(f"{label} must be >= {min_value}")
    return value


def _read_float(
    payload: dict[str, object],
    key: str,
    default: float | None = None,
    *,
    error_key: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    if key not in payload:
        return default
    value = payload[key]
    label = error_key or key
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{label} must be >= {min_value:g}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{label} must be <= {max_value:g}")
    return parsed


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
                field=_read_required_string(raw_condition, "field", "selection.conditions.field"),
                op=_read_required_string(raw_condition, "op", "selection.conditions.op"),
                value=raw_condition.get("value"),
                other_field=_read_optional_string(raw_condition, "other_field", "selection.conditions.other_field"),
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
        n=_read_int(raw, "n", error_key="selection.n", min_value=1),
        top_n=_read_int(raw, "top_n", error_key="selection.top_n", min_value=1),
        bottom_n=_read_int(raw, "bottom_n", error_key="selection.bottom_n", min_value=1),
        ascending=_read_optional_bool(raw, "ascending", False, "selection.ascending"),
        threshold=_read_float(raw, "threshold", error_key="selection.threshold"),
        path=_read_optional_string(raw, "path", "selection.path"),
        hook_id=_read_optional_string(raw, "hook_id", "selection.hook_id"),
        params=_read_params(raw, "params", "selection.params"),
        hold_days=_read_int(raw, "hold_days", 0, error_key="selection.hold_days", min_value=0) or 0,
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


def _read_portfolio_shape(payload: dict[str, object]) -> PortfolioShapeSpec | None:
    raw = _read_object(payload, "portfolio_shape")
    if raw is None:
        return None
    return PortfolioShapeSpec(
        kind=_read_string_choice(
            raw,
            "kind",
            default="long_only",
            error_key="portfolio_shape.kind",
            allowed={"long_only", "long_short", "sector_neutral"},
        ),
        gross_long=_read_float(raw, "gross_long", 1.0, error_key="portfolio_shape.gross_long", min_value=0.0) or 0.0,
        gross_short=_read_float(raw, "gross_short", 1.0, error_key="portfolio_shape.gross_short", min_value=0.0) or 0.0,
        group_field=(
            _read_required_string(raw, "group_field", "portfolio_shape.group_field")
            if "group_field" in raw
            else "sector"
        ),
        group_budget=_read_string_choice(
            raw,
            "group_budget",
            default="equal_group",
            error_key="portfolio_shape.group_budget",
            allowed={"equal_group", "proportional_selected"},
        ),
    )


def _read_shorting(payload: dict[str, object]) -> ShortingSpec:
    raw = _read_object(payload, "shorting")
    if raw is None:
        return ShortingSpec()
    return ShortingSpec(
        enabled=_read_optional_bool(raw, "enabled", False, "shorting.enabled"),
        borrow_fee_annual=_read_float(raw, "borrow_fee_annual", 0.0, error_key="shorting.borrow_fee_annual", min_value=0.0) or 0.0,
        shortable_field=_read_optional_string(raw, "shortable_field", "shorting.shortable_field"),
        cash_collateral_ratio=_read_float(raw, "cash_collateral_ratio", 1.0, error_key="shorting.cash_collateral_ratio", min_value=0.0) or 0.0,
    )


def _read_position_rule(raw: object, default_kind: str, error_key: str) -> PositionRuleSpec:
    if raw is None:
        return PositionRuleSpec(kind=default_kind)
    if not isinstance(raw, dict):
        raise ValueError("position_policy rules must be objects")
    return PositionRuleSpec(
        kind=_read_required_string(raw, "kind", error_key) if "kind" in raw else default_kind,
        count=_read_int(raw, "count", 0, error_key=f"{error_key}.count", min_value=0) or 0,
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
                    id=_read_required_string(raw_bucket, "id", "position_policy.buckets.id"),
                    fraction=_read_float(
                        raw_bucket,
                        "fraction",
                        error_key="position_policy.buckets.fraction",
                        min_value=0.0,
                        max_value=1.0,
                    )
                    or 0.0,
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


def _read_schedule(payload: dict[str, object]) -> ScheduleSpec:
    raw = payload.get("schedule")
    if raw is None:
        return ScheduleSpec()
    if not isinstance(raw, dict):
        raise ValueError("schedule must be an object")
    evaluation = _read_schedule_evaluation(raw)
    return ScheduleSpec(
        kind=_read_string_choice(
            raw,
            "kind",
            default="named",
            error_key="schedule.kind",
            allowed={"named", "custom_dates", "signal_dates"},
        ),
        name=_read_optional_string(raw, "name", "schedule.name") if "name" in raw else "monthly",
        dates=_read_date_tuple(raw, "dates", "schedule.dates"),
        weight_change_tolerance=_read_float(
            raw,
            "weight_change_tolerance",
            1e-8,
            error_key="schedule.weight_change_tolerance",
            min_value=0.0,
        )
        or 0.0,
        evaluate_on_schedule=True,
        evaluation=evaluation,
    )


def _read_schedule_evaluation(raw_schedule: dict[str, object]) -> ScheduleEvaluationSpec | None:
    raw = raw_schedule.get("evaluation")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("schedule.evaluation must be an object")
    return ScheduleEvaluationSpec(
        kind=_read_string_choice(
            raw,
            "kind",
            default="named",
            error_key="schedule.evaluation.kind",
            allowed={"named", "custom_dates"},
        ),
        name=_read_optional_string(raw, "name", "schedule.evaluation.name") if "name" in raw else "daily",
        dates=_read_date_tuple(raw, "dates", "schedule.evaluation.dates"),
    )


def _read_weight_source(payload: dict[str, object]) -> WeightSourceSpec:
    raw = _read_object(payload, "weight_source")
    if raw is None:
        return WeightSourceSpec(kind="strategy")
    return WeightSourceSpec(
        kind=_read_string_choice(
            raw,
            "kind",
            default="strategy",
            error_key="weight_source.kind",
            allowed={"strategy", "hook", "dataset", "file"},
        ),
        hook_id=_read_optional_string(raw, "hook_id", "weight_source.hook_id"),
        dataset_id=_read_optional_string(raw, "dataset_id", "weight_source.dataset_id"),
        file_path=_read_optional_string(raw, "file_path", "weight_source.file_path"),
    )


def _read_data_policy(payload: dict[str, object]) -> DataPolicySpec:
    raw = _read_object(payload, "data_policy")
    if raw is None:
        return DataPolicySpec()
    return DataPolicySpec(**raw)


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

    if not isinstance(payload, dict):
        raise ValueError("execution spec must be an object")
    start = _read_date_string(payload, "start")
    end = _read_date_string(payload, "end")
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("start must be on or before end")

    selection = _read_selection(payload)

    return ExecutionSpec(
        start=start,
        end=end,
        capital=_read_float(payload, "capital", 100_000_000.0, min_value=0.0) or 0.0,
        strategy=_read_required_string(payload, "strategy") if "strategy" in payload else "trend_rank",
        name=_read_optional_string(payload, "name"),
        description=_read_optional_string(payload, "description"),
        top_n=_read_int(payload, "top_n", 20, min_value=1) or 20,
        lookback=_read_int(payload, "lookback", 20, min_value=1) or 20,
        flow_lookback=_read_int(payload, "flow_lookback", 20, min_value=1) or 20,
        momentum_lookback=_read_int(payload, "momentum_lookback", 60, min_value=1) or 60,
        liquidity_lookback=_read_int(payload, "liquidity_lookback", 20, min_value=1) or 20,
        momentum_weight=_read_float(payload, "momentum_weight", 0.5) or 0.0,
        schedule=_read_schedule(payload),
        fill_mode=_read_string_choice(
            payload,
            "fill_mode",
            default="next_open",
            allowed={"close", "next_open"},
        ),
        fee=_read_float(payload, "fee", 0.0, min_value=0.0) or 0.0,
        sell_tax=_read_float(payload, "sell_tax", 0.0, min_value=0.0) or 0.0,
        slippage=_read_float(payload, "slippage", 0.0, min_value=0.0) or 0.0,
        use_k200=_read_bool(payload, "use_k200", True),
        allow_fractional=_read_bool(payload, "allow_fractional", True),
        universe_id=_read_optional_string(payload, "universe_id"),
        benchmark_code=_read_optional_string(payload, "benchmark_code"),
        benchmark_name=_read_optional_string(payload, "benchmark_name"),
        benchmark_dataset=_read_optional_string(payload, "benchmark_dataset"),
        warmup_days=_read_int(payload, "warmup_days", 0, min_value=0) or 0,
        weight_source=_read_weight_source(payload),
        data_policy=_read_data_policy(payload),
        selection=selection,
        weighting=_read_weighting(payload, selection),
        portfolio_shape=_read_portfolio_shape(payload),
        shorting=_read_shorting(payload),
        position_policy=_read_position_policy(payload, selection),
        spec_source="spec_file",
        preset_id=_read_optional_string(payload, "preset_id"),
        notes=tuple(payload.get("notes", ())),
    )
