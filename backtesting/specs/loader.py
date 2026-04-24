from __future__ import annotations

import json
from pathlib import Path

from .models import DataPolicySpec, ExecutionSpec, ScheduleSpec, WeightSourceSpec


def _read_bool(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean")


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

    return ExecutionSpec(
        start=str(payload["start"]),
        end=str(payload["end"]),
        capital=float(payload.get("capital", 100_000_000.0)),
        strategy=str(payload.get("strategy", "momentum")),
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
        spec_source="spec_file",
        preset_id=payload.get("preset_id"),
        notes=tuple(payload.get("notes", ())),
    )
