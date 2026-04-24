from __future__ import annotations

from dataclasses import dataclass, field

from backtesting.catalog import DatasetId


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    kind: str = "named"
    name: str | None = "monthly"
    dates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WeightSourceSpec:
    kind: str = "strategy"
    hook_id: str | None = None
    dataset_id: str | None = None
    file_path: str | None = None


@dataclass(frozen=True, slots=True)
class DataPolicySpec:
    requested_weight_basis: str | None = None
    resolved_weight_basis: str | None = None
    fallback_order: tuple[str, ...] = ()
    fallbacks_applied: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    start: str
    end: str
    capital: float = 100_000_000.0
    strategy: str = "momentum"
    name: str | None = None
    description: str | None = None
    top_n: int = 20
    lookback: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 60
    liquidity_lookback: int = 20
    momentum_weight: float = 0.5
    schedule: ScheduleSpec = field(default_factory=ScheduleSpec)
    fill_mode: str = "next_open"
    fee: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    use_k200: bool = True
    allow_fractional: bool = True
    universe_id: str | None = None
    benchmark_code: str | None = None
    benchmark_name: str | None = None
    benchmark_dataset: str | None = None
    warmup_days: int = 0
    weight_source: WeightSourceSpec = field(default_factory=WeightSourceSpec)
    data_policy: DataPolicySpec = field(default_factory=DataPolicySpec)
    spec_source: str = "cli"
    preset_id: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedExecutionSpec:
    execution: ExecutionSpec
    dataset_ids: tuple[DatasetId, ...]
    schedule: ScheduleSpec
    hook_id: str | None = None
    resolution_notes: tuple[str, ...] = ()
