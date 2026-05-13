from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from backtesting.catalog import DatasetId


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    field: str
    op: str
    value: object | None = None
    other_field: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    kind: str
    field: str | None = None
    conditions: tuple[ConditionSpec, ...] = ()
    n: int | None = None
    ascending: bool = False
    threshold: float | None = None
    path: str | None = None
    hook_id: str | None = None
    params: dict[str, object] = dc_field(default_factory=dict)
    hold_days: int = 0


@dataclass(frozen=True, slots=True)
class WeightingSpec:
    kind: str = "equal_weight"
    field: str | None = None
    path: str | None = None
    hook_id: str | None = None
    params: dict[str, object] = dc_field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PositionBucketSpec:
    id: str
    fraction: float


@dataclass(frozen=True, slots=True)
class PositionRuleSpec:
    kind: str
    count: int = 0


@dataclass(frozen=True, slots=True)
class PositionPolicySpec:
    kind: str = "pass_through"
    buckets: tuple[PositionBucketSpec, ...] = ()
    entry: PositionRuleSpec = dc_field(default_factory=lambda: PositionRuleSpec("selection_passes"))
    adds: tuple[PositionRuleSpec, ...] = ()
    exit: PositionRuleSpec = dc_field(default_factory=lambda: PositionRuleSpec("selection_fails"))
    hook_id: str | None = None
    params: dict[str, object] = dc_field(default_factory=dict)


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
    strategy: str = "trend_rank"
    name: str | None = None
    description: str | None = None
    top_n: int = 20
    lookback: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 60
    liquidity_lookback: int = 20
    momentum_weight: float = 0.5
    schedule: ScheduleSpec = dc_field(default_factory=ScheduleSpec)
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
    weight_source: WeightSourceSpec = dc_field(default_factory=WeightSourceSpec)
    data_policy: DataPolicySpec = dc_field(default_factory=DataPolicySpec)
    selection: SelectionSpec | None = None
    weighting: WeightingSpec | None = None
    position_policy: PositionPolicySpec | None = None
    spec_source: str = "cli"
    preset_id: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def uses_composable_plan(self) -> bool:
        return self.selection is not None or self.weighting is not None or self.position_policy is not None


@dataclass(frozen=True, slots=True)
class ResolvedExecutionSpec:
    execution: ExecutionSpec
    dataset_ids: tuple[DatasetId, ...]
    schedule: ScheduleSpec
    hook_id: str | None = None
    resolution_notes: tuple[str, ...] = ()
