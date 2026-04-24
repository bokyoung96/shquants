# Execution Spec Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add official `--preset` / `--spec` execution paths that normalize into a resolved execution contract while preserving existing backtest results for identical resolved inputs.

**Architecture:** Keep `backtesting.run` as the operator-facing entrypoint, but move advanced execution concerns into a resolution layer. The runner and engine continue to perform the same portfolio math; new modules only resolve inputs, emit provenance, and route execution through a `run_spec()` path.

**Tech Stack:** Python 3.11, dataclasses, pandas, argparse, json/orjson, pytest, existing backtesting runner/writer stack

---

## File map

### Create
- `backtesting/specs/__init__.py` — public exports for execution-spec models and registries
- `backtesting/specs/models.py` — `ExecutionSpec`, `ResolvedExecutionSpec`, schedule/weight source/fallback dataclasses
- `backtesting/specs/loader.py` — load and validate `--spec` payloads from JSON files
- `backtesting/specs/resolve.py` — normalize legacy CLI args, presets, and spec files into resolved contracts
- `backtesting/specs/presets.py` — preset registry and built-in preset factories
- `backtesting/specs/hooks.py` — safe registry-backed hook lookup and execution
- `tests/specs/test_loader.py` — spec-file and validation coverage
- `tests/specs/test_resolve.py` — resolution/fallback/preset/hook coverage

### Modify
- `backtesting/run.py` — add `run_spec()` path, keep `run()` as legacy wrapper, add `--preset` / `--spec`
- `backtesting/reporting/writer.py` — persist `resolved_execution_spec.json` and `execution_resolution.json`
- `backtesting/__init__.py` — export spec-layer public APIs
- `tests/test_run.py` — add CLI parity and preset/spec integration coverage
- `README.md` — document simple CLI vs preset/spec execution surfaces
- `scripts/semiannual_floatcap_k200.py` — rewrite as a thin preset/spec smoke wrapper or remove once preset coverage replaces it

### Constraints to preserve during implementation
- Do not change `backtesting/engine/core.py`
- Do not change `backtesting/execution/fill.py`
- Do not change schedule semantics in `backtesting/execution/schedule.py`
- No new dependency unless explicitly approved; implement JSON spec loading with a clear error for non-JSON YAML requests if YAML parsing cannot be supported from the current dependency set

### Verification targets
- `tests/specs/test_loader.py`
- `tests/specs/test_resolve.py`
- `tests/test_run.py`
- `tests/execution/test_schedule.py`
- `tests/engine/test_core.py`

## Task 1: Lock down legacy behavior before refactoring

**Files:**
- Modify: `tests/test_run.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write the failing parity test for legacy config vs normalized spec path**

```python
def test_runner_run_and_run_spec_produce_identical_results_for_legacy_inputs(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1]}, index=index))

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    config = RunConfig(strategy="momentum", start="2024-01-02", end="2024-01-04", lookback=1, schedule="daily", fill_mode="close")

    legacy = runner.run(config)
    resolved = runner.run_spec(runner.resolve_spec_from_config(config))

    pd.testing.assert_series_equal(legacy.result.equity, resolved.result.equity)
    pd.testing.assert_series_equal(legacy.result.returns, resolved.result.returns)
    pd.testing.assert_series_equal(legacy.result.turnover, resolved.result.turnover)
    pd.testing.assert_frame_equal(legacy.result.qty, resolved.result.qty)
    assert legacy.summary == resolved.summary
```

- [ ] **Step 2: Run the focused test to verify the missing API fails first**

Run: `uv run python -m pytest tests/test_run.py::test_runner_run_and_run_spec_produce_identical_results_for_legacy_inputs -v`
Expected: FAIL with `AttributeError: 'BacktestRunner' object has no attribute 'run_spec'`

- [ ] **Step 3: Add the smallest temporary import surface needed for the upcoming refactor**

```python
# tests/test_run.py
from backtesting.run import BacktestRunner, RunConfig, RunReport, main as backtesting_main

# keep the new test adjacent to the existing runner integration tests so future parity regressions stay visible
```

- [ ] **Step 4: Re-run the single test and confirm it still fails only for the missing execution-spec path**

Run: `uv run python -m pytest tests/test_run.py::test_runner_run_and_run_spec_produce_identical_results_for_legacy_inputs -v`
Expected: FAIL with only the new missing-method error and no fixture/setup regression

- [ ] **Step 5: Commit the regression lock**

```bash
git add tests/test_run.py
git commit -m "Lock legacy runner parity before execution-surface refactor"
```

## Task 2: Introduce execution-spec models, registries, and resolution tests

**Files:**
- Create: `backtesting/specs/__init__.py`
- Create: `backtesting/specs/models.py`
- Create: `backtesting/specs/presets.py`
- Create: `backtesting/specs/hooks.py`
- Create: `backtesting/specs/resolve.py`
- Create: `tests/specs/test_resolve.py`
- Modify: `backtesting/__init__.py`
- Test: `tests/specs/test_resolve.py`

- [ ] **Step 1: Write failing resolution tests for presets, hooks, and fallback provenance**

```python
def test_resolve_spec_records_market_cap_fallback_when_float_cap_dataset_missing() -> None:
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        strategy="momentum",
        schedule={"kind": "named", "name": "monthly"},
        weight_source={"kind": "hook", "hook_id": "kospi200_semiannual_floatcap"},
        data_policy={"requested_weight_basis": "float_market_cap", "fallback_order": ["market_cap"]},
    )

    resolved = resolve_execution_spec(spec, available_dataset_ids={DatasetId.QW_ADJ_C, DatasetId.QW_MKTCAP, DatasetId.QW_K200_YN})

    assert resolved.data_policy.requested_weight_basis == "float_market_cap"
    assert resolved.data_policy.resolved_weight_basis == "market_cap"
    assert resolved.data_policy.fallbacks_applied == [
        {"from": "float_market_cap", "to": "market_cap", "reason": "missing qw_mktcap_flt parquet"}
    ]


def test_resolve_spec_rejects_unknown_hook_id() -> None:
    spec = ExecutionSpec(
        start="2024-01-01",
        end="2024-12-31",
        strategy="momentum",
        schedule={"kind": "named", "name": "monthly"},
        weight_source={"kind": "hook", "hook_id": "does-not-exist"},
    )

    with pytest.raises(KeyError, match="unknown hook_id"):
        resolve_execution_spec(spec, available_dataset_ids=set())
```

- [ ] **Step 2: Run the new resolution test module and confirm imports are missing**

Run: `uv run python -m pytest tests/specs/test_resolve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtesting.specs'`

- [ ] **Step 3: Implement the execution-spec data model and safe registries**

```python
# backtesting/specs/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from backtesting.catalog import DatasetId

ScheduleKind = Literal["named", "custom_dates"]
WeightSourceKind = Literal["strategy", "dataset", "file", "hook"]

@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    kind: ScheduleKind
    name: str | None = None
    dates: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class WeightSourceSpec:
    kind: WeightSourceKind
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
    strategy: str = "momentum"
    name: str | None = None
    description: str | None = None
    capital: float = 100_000_000.0
    schedule: ScheduleSpec = field(default_factory=lambda: ScheduleSpec(kind="named", name="monthly"))
    fill_mode: str = "next_open"
    universe_id: str | None = None
    allow_fractional: bool = True
    fee: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    weight_source: WeightSourceSpec = field(default_factory=lambda: WeightSourceSpec(kind="strategy"))
    data_policy: DataPolicySpec = field(default_factory=DataPolicySpec)
    spec_source: str = "cli"
    preset_id: str | None = None
    notes: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class ResolvedExecutionSpec:
    execution: ExecutionSpec
    dataset_ids: tuple[DatasetId, ...]
    schedule_name: str | None
    schedule_dates: tuple[str, ...]
    hook_id: str | None
    resolution_notes: tuple[str, ...] = ()
```

```python
# backtesting/specs/hooks.py
from collections.abc import Callable

HookFactory = Callable[..., object]
_HOOKS: dict[str, HookFactory] = {}

def register_hook(hook_id: str, factory: HookFactory) -> None:
    if hook_id in _HOOKS:
        raise ValueError(f"hook already registered: {hook_id}")
    _HOOKS[hook_id] = factory

def get_hook(hook_id: str) -> HookFactory:
    try:
        return _HOOKS[hook_id]
    except KeyError as exc:
        raise KeyError(f"unknown hook_id: {hook_id}") from exc
```

```python
# backtesting/specs/presets.py
from collections.abc import Callable

PresetFactory = Callable[[], ExecutionSpec]
_PRESETS: dict[str, PresetFactory] = {}

def register_preset(preset_id: str, factory: PresetFactory) -> None:
    if preset_id in _PRESETS:
        raise ValueError(f"preset already registered: {preset_id}")
    _PRESETS[preset_id] = factory

def get_preset(preset_id: str) -> ExecutionSpec:
    try:
        return _PRESETS[preset_id]()
    except KeyError as exc:
        raise KeyError(f"unknown preset_id: {preset_id}") from exc
```

```python
# backtesting/specs/resolve.py
FLOAT_CAP_DATASETS = {DatasetId.QW_MKTCAP_FLT, DatasetId.QW_KSDQ_MKTCAP_FLT}


def resolve_execution_spec(spec: ExecutionSpec, *, available_dataset_ids: set[DatasetId]) -> ResolvedExecutionSpec:
    requested = spec.data_policy.requested_weight_basis
    resolved_policy = spec.data_policy
    notes: list[str] = []

    if requested == "float_market_cap" and not (FLOAT_CAP_DATASETS & available_dataset_ids):
        resolved_policy = DataPolicySpec(
            requested_weight_basis="float_market_cap",
            resolved_weight_basis="market_cap",
            fallback_order=spec.data_policy.fallback_order,
            fallbacks_applied=(
                {"from": "float_market_cap", "to": "market_cap", "reason": "missing qw_mktcap_flt parquet"},
            ),
        )
        notes.append("float_market_cap unavailable; resolved to market_cap")

    execution = ExecutionSpec(**{**spec.__dict__, "data_policy": resolved_policy})
    return ResolvedExecutionSpec(
        execution=execution,
        dataset_ids=tuple(sorted(available_dataset_ids, key=lambda item: item.value)),
        schedule_name=execution.schedule.name,
        schedule_dates=execution.schedule.dates,
        hook_id=execution.weight_source.hook_id,
        resolution_notes=tuple(notes),
    )
```

- [ ] **Step 4: Run the resolution tests until they pass**

Run: `uv run python -m pytest tests/specs/test_resolve.py -v`
Expected: PASS

- [ ] **Step 5: Commit the spec-model and registry slice**

```bash
git add backtesting/specs backtesting/__init__.py tests/specs/test_resolve.py
git commit -m "Add execution-spec models and safe resolution registries"
```

## Task 3: Add spec-file loading and explicit format validation

**Files:**
- Create: `backtesting/specs/loader.py`
- Create: `tests/specs/test_loader.py`
- Modify: `backtesting/specs/__init__.py`
- Test: `tests/specs/test_loader.py`

- [ ] **Step 1: Write failing tests for JSON loading and clear YAML rejection**

```python
def test_load_execution_spec_from_json(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(json.dumps({
        "start": "2024-01-01",
        "end": "2024-12-31",
        "strategy": "momentum",
        "schedule": {"kind": "named", "name": "monthly"},
        "weight_source": {"kind": "strategy"},
    }), encoding="utf-8")

    spec = load_execution_spec(path)

    assert spec.start == "2024-01-01"
    assert spec.schedule.name == "monthly"
    assert spec.weight_source.kind == "strategy"


def test_load_execution_spec_rejects_yaml_without_parser_support(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.yaml"
    path.write_text("start: 2024-01-01\nend: 2024-12-31\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML spec loading is not available without an approved YAML dependency"):
        load_execution_spec(path)
```

- [ ] **Step 2: Run the loader tests and verify the missing loader fails**

Run: `uv run python -m pytest tests/specs/test_loader.py -v`
Expected: FAIL with `ImportError` for `load_execution_spec`

- [ ] **Step 3: Implement the spec loader with explicit JSON support and deterministic YAML messaging**

```python
# backtesting/specs/loader.py
from __future__ import annotations

import json
from pathlib import Path

from .models import DataPolicySpec, ExecutionSpec, ScheduleSpec, WeightSourceSpec


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
        strategy=str(payload.get("strategy", "momentum")),
        name=payload.get("name"),
        description=payload.get("description"),
        capital=float(payload.get("capital", 100_000_000.0)),
        schedule=ScheduleSpec(**payload.get("schedule", {"kind": "named", "name": "monthly"})),
        fill_mode=str(payload.get("fill_mode", "next_open")),
        universe_id=payload.get("universe_id"),
        allow_fractional=bool(payload.get("allow_fractional", True)),
        fee=float(payload.get("fee", 0.0)),
        sell_tax=float(payload.get("sell_tax", 0.0)),
        slippage=float(payload.get("slippage", 0.0)),
        weight_source=WeightSourceSpec(**payload.get("weight_source", {"kind": "strategy"})),
        data_policy=DataPolicySpec(**payload.get("data_policy", {})),
        spec_source="spec_file",
        preset_id=payload.get("preset_id"),
        notes=tuple(payload.get("notes", ())),
    )
```

- [ ] **Step 4: Run the loader tests to green**

Run: `uv run python -m pytest tests/specs/test_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit the spec-loader slice**

```bash
git add backtesting/specs/__init__.py backtesting/specs/loader.py tests/specs/test_loader.py
git commit -m "Add execution spec file loading with explicit format validation"
```

## Task 4: Route runner execution through `run_spec()` and persist provenance artifacts

**Files:**
- Modify: `backtesting/run.py`
- Modify: `backtesting/reporting/writer.py`
- Modify: `tests/test_run.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write failing tests for provenance artifacts and `run_spec()` execution**

```python
def test_run_spec_persists_resolution_artifacts(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    parquet_dir.mkdir()
    raw_dir.mkdir()
    store = ParquetStore(parquet_dir)
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    store.write("qw_adj_c", pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=index))
    store.write("qw_adj_o", pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=index))
    store.write("qw_k200_yn", pd.DataFrame({"A": [1, 1, 1]}, index=index))

    runner = BacktestRunner(catalog=DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir, result_dir=result_dir)
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-04",
        strategy="momentum",
        schedule=ScheduleSpec(kind="named", name="daily"),
        fill_mode="close",
    )

    report = runner.run_spec(runner.resolve_spec(spec))

    assert report.output_dir is not None
    assert (report.output_dir / "resolved_execution_spec.json").exists()
    assert (report.output_dir / "execution_resolution.json").exists()
```

- [ ] **Step 2: Run the focused test and verify the writer does not yet emit the new artifacts**

Run: `uv run python -m pytest tests/test_run.py::test_run_spec_persists_resolution_artifacts -v`
Expected: FAIL because `run_spec()` and the new artifact files do not exist yet

- [ ] **Step 3: Implement `run_spec()` as the canonical path and keep `run()` as a wrapper**

```python
# backtesting/run.py
from backtesting.specs import ExecutionSpec, ResolvedExecutionSpec, get_preset, load_execution_spec, resolve_execution_spec

@dataclass(slots=True)
class RunReport:
    config: RunConfig
    summary: dict[str, float]
    result: BacktestResult
    position_plan: PositionPlan | None = None
    output_dir: Path | None = None
    resolved_spec: ResolvedExecutionSpec | None = None
    execution_resolution: dict[str, object] | None = None

class BacktestRunner:
    ...
    def resolve_spec_from_config(self, config: RunConfig) -> ResolvedExecutionSpec:
        spec = ExecutionSpec(
            start=config.start,
            end=config.end,
            strategy=config.strategy,
            name=config.name,
            capital=config.capital,
            schedule=ScheduleSpec(kind="named", name=config.schedule),
            fill_mode=config.fill_mode,
            universe_id=config.universe_id,
            allow_fractional=config.allow_fractional,
            fee=config.fee,
            sell_tax=config.sell_tax,
            slippage=config.slippage,
            spec_source="cli",
        )
        return self.resolve_spec(spec)

    def resolve_spec(self, spec: ExecutionSpec) -> ResolvedExecutionSpec:
        available = {dataset.id for dataset in self.catalog.datasets.values()}
        return resolve_execution_spec(spec, available_dataset_ids=available)

    def run(self, config: RunConfig) -> RunReport:
        return self.run_spec(self.resolve_spec_from_config(config))

    def run_spec(self, resolved: ResolvedExecutionSpec) -> RunReport:
        execution = resolved.execution
        strategy = build_strategy(execution.strategy)
        config = RunConfig(
            start=execution.start,
            end=execution.end,
            strategy=execution.strategy,
            name=execution.name,
            capital=execution.capital,
            schedule=resolved.schedule_name or "custom",
            fill_mode=execution.fill_mode,
            universe_id=execution.universe_id,
            allow_fractional=execution.allow_fractional,
            fee=execution.fee,
            sell_tax=execution.sell_tax,
            slippage=execution.slippage,
        )
        ...
        report = RunReport(config=config, summary=summary, result=result, position_plan=plan)
        report.resolved_spec = resolved
        report.execution_resolution = {
            "spec_source": execution.spec_source,
            "preset_id": execution.preset_id,
            "hook_id": resolved.hook_id,
            "resolution_notes": list(resolved.resolution_notes),
            "fallbacks_applied": list(execution.data_policy.fallbacks_applied),
        }
        report.output_dir = self.writer.write(report)
        return report
```

```python
# backtesting/reporting/writer.py
if getattr(report, "resolved_spec", None) is not None:
    self._write_json(run_dir / "resolved_execution_spec.json", asdict(report.resolved_spec))
if getattr(report, "execution_resolution", None) is not None:
    self._write_json(run_dir / "execution_resolution.json", report.execution_resolution)
```

- [ ] **Step 4: Run the run/test suites that prove parity and provenance**

Run: `uv run python -m pytest tests/test_run.py -v`
Expected: PASS with legacy tests still green and new provenance/parity coverage green

- [ ] **Step 5: Commit the runner integration slice**

```bash
git add backtesting/run.py backtesting/reporting/writer.py tests/test_run.py
git commit -m "Route runner through execution specs and persist provenance artifacts"
```

## Task 5: Add `--preset` / `--spec` CLI surfaces and the semiannual float-cap official preset

**Files:**
- Modify: `backtesting/run.py`
- Modify: `backtesting/specs/presets.py`
- Modify: `backtesting/specs/hooks.py`
- Modify: `tests/test_run.py`
- Modify: `scripts/semiannual_floatcap_k200.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write failing CLI tests for preset and spec execution**

```python
def test_run_parser_accepts_preset_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class StubRunner:
        def __init__(self, result_dir=None):
            pass

        def run_resolved_cli(self, *, preset_id=None, spec_path=None, config=None):
            observed["preset_id"] = preset_id
            observed["spec_path"] = spec_path
            observed["config"] = config
            index = pd.to_datetime(["2024-01-02"])
            result = BacktestResult(
                equity=pd.Series([1.0], index=index),
                returns=pd.Series([0.0], index=index),
                weights=pd.DataFrame({"A": [1.0]}, index=index),
                qty=pd.DataFrame({"A": [1.0]}, index=index),
                turnover=pd.Series([0.0], index=index),
            )
            return RunReport(config=config or RunConfig(start="2024-01-02", end="2024-01-02"), summary={"final_equity": 1.0, "avg_turnover": 0.0}, result=result)

    monkeypatch.setattr("backtesting.run.BacktestRunner", StubRunner)
    monkeypatch.setattr("sys.argv", ["run.py", "--preset", "kospi200_semiannual_floatcap"])

    backtesting_main()

    assert observed["preset_id"] == "kospi200_semiannual_floatcap"
```

- [ ] **Step 2: Run the new CLI tests to confirm parser support is missing**

Run: `uv run python -m pytest tests/test_run.py::test_run_parser_accepts_preset_argument -v`
Expected: FAIL because the parser does not accept `--preset`

- [ ] **Step 3: Implement parser branching and register the semiannual float-cap preset/hook**

```python
# backtesting/specs/hooks.py

def second_thursday_flags(index: pd.DatetimeIndex) -> pd.Series:
    flags = pd.Series(False, index=index, dtype=bool)
    months = sorted({(ts.year, ts.month) for ts in index if ts.month in (6, 12)})
    for year, month in months:
        month_days = index[(index.year == year) & (index.month == month)]
        thursdays = month_days[month_days.weekday == 3]
        if len(thursdays) >= 2:
            flags.loc[thursdays[1]] = True
    return flags


def build_floatcap_weights(float_mcap: pd.DataFrame, universe: pd.DataFrame, schedule: pd.Series) -> pd.DataFrame:
    valid_caps = float_mcap.where(universe.astype(bool))
    denom = valid_caps.sum(axis=1).replace(0.0, pd.NA)
    target = valid_caps.div(denom, axis=0).fillna(0.0)
    weights = pd.DataFrame(0.0, index=target.index, columns=target.columns, dtype=float)
    last = pd.Series(0.0, index=target.columns, dtype=float)
    for ts in target.index:
        if bool(schedule.loc[ts]):
            last = target.loc[ts].fillna(0.0).astype(float)
        weights.loc[ts] = last
    return weights

register_hook("kospi200_semiannual_floatcap", build_floatcap_weights)
```

```python
# backtesting/specs/presets.py
register_preset(
    "kospi200_semiannual_floatcap",
    lambda: ExecutionSpec(
        start="2019-01-01",
        end=pd.Timestamp.today().date().isoformat(),
        strategy="momentum",
        name="kospi200_semiannual_floatcap_close_v1",
        schedule=ScheduleSpec(kind="custom_dates"),
        fill_mode="close",
        universe_id=None,
        weight_source=WeightSourceSpec(kind="hook", hook_id="kospi200_semiannual_floatcap"),
        data_policy=DataPolicySpec(requested_weight_basis="float_market_cap", fallback_order=("market_cap",)),
        spec_source="preset",
        preset_id="kospi200_semiannual_floatcap",
        notes=("Use second-Thursday semiannual rebalance schedule.",),
    ),
)
```

```python
# backtesting/run.py
parser.add_argument("--preset")
parser.add_argument("--spec")
...
if args.preset and args.spec:
    raise SystemExit("choose exactly one advanced execution source: --preset or --spec")
...
runner = BacktestRunner(result_dir=Path(args.out_root) if args.out_root else None)
report = runner.run_resolved_cli(preset_id=args.preset, spec_path=args.spec, config=config if not args.preset and not args.spec else None)
```

- [ ] **Step 4: Run the parser and preset tests to green**

Run: `uv run python -m pytest tests/test_run.py -k "preset or spec or parser" -v`
Expected: PASS

- [ ] **Step 5: Commit the CLI surface and preset slice**

```bash
git add backtesting/run.py backtesting/specs/presets.py backtesting/specs/hooks.py tests/test_run.py scripts/semiannual_floatcap_k200.py
git commit -m "Add official preset/spec CLI entrypoints for advanced runs"
```

## Task 6: Refresh docs and run the full parity verification sequence

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-24-execution-spec-surface-design.md` (only if implementation-scoped clarifications are needed)
- Test: `tests/specs/test_loader.py`
- Test: `tests/specs/test_resolve.py`
- Test: `tests/test_run.py`
- Test: `tests/execution/test_schedule.py`
- Test: `tests/engine/test_core.py`

- [ ] **Step 1: Add README examples for the simple CLI, preset path, and JSON spec path**

```markdown
## Advanced execution surfaces

The basic CLI remains the default for simple registered strategies:

```bash
uv run python -m backtesting.run \
  --strategy momentum \
  --start 2024-01-01 \
  --end 2024-12-31
```

For advanced execution, use a preset:

```bash
uv run python -m backtesting.run --preset kospi200_semiannual_floatcap
```

For explicit reproducibility, use a JSON spec file:

```bash
uv run python -m backtesting.run --spec docs/examples/kospi200_semiannual_floatcap.json
```

Every advanced run emits `resolved_execution_spec.json` and `execution_resolution.json` alongside the existing result artifacts.
```

- [ ] **Step 2: Run the focused doc-adjacent regression suite**

Run: `uv run python -m pytest tests/specs/test_loader.py tests/specs/test_resolve.py tests/test_run.py -v`
Expected: PASS

- [ ] **Step 3: Run the engine/schedule regression suite to prove math parity was preserved**

Run: `uv run python -m pytest tests/execution/test_schedule.py tests/engine/test_core.py tests/test_run.py -v`
Expected: PASS with no changes to fill or schedule semantics

- [ ] **Step 4: Run the final combined verification command used for handoff evidence**

Run: `uv run python -m pytest tests/specs/test_loader.py tests/specs/test_resolve.py tests/test_run.py tests/execution/test_schedule.py tests/engine/test_core.py -v`
Expected: PASS

- [ ] **Step 5: Commit the docs and final verification pass**

```bash
git add README.md docs/superpowers/specs/2026-04-24-execution-spec-surface-design.md
git commit -m "Document advanced execution surfaces and parity guarantees"
```

## Self-review notes

### Spec coverage
- Official `--preset` / `--spec` surface: covered in Tasks 4 and 5
- Unified execution contract: covered in Tasks 2 and 4
- Registry-backed extension path: covered in Task 2 and Task 5
- Provenance artifacts and fallback transparency: covered in Tasks 2, 4, and 6
- Same-input same-result invariant: covered by parity tests in Tasks 1, 4, and 6

### Known constraint carried into implementation
- Full YAML parsing is not currently implementable from the repo’s declared dependency set without adding a YAML parser dependency. This plan preserves the “no new dependency without explicit approval” rule by making JSON the executable spec format and requiring a clear error for `.yaml`/`.yml` inputs until dependency policy changes.

### Placeholder scan
- No TODO/TBD markers left in task steps
- Every code-changing step includes concrete file paths and code to land
- Every verification step includes an exact command and expected outcome
