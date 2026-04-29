# Spec-Driven Selection, Weighting, and Staged Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a spec-driven layer that maps investment intent into selection, weighting, and staged position policies while preserving the existing `BacktestEngine`.

**Architecture:** Keep `BacktestEngine` as the target-weight execution core. Extend `ExecutionSpec` with `SelectionSpec`, `WeightingSpec`, and `PositionPolicySpec`, then add feature, selection, weighting, and position-policy builders that produce a valid `PositionPlan` before the existing engine runs. Preserve legacy `strategy="momentum"` / `top_n` behavior as the fallback path.

**Tech Stack:** Python 3.12, dataclasses, pandas, pytest, existing `backtesting` package patterns, no new dependencies.

---

## File structure

### Modify existing files

- `backtesting/specs/models.py` — add dataclasses for `ConditionSpec`, `SelectionSpec`, `WeightingSpec`, `PositionPolicySpec`, bucket/rule specs, and helper methods for detecting the spec-driven path.
- `backtesting/specs/loader.py` — parse nested JSON objects for selection, weighting, and position policy while preserving legacy specs.
- `backtesting/specs/resolve.py` — add dataset and warmup resolution for feature fields used by selection and weighting.
- `backtesting/specs/__init__.py` — export new spec dataclasses.
- `backtesting/run.py` — route specs with `selection` / `weighting` / `position_policy` through the new plan builder and keep the existing strategy/hook branches intact.
- `backtesting/policy/__init__.py` — export existing staged policy classes and the new policy builder.
- `backtesting/__init__.py` — export new public spec and builder symbols.

### Create new files

- `backtesting/features/__init__.py` — public feature registry exports.
- `backtesting/features/registry.py` — registered feature definitions, dataset dependencies, warmup needs, and feature-frame construction.
- `backtesting/selection/__init__.py` — public selection builder exports.
- `backtesting/selection/builders.py` — condition evaluation plus `filter`, `rank_top_n`, `score_threshold`, `event`, `explicit`, and `hook` selection builders.
- `backtesting/weighting/__init__.py` — public weighting builder exports.
- `backtesting/weighting/builders.py` — `equal_weight`, `market_cap`, `float_market_cap`, `score`, `inverse_vol`, `explicit`, and `hook` weighting builders.
- `backtesting/policy/builder.py` — convert `PositionPolicySpec` into `PassThroughPolicy`, `BudgetPreservingStagedPolicy`, or a policy hook.
- `backtesting/specs/plan_builder.py` — orchestrate features, selection, weighting, and position policy into a `PositionPlan`.

### Create/modify tests

- `tests/specs/test_loader.py` — parsing tests for the new nested specs.
- `tests/specs/test_resolve.py` — dataset and warmup resolution tests.
- `tests/features/test_registry.py` — feature dependency and frame-building tests.
- `tests/selection/test_builders.py` — condition, filter, rank, threshold, event, explicit, and hook selection tests.
- `tests/weighting/test_builders.py` — equal, cap-weighted, score, inverse-vol, explicit, and hook weighting tests.
- `tests/policy/test_builder.py` — pass-through and staged policy-builder tests.
- `tests/specs/test_plan_builder.py` — end-to-end plan-building tests without the engine.
- `tests/test_run.py` — runner integration and legacy parity tests.

---

### Task 1: Expand execution spec models

**Files:**
- Modify: `backtesting/specs/models.py`
- Modify: `backtesting/specs/__init__.py`
- Test: `tests/specs/test_loader.py`

- [ ] **Step 1: Write the failing spec-model parsing tests**

Append these tests to `tests/specs/test_loader.py`:

```python
import json
from pathlib import Path

from backtesting.specs import load_execution_spec


def test_load_execution_spec_parses_selection_weighting_and_staged_policy(tmp_path: Path) -> None:
    path = tmp_path / "condition_strategy.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-02",
                "end": "2024-01-05",
                "selection": {
                    "kind": "filter",
                    "conditions": [
                        {"field": "momentum_60d", "op": ">", "value": 0},
                        {"field": "market_cap", "op": ">=", "value": 100_000_000_000},
                    ],
                },
                "weighting": {"kind": "equal_weight"},
                "position_policy": {
                    "kind": "staged",
                    "buckets": [
                        {"id": "entry", "fraction": 0.5},
                        {"id": "add_1", "fraction": 0.5},
                    ],
                    "rules": {
                        "entry": {"kind": "selection_passes"},
                        "adds": [{"kind": "still_passes_after_rebalances", "count": 1}],
                        "exit": {"kind": "selection_fails"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.selection is not None
    assert spec.selection.kind == "filter"
    assert [condition.field for condition in spec.selection.conditions] == ["momentum_60d", "market_cap"]
    assert spec.weighting is not None
    assert spec.weighting.kind == "equal_weight"
    assert spec.position_policy is not None
    assert spec.position_policy.kind == "staged"
    assert [bucket.id for bucket in spec.position_policy.buckets] == ["entry", "add_1"]
    assert spec.uses_composable_plan is True


def test_load_execution_spec_keeps_legacy_strategy_specs_on_legacy_path(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-02",
                "end": "2024-01-05",
                "strategy": "momentum",
                "top_n": 3,
                "lookback": 1,
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.selection is None
    assert spec.weighting is None
    assert spec.position_policy is None
    assert spec.uses_composable_plan is False
```

- [ ] **Step 2: Run tests and verify they fail for missing fields**

Run:

```powershell
uv run pytest tests/specs/test_loader.py::test_load_execution_spec_parses_selection_weighting_and_staged_policy tests/specs/test_loader.py::test_load_execution_spec_keeps_legacy_strategy_specs_on_legacy_path -q
```

Expected: FAIL with an `AttributeError` or import failure for `selection`, `weighting`, `position_policy`, or `uses_composable_plan`.

- [ ] **Step 3: Implement spec dataclasses**

Replace `backtesting/specs/models.py` with the existing classes plus these new dataclasses and fields:

```python
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
    params: dict[str, object] = field(default_factory=dict)
    hold_days: int = 0


@dataclass(frozen=True, slots=True)
class WeightingSpec:
    kind: str = "equal_weight"
    field: str | None = None
    path: str | None = None
    hook_id: str | None = None
    params: dict[str, object] = field(default_factory=dict)


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
    entry: PositionRuleSpec = field(default_factory=lambda: PositionRuleSpec(kind="selection_passes"))
    adds: tuple[PositionRuleSpec, ...] = ()
    exit: PositionRuleSpec = field(default_factory=lambda: PositionRuleSpec(kind="selection_fails"))
    hook_id: str | None = None
    params: dict[str, object] = field(default_factory=dict)
```

Add these fields to `ExecutionSpec`:

```python
selection: SelectionSpec | None = None
weighting: WeightingSpec | None = None
position_policy: PositionPolicySpec | None = None
```

Add this property to `ExecutionSpec`:

```python
@property
def uses_composable_plan(self) -> bool:
    return self.selection is not None or self.weighting is not None or self.position_policy is not None
```

- [ ] **Step 4: Export the dataclasses**

Update `backtesting/specs/__init__.py` to import and include:

```python
ConditionSpec
SelectionSpec
WeightingSpec
PositionBucketSpec
PositionRuleSpec
PositionPolicySpec
```

Keep all existing exports.

- [ ] **Step 5: Continue directly to Task 2 before committing**

Task 1 tests need loader support from Task 2. Do not commit after only the dataclass changes.

### Task 2: Parse nested selection, weighting, and position-policy specs

**Files:**
- Modify: `backtesting/specs/loader.py`
- Test: `tests/specs/test_loader.py`

- [ ] **Step 1: Add loader tests for defaults and invalid shapes**

Append to `tests/specs/test_loader.py`:

```python
import pytest


def test_load_execution_spec_defaults_weighting_when_selection_exists(tmp_path: Path) -> None:
    path = tmp_path / "filter_only.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-02",
                "end": "2024-01-05",
                "selection": {"kind": "rank_top_n", "field": "momentum_20d", "n": 2},
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.selection is not None
    assert spec.weighting is not None
    assert spec.weighting.kind == "equal_weight"
    assert spec.position_policy is not None
    assert spec.position_policy.kind == "pass_through"


def test_load_execution_spec_rejects_non_object_position_policy(tmp_path: Path) -> None:
    path = tmp_path / "bad_policy.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-02",
                "end": "2024-01-05",
                "selection": {"kind": "filter", "conditions": []},
                "position_policy": "staged",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="position_policy must be an object"):
        load_execution_spec(path)
```

- [ ] **Step 2: Run loader tests and verify failure**

Run:

```powershell
uv run pytest tests/specs/test_loader.py -q
```

Expected: FAIL until parser helpers are added.

- [ ] **Step 3: Implement parser helpers**

In `backtesting/specs/loader.py`, import the new dataclasses and add helper functions for:

```python
_read_object(payload, key)
_read_conditions(raw_conditions)
_read_selection(payload)
_read_weighting(payload, selection)
_read_position_rule(raw, default_kind)
_read_position_policy(payload, selection)
```

Required behavior:

- `selection` must be an object when present.
- `weighting` defaults to `WeightingSpec(kind="equal_weight")` when selection exists and weighting is omitted.
- `position_policy` defaults to `PositionPolicySpec(kind="pass_through")` when selection exists and position policy is omitted.
- `position_policy.rules.entry`, `position_policy.rules.adds`, and `position_policy.rules.exit` parse into `PositionRuleSpec`.
- Invalid non-object shapes raise `ValueError` with the key name in the message.

In `load_execution_spec`, create:

```python
selection = _read_selection(payload)
weighting = _read_weighting(payload, selection)
position_policy = _read_position_policy(payload, selection)
```

Pass these to `ExecutionSpec`.

- [ ] **Step 4: Run loader tests**

Run:

```powershell
uv run pytest tests/specs/test_loader.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1 and Task 2 together**

```powershell
git add backtesting/specs/models.py backtesting/specs/__init__.py backtesting/specs/loader.py tests/specs/test_loader.py
git commit -m "Represent investment intent explicitly in execution specs" -m "Selection, weighting, and position policy are now part of the execution contract so user intent does not collapse into top-n momentum defaults." -m "Constraint: Preserve legacy strategy/top_n specs" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/specs/test_loader.py -q"
```

### Task 3: Add feature registry and feature-frame construction

**Files:**
- Create: `backtesting/features/__init__.py`
- Create: `backtesting/features/registry.py`
- Test: `tests/features/test_registry.py`

- [ ] **Step 1: Write feature registry tests**

Create `tests/features/test_registry.py` with tests that verify:

```python
from backtesting.catalog import DatasetId
from backtesting.features import build_features, feature_dataset_ids, feature_warmup_days


def test_feature_dataset_ids_include_base_inputs() -> None:
    ids = feature_dataset_ids(["momentum_60d", "market_cap", "avg_trading_value_20d"])
    assert DatasetId.QW_ADJ_C in ids
    assert DatasetId.QW_MKTCAP in ids
    assert DatasetId.QW_V in ids


def test_feature_warmup_days_uses_largest_lookback() -> None:
    assert feature_warmup_days(["momentum_60d", "momentum_20d"]) == 60
```

Also include a `MarketData` fixture with `close`, `volume`, `market_cap`, `float_market_cap`, `foreign_ratio`, `inst_flow`, and `retail_flow`, then assert:

```python
features = build_features(market, ["momentum_20d", "market_cap", "avg_trading_value_20d"])
assert features["market_cap"].equals(market.frames["market_cap"])
assert features["momentum_20d"].shape == market.frames["close"].shape
assert features["avg_trading_value_20d"].shape == market.frames["close"].shape
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/features/test_registry.py -q
```

Expected: FAIL because `backtesting.features` does not exist.

- [ ] **Step 3: Implement feature registry**

Create `backtesting/features/registry.py` with:

- `FeatureDefinition(field, dataset_ids, warmup_days, build)`
- `register_feature`
- `get_feature`
- `feature_dataset_ids`
- `feature_warmup_days`
- `build_features`

Register these fields:

```text
close -> QW_ADJ_C
open -> QW_ADJ_O
momentum_20d -> QW_ADJ_C, warmup 20
momentum_60d -> QW_ADJ_C, warmup 60
market_cap -> QW_MKTCAP
float_market_cap -> QW_MKTCAP_FLT
avg_trading_value_20d -> QW_ADJ_C and QW_V, warmup 20
foreign_ratio -> QW_FOREIGN_RATIO
institution_flow_20d -> QW_INSTITUTION, warmup 20
retail_flow_20d -> QW_RETAIL, warmup 20
```

Create `backtesting/features/__init__.py` exporting all public feature functions.

- [ ] **Step 4: Run feature tests**

Run:

```powershell
uv run pytest tests/features/test_registry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backtesting/features tests/features/test_registry.py
git commit -m "Add registered features for spec-driven plans" -m "Feature definitions provide dataset dependencies, warmup needs, and feature-frame construction for selection and weighting specs." -m "Constraint: Avoid arbitrary formula execution in first implementation" -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: uv run pytest tests/features/test_registry.py -q"
```

### Task 4: Add selection builders

**Files:**
- Create: `backtesting/selection/__init__.py`
- Create: `backtesting/selection/builders.py`
- Test: `tests/selection/test_builders.py`

- [ ] **Step 1: Write selection tests**

Create `tests/selection/test_builders.py` covering:

- `filter` keeps every passing name and does not cap to top N.
- `rank_top_n` selects exactly configured `n` names per date.
- `score_threshold` selects scores greater than or equal to threshold.
- `event` extends event flags by `hold_days`.
- `hook` delegates to a registered selection hook.
- invalid condition operator raises `ValueError`.

Use feature frames with columns `A`, `B`, and `C` and dates `2024-01-02` through `2024-01-04`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/selection/test_builders.py -q
```

Expected: FAIL because `backtesting.selection` does not exist.

- [ ] **Step 3: Implement selection builders**

Create `backtesting/selection/builders.py` with:

- `SelectionHook`
- `register_selection_hook`
- `selection_fields`
- `build_selection`
- condition evaluator supporting `>`, `>=`, `<`, `<=`, `==`, `!=`, `notna`, and `isna`
- builders for `filter`, `rank_top_n`, `score_threshold`, `event`, `explicit`, and `hook`

Important implementation rules:

- `filter` starts from all `True` and ANDs every condition.
- `rank_top_n` requires a positive `n` and does not run for filter specs.
- `explicit` reads a CSV with dates as index and symbols as columns.
- `hook` requires a registered `hook_id`.

Create `backtesting/selection/__init__.py` exporting `SelectionHook`, `build_selection`, `register_selection_hook`, and `selection_fields`.

- [ ] **Step 4: Run selection tests**

Run:

```powershell
uv run pytest tests/selection/test_builders.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backtesting/selection tests/selection/test_builders.py
git commit -m "Select securities from explicit spec primitives" -m "Selection builders support filter, rank, threshold, event, explicit, and hook inputs so top_n is only one selection mode." -m "Constraint: Filter selection must include all passing names" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/selection/test_builders.py -q"
```

### Task 5: Add weighting builders

**Files:**
- Create: `backtesting/weighting/__init__.py`
- Create: `backtesting/weighting/builders.py`
- Test: `tests/weighting/test_builders.py`

- [ ] **Step 1: Write weighting tests**

Create `tests/weighting/test_builders.py` covering:

- `equal_weight` normalizes across selected names.
- `market_cap` normalizes market cap only within selected names.
- `float_market_cap` uses `float_market_cap`.
- `score` uses positive values from configured score field.
- `inverse_vol` returns finite weights when close prices are available.
- `explicit` reads target weights from CSV.
- `hook` delegates to a registered weighting hook.
- `weighting_fields` returns required feature fields.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/weighting/test_builders.py -q
```

Expected: FAIL because `backtesting.weighting` does not exist.

- [ ] **Step 3: Implement weighting builders**

Create `backtesting/weighting/builders.py` with:

- `WeightingHook`
- `register_weighting_hook`
- `weighting_fields`
- `build_weights`
- `_normalize`
- `_weighted_by_feature`

Required behavior:

- `equal_weight`: selected names receive `1 / count`.
- `market_cap`: selected names are weighted by `features["market_cap"]`.
- `float_market_cap`: selected names are weighted by `features["float_market_cap"]`.
- `score`: selected names are weighted by non-negative configured score values.
- `inverse_vol`: selected names are weighted by inverse 20-day realized volatility from `close`.
- `explicit`: reads CSV target weights and aligns to selection index/columns.
- `hook`: calls a registered weighting hook and aligns output to selection.

Create `backtesting/weighting/__init__.py` exporting `WeightingHook`, `build_weights`, `register_weighting_hook`, and `weighting_fields`.

- [ ] **Step 4: Run weighting tests**

Run:

```powershell
uv run pytest tests/weighting/test_builders.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backtesting/weighting tests/weighting/test_builders.py
git commit -m "Weight selected securities from explicit spec primitives" -m "Weighting builders normalize selected names through equal, cap, score, inverse-vol, explicit, and hook methods." -m "Constraint: Weighting remains separate from selection" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/weighting/test_builders.py -q"
```

### Task 6: Add position-policy builder for pass-through and staged trading

**Files:**
- Create: `backtesting/policy/builder.py`
- Modify: `backtesting/policy/__init__.py`
- Test: `tests/policy/test_builder.py`

- [ ] **Step 1: Write policy-builder tests**

Create `tests/policy/test_builder.py` covering:

- `PositionPolicySpec(kind="pass_through")` preserves base target weights.
- `PositionPolicySpec(kind="staged")` activates the first bucket on selection pass.
- staged add bucket activates when `still_passes_after_rebalances` is true for the configured count.
- selection failure clears staged exposure.
- unsupported position rule kind raises `ValueError`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/policy/test_builder.py -q
```

Expected: FAIL because `build_position_plan_from_spec` does not exist.

- [ ] **Step 3: Implement policy builder**

Create `backtesting/policy/builder.py` with:

- `build_position_plan_from_spec(spec, base_target_weights, selection_mask, market)`
- `_build_staged`
- `_rule_key`
- `_rule_context`

Rules:

- `pass_through` uses existing `PassThroughPolicy`.
- `staged` uses existing `BudgetPreservingStagedPolicy`, `BucketDefinition`, and `StagedRuleSet`.
- `selection_passes` maps to the current selection mask.
- `selection_fails` maps to inverse selection mask.
- `still_passes_after_rebalances` maps to `selection_mask & selection_mask.shift(count)`.
- unsupported `hook` position policy raises a clear `ValueError` in this builder; full-plan hooks already exist through `weight_source.kind == "hook"`.

Modify `backtesting/policy/__init__.py` to export:

```python
BucketDefinition
BudgetPreservingStagedPolicy
PassThroughPolicy
PositionPlan
PositionPolicy
StagedRuleSet
build_position_plan_from_spec
```

- [ ] **Step 4: Run policy tests**

Run:

```powershell
uv run pytest tests/policy/test_builder.py tests/policy/test_staged.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backtesting/policy tests/policy/test_builder.py
git commit -m "Expose staged trading as a spec position policy" -m "PositionPolicySpec now maps to pass-through or existing budget-preserving staged policy behavior, allowing split buying after selection and weighting." -m "Constraint: Reuse BudgetPreservingStagedPolicy instead of reimplementing staged state" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/policy/test_builder.py tests/policy/test_staged.py -q"
```

### Task 7: Resolve datasets and warmup for spec-driven plans

**Files:**
- Modify: `backtesting/specs/resolve.py`
- Test: `tests/specs/test_resolve.py`

- [ ] **Step 1: Write resolution tests**

Append tests to `tests/specs/test_resolve.py` proving:

- filter conditions on `momentum_60d` and `market_cap` add `QW_ADJ_C` and `QW_MKTCAP`.
- `WeightingSpec(kind="float_market_cap")` adds `QW_MKTCAP_FLT`.
- resolved `warmup_days` is at least 60 for `momentum_60d`.
- unknown feature fields raise `KeyError` with `unknown feature field`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/specs/test_resolve.py -q
```

Expected: FAIL until resolver includes new feature fields.

- [ ] **Step 3: Implement field collection and dataset resolution**

In `backtesting/specs/resolve.py`, import:

```python
from backtesting.features import feature_dataset_ids, feature_warmup_days
from backtesting.selection import selection_fields
from backtesting.weighting import weighting_fields
```

Add:

```python
def _spec_feature_fields(spec: ExecutionSpec) -> tuple[str, ...]:
    fields: list[str] = []
    if spec.selection is not None:
        fields.extend(selection_fields(spec.selection))
    if spec.weighting is not None:
        fields.extend(weighting_fields(spec.weighting))
    return tuple(dict.fromkeys(fields))
```

Inside `resolve_execution_spec`, collect feature fields before legacy strategy dataset resolution:

```python
feature_fields = _spec_feature_fields(spec)
if feature_fields:
    dataset_ids.extend(_resolve_universe_dataset(universe_spec, dataset_id) for dataset_id in feature_dataset_ids(feature_fields))
    required_warmup = feature_warmup_days(feature_fields)
    if required_warmup > spec.warmup_days:
        spec = replace(spec, warmup_days=required_warmup)
        notes.append(f"warmup_days increased to {required_warmup} for feature lookbacks")
```

Change legacy strategy dataset resolution to run only when `not spec.uses_composable_plan`.

- [ ] **Step 4: Run resolver tests**

Run:

```powershell
uv run pytest tests/specs/test_resolve.py tests/features/test_registry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backtesting/specs/resolve.py tests/specs/test_resolve.py
git commit -m "Resolve data needs from spec-driven features" -m "Execution resolution now derives datasets and warmup from selection and weighting fields before the runner loads market data." -m "Constraint: Legacy strategy resolution remains unchanged for specs without selection or weighting" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/specs/test_resolve.py tests/features/test_registry.py -q"
```

### Task 8: Build a PositionPlan from selection, weighting, and position policy

**Files:**
- Create: `backtesting/specs/plan_builder.py`
- Modify: `backtesting/specs/__init__.py`
- Test: `tests/specs/test_plan_builder.py`

- [ ] **Step 1: Write plan-builder tests**

Create `tests/specs/test_plan_builder.py` covering:

- filter selection plus equal weighting produces equal weights for all passing names.
- market universe masks out names before weighting.
- staged position policy is applied after base weights are built.
- missing `selection` raises `ValueError`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/specs/test_plan_builder.py -q
```

Expected: FAIL because `plan_builder` does not exist.

- [ ] **Step 3: Implement plan builder**

Create `backtesting/specs/plan_builder.py` with:

```python
def build_position_plan_from_execution_spec(spec: ExecutionSpec, market: MarketData) -> PositionPlan:
    if spec.selection is None:
        raise ValueError("spec-driven plan requires selection")
    weighting = spec.weighting or WeightingSpec(kind="equal_weight")
    position_policy = spec.position_policy or PositionPolicySpec(kind="pass_through")
    fields = tuple(dict.fromkeys((*selection_fields(spec.selection), *weighting_fields(weighting))))
    features = build_features(market, fields)
    selection = build_selection(spec.selection, features)
    if market.universe is not None:
        universe = market.universe.reindex(index=selection.index, columns=selection.columns).fillna(False).astype(bool)
        selection = selection & universe
    base_weights = build_weights(weighting, selection, features)
    return build_position_plan_from_spec(
        position_policy,
        base_target_weights=base_weights,
        selection_mask=selection,
        market=market,
    )
```

Update `backtesting/specs/__init__.py` to export `build_position_plan_from_execution_spec`.

- [ ] **Step 4: Run plan-builder tests**

Run:

```powershell
uv run pytest tests/specs/test_plan_builder.py tests/selection/test_builders.py tests/weighting/test_builders.py tests/policy/test_builder.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backtesting/specs/plan_builder.py backtesting/specs/__init__.py tests/specs/test_plan_builder.py
git commit -m "Compose spec-driven plans before engine execution" -m "Plan building now joins registered features, selection, weighting, and position policy into the existing PositionPlan contract." -m "Constraint: BacktestEngine continues to consume only target weights" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/specs/test_plan_builder.py tests/selection/test_builders.py tests/weighting/test_builders.py tests/policy/test_builder.py -q"
```

### Task 9: Route spec-driven plans through BacktestRunner

**Files:**
- Modify: `backtesting/run.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write runner integration tests**

Append tests to `tests/test_run.py` proving:

- a spec with `selection.kind="filter"`, `weighting.kind="equal_weight"`, and `top_n=1` still buys all passing names, not one name.
- a spec with `position_policy.kind="staged"` writes a bucket ledger and produces staged target weights.
- `test_runner_run_and_run_spec_produce_identical_results_for_legacy_inputs` still passes.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/test_run.py::test_runner_executes_filter_equal_weight_spec_without_using_top_n tests/test_run.py::test_runner_executes_staged_spec_with_bucket_ledger -q
```

Expected: FAIL until `BacktestRunner.run_spec` uses the new plan builder.

- [ ] **Step 3: Update runner branch**

In `backtesting/run.py`, import `build_position_plan_from_execution_spec`.

Inside `run_spec`, insert this branch between the existing full-plan hook branch and the legacy strategy branch:

```python
elif spec.uses_composable_plan:
    plan = build_position_plan_from_execution_spec(spec, market)
    schedule_input = self._schedule_from_spec(resolved_spec)
    extra_tradable = None
    resolution_meta = {"plan_source": "selection_weighting_position_policy"}
```

Leave the existing `weight_source.kind == "hook"` branch and the legacy strategy branch otherwise intact.

- [ ] **Step 4: Run runner tests**

Run:

```powershell
uv run pytest tests/test_run.py::test_runner_executes_filter_equal_weight_spec_without_using_top_n tests/test_run.py::test_runner_executes_staged_spec_with_bucket_ledger tests/test_run.py::test_runner_run_and_run_spec_produce_identical_results_for_legacy_inputs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backtesting/run.py tests/test_run.py
git commit -m "Route spec-driven plans through the existing runner" -m "BacktestRunner now builds PositionPlan objects from selection, weighting, and position policy specs while preserving legacy strategy execution." -m "Constraint: Existing engine and legacy momentum parity must remain unchanged" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/test_run.py::test_runner_executes_filter_equal_weight_spec_without_using_top_n tests/test_run.py::test_runner_executes_staged_spec_with_bucket_ledger tests/test_run.py::test_runner_run_and_run_spec_produce_identical_results_for_legacy_inputs -q"
```

### Task 10: Add example specs and public exports

**Files:**
- Modify: `backtesting/__init__.py`
- Create: `docs/superpowers/specs/examples/filter-equal-weight.json`
- Create: `docs/superpowers/specs/examples/filter-staged.json`
- Test: `tests/specs/test_loader.py`

- [ ] **Step 1: Add example spec validation test**

Append to `tests/specs/test_loader.py`:

```python
def test_example_specs_parse() -> None:
    examples = [
        Path("docs/superpowers/specs/examples/filter-equal-weight.json"),
        Path("docs/superpowers/specs/examples/filter-staged.json"),
    ]
    for path in examples:
        spec = load_execution_spec(path)
        assert spec.uses_composable_plan
```

- [ ] **Step 2: Create example spec files**

Create `docs/superpowers/specs/examples/filter-equal-weight.json` with filter selection on `momentum_60d`, `market_cap`, and `avg_trading_value_20d`, `equal_weight`, and `pass_through`.

Create `docs/superpowers/specs/examples/filter-staged.json` with filter selection on `momentum_60d` and `market_cap`, `equal_weight`, and a three-bucket staged policy using fractions `0.34`, `0.33`, and `0.33`.

- [ ] **Step 3: Export public symbols**

Update `backtesting/__init__.py` to import and include in `__all__`:

```python
ConditionSpec
SelectionSpec
WeightingSpec
PositionBucketSpec
PositionRuleSpec
PositionPolicySpec
build_position_plan_from_execution_spec
```

- [ ] **Step 4: Run example and export tests**

Run:

```powershell
uv run pytest tests/specs/test_loader.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add backtesting/__init__.py docs/superpowers/specs/examples tests/specs/test_loader.py
git commit -m "Document spec-driven strategy examples" -m "Example specs show filter, equal-weight, and staged-trading usage while public exports make the new primitives discoverable." -m "Constraint: Examples must parse through the same loader used by production runs" -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: uv run pytest tests/specs/test_loader.py -q"
```

### Task 11: Final regression and cleanup pass

**Files:**
- Review all files changed in previous tasks.
- Modify only if verification exposes a concrete issue.

- [ ] **Step 1: Run focused spec-driven suite**

Run:

```powershell
uv run pytest tests/features tests/selection tests/weighting tests/policy tests/specs -q
```

Expected: PASS.

- [ ] **Step 2: Run legacy backtesting suite**

Run:

```powershell
uv run pytest tests/test_run.py tests/strategy tests/strategies tests/construction tests/execution tests/analytics -q
```

Expected: PASS.

- [ ] **Step 3: Run static/lint checks available in project**

Run:

```powershell
uv run ruff check backtesting tests
```

Expected: PASS. If `ruff` is unavailable in the project environment, capture the command output in the final report and continue with pytest evidence.

- [ ] **Step 4: Inspect git diff for accidental engine changes**

Run:

```powershell
git diff -- backtesting/engine/core.py
```

Expected: no output. If output exists, revert only unintended engine changes and rerun the tests above.

- [ ] **Step 5: Commit verification fixes if any were required**

If Step 1-4 required code changes, commit them with:

```powershell
git add backtesting tests docs/superpowers/specs/examples
git commit -m "Stabilize spec-driven backtesting integration" -m "Verification exposed integration issues after the selection/weighting/position-policy slices, so this commit keeps the suite coherent without changing the engine contract." -m "Constraint: BacktestEngine remains unchanged" -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: uv run pytest tests/features tests/selection tests/weighting tests/policy tests/specs -q; uv run pytest tests/test_run.py tests/strategy tests/strategies tests/construction tests/execution tests/analytics -q"
```

If no code changes were required, do not create an empty commit.

---

## Self-review checklist

- Spec coverage:
  - Selection kinds: `filter`, `rank_top_n`, `score_threshold`, `event`, `explicit`, `hook` are covered in Task 4.
  - Weighting kinds: `equal_weight`, `market_cap`, `float_market_cap`, `score`, `inverse_vol`, `explicit`, `hook` are covered in Task 5.
  - Staged / split trading is covered in Task 6 and integrated in Task 9.
  - Feature registry and dataset resolution are covered in Tasks 3 and 7.
  - Runner integration while preserving engine behavior is covered in Task 9 and Task 11.
  - Legacy compatibility is covered in Tasks 1, 7, 9, and 11.
- Type consistency:
  - `SelectionSpec`, `WeightingSpec`, and `PositionPolicySpec` are introduced in Task 1 and used consistently in later tasks.
  - `build_position_plan_from_execution_spec` is created in Task 8 and imported by `run.py` in Task 9.
  - `build_position_plan_from_spec` is created in Task 6 and used by Task 8.
- Verification commands:
  - Each task includes a targeted pytest command.
  - Final task includes focused suite, legacy suite, ruff, and engine-diff verification.
