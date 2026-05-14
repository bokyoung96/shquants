# Signed Target Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `target_weights.file` lane that runs externally supplied signed target weight matrices without portfolio-construction reinterpretation.

**Architecture:** Add `TargetWeightsSpec` to `ExecutionSpec`, parse it from JSON specs, and route it before composable/strategy paths. A new `backtesting/specs/target_weights.py` module owns CSV reading, signed validation, `PositionPlan` construction, and diagnostics.

**Tech Stack:** Python dataclasses, pandas, pytest, existing `BacktestRunner`/`BacktestCalculation`/`PositionPlan` APIs.

---

### Task 1: Model And Loader Contract

**Files:**
- Modify: `backtesting/specs/models.py`
- Modify: `backtesting/specs/loader.py`
- Modify: `backtesting/specs/__init__.py`
- Test: `tests/specs/test_loader.py`

- [ ] **Step 1: Write failing loader tests**

Add tests that load:

```python
{
    "start": "2024-01-01",
    "end": "2024-12-31",
    "target_weights": {
        "kind": "file",
        "path": "weights.csv",
        "missing_policy": "zero",
        "untradable_policy": "fail",
        "unshortable_policy": "fail"
    }
}
```

Expected dataclass:

```python
TargetWeightsSpec(
    kind="file",
    path="weights.csv",
    missing_policy="zero",
    untradable_policy="fail",
    unshortable_policy="fail",
)
```

Also reject invalid policy strings.

- [ ] **Step 2: Run failing tests**

Run: `uv run python -m pytest tests/specs/test_loader.py -q`

Expected: fail because `TargetWeightsSpec` and loader parsing do not exist.

- [ ] **Step 3: Add minimal model and parser**

Add:

```python
@dataclass(frozen=True, slots=True)
class TargetWeightsSpec:
    kind: str = "file"
    path: str | None = None
    hook_id: str | None = None
    missing_policy: str = "zero"
    untradable_policy: str = "fail"
    unshortable_policy: str = "fail"
```

Add `target_weights: TargetWeightsSpec | None = None` to `ExecutionSpec`.

Add `_read_target_weights()` in `loader.py` with allowed values:

```python
kind: {"file"}
missing_policy: {"zero"}
untradable_policy: {"fail"}
unshortable_policy: {"fail"}
```

Export `TargetWeightsSpec` from `backtesting/specs/__init__.py`.

- [ ] **Step 4: Verify model tests pass**

Run: `uv run python -m pytest tests/specs/test_loader.py -q`

Expected: pass.

### Task 2: Signed CSV Reader And Plan Builder

**Files:**
- Create: `backtesting/specs/target_weights.py`
- Test: `tests/specs/test_target_weights.py`

- [ ] **Step 1: Write failing unit tests**

Cover:

```python
def test_target_weights_file_preserves_signed_weights(tmp_path):
    path = tmp_path / "weights.csv"
    path.write_text(",A,B\n2024-01-02,0.5,-0.5\n", encoding="utf-8")
    spec = ExecutionSpec(
        start="2024-01-02",
        end="2024-01-02",
        target_weights=TargetWeightsSpec(kind="file", path=str(path)),
        shorting=ShortingSpec(enabled=True),
    )
    market = MarketData(frames={"close": pd.DataFrame({"A": [10.0], "B": [10.0]}, index=pd.to_datetime(["2024-01-02"]))}, universe=None, benchmark=None)
    plan, meta = build_position_plan_from_target_weights(spec, market)
    assert_frame_equal(plan.target_weights, pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=pd.to_datetime(["2024-01-02"])))
    assert meta["plan_source"] == "target_weights"
```

Also test duplicate dates, duplicate symbols, non-ISO dates, non-numeric cells, infinities, negative weights without shorting, unknown nonzero symbols, and unshortable short targets.

- [ ] **Step 2: Run failing tests**

Run: `uv run python -m pytest tests/specs/test_target_weights.py -q`

Expected: fail because module does not exist.

- [ ] **Step 3: Implement reader and builder**

Create functions:

```python
def build_position_plan_from_target_weights(spec: ExecutionSpec, market: MarketData) -> tuple[PositionPlan, dict[str, object]]:
    ...

def read_target_weights_csv(path: Path) -> pd.DataFrame:
    ...
```

Rules:

- Preserve signed numeric values.
- Reindex to `market.frames["close"]`.
- Reject any nonzero symbol not in close columns.
- Reject negative values when `spec.shorting.enabled` is false.
- If `spec.shorting.shortable_field` is set and a `shortable` frame exists, reject non-shortable short targets.
- Build a pass-through `PositionPlan` using `PositionPolicySpec(kind="pass_through")` and `build_position_plan_from_spec`.
- Return metadata with `plan_source`, average/max gross exposure, average/min/max net exposure.

- [ ] **Step 4: Verify unit tests pass**

Run: `uv run python -m pytest tests/specs/test_target_weights.py -q`

Expected: pass.

### Task 3: Calculation And Runner Integration

**Files:**
- Modify: `backtesting/calculation.py`
- Modify: `backtesting/run.py`
- Modify: `backtesting/specs/resolve.py`
- Test: `tests/run/test_runner_specs.py`

- [ ] **Step 1: Write failing integration tests**

Add runner tests that:

- create close and k200 parquet frames,
- write a signed weights CSV,
- run an `ExecutionSpec(target_weights=..., shorting=ShortingSpec(enabled=True))`,
- assert `report.result.weights` keeps negative weights,
- assert `report.result.qty` contains negative quantity,
- assert `report.execution_resolution["plan_source"] == "target_weights"`,
- assert `signal_dates` turnover is based on target weight changes.

- [ ] **Step 2: Run failing integration tests**

Run: `uv run python -m pytest tests/run/test_runner_specs.py -q`

Expected: fail because calculation does not route `target_weights`.

- [ ] **Step 3: Wire plan priority**

In `BacktestCalculationAdapters`, add:

```python
build_target_weight_plan: Callable[[ExecutionSpec, MarketData], tuple[PositionPlan, dict[str, object]]]
```

In `_build_plan()`, route before hook/composable/strategy:

```python
if spec.target_weights is not None:
    plan, metadata = self.adapters.build_target_weight_plan(spec, market)
    return plan, None, None, metadata, resolved_spec
```

In `BacktestRunner.run_spec()`, pass `build_position_plan_from_target_weights`.

In `resolve_execution_spec()`, do not build a registered strategy when `spec.target_weights is not None`.

- [ ] **Step 4: Verify integration tests pass**

Run: `uv run python -m pytest tests/run/test_runner_specs.py -q`

Expected: pass.

### Task 4: Regression And Full Verification

**Files:**
- Verify only unless targeted fixes are needed.

- [ ] **Step 1: Run focused regression suite**

Run:

```powershell
uv run python -m pytest tests/specs tests/run tests/engine tests/weighting tests/construction -q
```

Expected: pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
uv run python -m pytest -q
```

Expected: pass.

- [ ] **Step 3: Commit implementation**

Use Lore commit format. Include tested commands and note that `weighting.explicit` behavior remains unchanged.

---

## Self-Review

- Spec coverage: the plan covers model parsing, signed file input, fail-fast validation, runner priority, diagnostics, and regressions.
- Placeholder scan: no `TODO`, `TBD`, or unspecified implementation steps remain.
- Type consistency: `TargetWeightsSpec`, `target_weights`, and `build_position_plan_from_target_weights` names are consistent across tasks.
