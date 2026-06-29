# Flow-Filtered Breakout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the flow-filtered 52-week breakout experiment by removing OP filters from the breakout grid and adding simple `> 0` foreign/institution flow gates.

**Architecture:** Keep the existing `tech_gamma_*` research pipeline. Extend `scripts/tech_gamma_research_filters.py` so `factor_filter` supports flow-only gates, then constrain `scripts/tech_gamma_breakout_grid_specs.py` to the approved flow-only grid. Preserve existing OP filter support outside this grid to avoid breaking older scripts/tests.

**Tech Stack:** Python, pandas, pytest, pyarrow, existing `TechGammaConfig` dataclass and `BreakoutStrategySpec` grid builder.

---

### Task 1: Add Flow-Only Filter Behavior

**Files:**
- Modify: `scripts/tech_gamma_research_filters.py`
- Test: `tests/scripts/test_tech_gamma_relative_research.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/scripts/test_tech_gamma_relative_research.py`:

```python
def test_flow_filters_use_prior_trailing_flow_to_cap() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["A005930", "A005930", "A005930"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    data = ResearchFeatureData(
        market_cap=pd.DataFrame({"A005930": [100.0, 100.0, 100.0]}, index=dates),
        foreign_flow=pd.DataFrame({"A005930": [-2.0, 5.0, 7.0]}, index=dates),
        institution_flow=pd.DataFrame({"A005930": [-1.0, -1.0, 3.0]}, index=dates),
    )

    foreign = apply_research_features(
        frame,
        TechGammaConfig(factor_filter="foreign_positive", factor_lookback_days=1),
        data,
    ).set_index("date")
    either = apply_research_features(
        frame,
        TechGammaConfig(factor_filter="foreign_or_institution_positive", factor_lookback_days=1),
        data,
    ).set_index("date")
    both = apply_research_features(
        frame,
        TechGammaConfig(factor_filter="foreign_and_institution_positive", factor_lookback_days=1),
        data,
    ).set_index("date")

    assert not bool(foreign.loc[pd.Timestamp("2024-01-03"), "factor_filter_ok"])
    assert bool(foreign.loc[pd.Timestamp("2024-01-04"), "factor_filter_ok"])
    assert bool(either.loc[pd.Timestamp("2024-01-04"), "factor_filter_ok"])
    assert not bool(both.loc[pd.Timestamp("2024-01-04"), "factor_filter_ok"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/scripts/test_tech_gamma_relative_research.py::test_flow_filters_use_prior_trailing_flow_to_cap -q
```

Expected: FAIL with `KeyError: "unknown factor filter 'foreign_positive'"`.

- [ ] **Step 3: Write minimal implementation**

Modify `_factor_filter()` in `scripts/tech_gamma_research_filters.py` so it includes the approved flow-only names:

```python
        case "foreign_positive" | "foreign_flow_positive":
            return features["foreign_flow_to_cap"].gt(0.0)
        case "institution_positive" | "institution_flow_positive":
            return features["institution_flow_to_cap"].gt(0.0)
        case "foreign_or_institution_positive":
            return features["foreign_flow_to_cap"].gt(0.0) | features["institution_flow_to_cap"].gt(0.0)
        case "foreign_and_institution_positive":
            return features["foreign_flow_to_cap"].gt(0.0) & features["institution_flow_to_cap"].gt(0.0)
```

Keep existing OP cases for backward compatibility.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/scripts/test_tech_gamma_relative_research.py::test_flow_filters_use_prior_trailing_flow_to_cap -q
```

Expected: PASS.

### Task 2: Constrain Breakout Grid to Flow-Only Schema

**Files:**
- Modify: `scripts/tech_gamma_breakout_grid_specs.py`
- Test: `tests/scripts/test_tech_gamma_relative_research.py`

- [ ] **Step 1: Write the failing test**

Replace `test_breakout_grid_is_structural_and_bounded_without_absolute_055_cutoff()` with:

```python
def test_breakout_grid_is_flow_filtered_without_op_filters() -> None:
    specs = build_strategy_specs(max_strategies=5000)
    filters = {spec.config.factor_filter for spec in specs}
    lookbacks = {spec.config.factor_lookback_days for spec in specs}
    positivity_lookbacks = {spec.config.positivity_lookback_days for spec in specs}
    benchmarks = {spec.config.positivity_benchmark for spec in specs}

    assert len(specs) == 5000
    assert len({spec.name for spec in specs}) == 5000
    assert filters == {
        "none",
        "foreign_positive",
        "institution_positive",
        "foreign_or_institution_positive",
        "foreign_and_institution_positive",
    }
    assert not any("op_" in factor_filter for factor_filter in filters)
    assert lookbacks == {20, 40, 60}
    assert positivity_lookbacks == {60, 90, 126}
    assert benchmarks == {"sector_cap_weighted", "index_cap_weighted"}
    assert all(spec.config.min_daily_positivity == 0.0 for spec in specs)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/scripts/test_tech_gamma_relative_research.py::test_breakout_grid_is_flow_filtered_without_op_filters -q
```

Expected: FAIL because the current grid still includes OP filters and the older positivity dimensions.

- [ ] **Step 3: Write minimal implementation**

Modify `_sampled_grid()` in `scripts/tech_gamma_breakout_grid_specs.py`:

```python
            (60, 90, 126),
            ("sector_cap_weighted", "index_cap_weighted"),
            (0.0, 0.02, 0.05),
            (
                "none",
                "foreign_positive",
                "institution_positive",
                "foreign_or_institution_positive",
                "foreign_and_institution_positive",
            ),
            (20, 40, 60),
```

Do not change execution dimensions in this task.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/scripts/test_tech_gamma_relative_research.py::test_breakout_grid_is_flow_filtered_without_op_filters -q
```

Expected: PASS.

### Task 3: Expose Flow Filter Choices in CLI

**Files:**
- Modify: `scripts/run_tech_gamma_long_only.py`
- Test: `tests/scripts/test_run_tech_gamma_long_only.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/scripts/test_run_tech_gamma_long_only.py`:

```python
def test_parse_args_accepts_flow_only_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_tech_gamma_long_only.py",
            "--factor-filter",
            "foreign_or_institution_positive",
        ],
    )

    from scripts.run_tech_gamma_long_only import parse_args

    args = parse_args()

    assert args.factor_filter == "foreign_or_institution_positive"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/scripts/test_run_tech_gamma_long_only.py::test_parse_args_accepts_flow_only_filter -q
```

Expected: FAIL because argparse rejects the new choice.

- [ ] **Step 3: Write minimal implementation**

In `scripts/run_tech_gamma_long_only.py`, update the `--factor-filter` choices tuple to include:

```python
            "foreign_positive",
            "institution_positive",
            "foreign_or_institution_positive",
            "foreign_and_institution_positive",
```

Keep existing values to avoid breaking saved configs.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/scripts/test_run_tech_gamma_long_only.py::test_parse_args_accepts_flow_only_filter -q
```

Expected: PASS.

### Task 4: Verify Integrated Research Pipeline

**Files:**
- Existing test suite only

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest tests/scripts/test_tech_gamma_relative_research.py tests/scripts/test_run_tech_gamma_long_only.py tests/scripts/test_tech_gamma_positivity.py tests/scripts/test_tech_gamma_universe.py tests/data/test_kr_stock_5m.py -q
```

Expected: PASS.

- [ ] **Step 2: Compile changed scripts**

Run:

```bash
uv run python -m py_compile scripts/tech_gamma_research_filters.py scripts/tech_gamma_breakout_grid_specs.py scripts/run_tech_gamma_long_only.py scripts/run_tech_gamma_breakout_grid.py
```

Expected: exit code 0.

- [ ] **Step 3: Smoke-check spec generation**

Run:

```bash
uv run python - <<'PY'
from scripts.tech_gamma_breakout_grid_specs import build_strategy_specs

specs = build_strategy_specs(max_strategies=15)
print(len(specs))
print(sorted({spec.config.factor_filter for spec in specs}))
PY
```

Expected output includes `15` and only flow filter names from the approved set.

### Task 5: Commit Implementation

**Files:**
- Modify: `scripts/tech_gamma_research_filters.py`
- Modify: `scripts/tech_gamma_breakout_grid_specs.py`
- Modify: `scripts/run_tech_gamma_long_only.py`
- Modify: `tests/scripts/test_tech_gamma_relative_research.py`
- Modify: `tests/scripts/test_run_tech_gamma_long_only.py`
- Add: `docs/superpowers/plans/2026-06-29-flow-filtered-breakout-implementation.md`

- [ ] **Step 1: Inspect final diff**

Run:

```bash
git diff -- scripts tests docs/superpowers/plans/2026-06-29-flow-filtered-breakout-implementation.md
```

Expected: only the planned flow-filter implementation, tests, and plan file.

- [ ] **Step 2: Commit with Lore protocol**

Run:

```bash
git add scripts/tech_gamma_research_filters.py scripts/tech_gamma_breakout_grid_specs.py scripts/run_tech_gamma_long_only.py tests/scripts/test_tech_gamma_relative_research.py tests/scripts/test_run_tech_gamma_long_only.py docs/superpowers/plans/2026-06-29-flow-filtered-breakout-implementation.md
git commit -m "Prefer flow-confirmed breakout variants for first rerun" \
  -m "The first rerun should isolate whether foreign or institutional sponsorship improves positivity-supported 52-week breakouts, so the breakout grid removes OP filters and keeps flow as simple prior-data positive gates." \
  -m "Constraint: User requested flow > 0 filters before score blending." \
  -m "Rejected: Blend flow into signal score | less interpretable before proving the gate helps." \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: focused pytest suite and py_compile commands from Task 4" \
  -m "Not-tested: Full 5000-strategy backtest runtime."
```

Expected: commit succeeds.

