# Backtest Limit Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the current explicit OpenClaw backtesting limits by adding signal-date evaluation schedules, first-class shorting assumptions, and a second sector-neutral group-budget variant.

**Architecture:** Keep the canonical surface in `ExecutionSpec` and keep portfolio math in backtesting. `ScheduleSpec` owns signal evaluation cadence, `ShortingSpec` owns short availability and borrow assumptions, and `PortfolioShapeSpec` owns sector-neutral budget allocation.

**Tech Stack:** Python dataclasses, pandas, pytest, existing backtesting engine/spec/construction modules.

---

### Task 1: Signal-Date Evaluation Schedule

**Files:**
- Modify: `backtesting/specs/models.py`
- Modify: `backtesting/specs/loader.py`
- Modify: `backtesting/specs/plan_builder.py`
- Modify: `backtesting/run.py`
- Test: `tests/specs/test_loader.py`
- Test: `tests/specs/test_plan_builder.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write failing tests**

Add loader, plan-builder, and runner tests for:
- `schedule.kind = "signal_dates"` with nested `evaluation = {"kind": "named", "name": "weekly"}`.
- Mid-week signal changes are ignored until the weekly evaluation date.
- Rebalance flags still come from final target-weight changes.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `uv run python -m pytest tests/specs/test_loader.py -k signal_dates tests/specs/test_plan_builder.py -k signal_dates tests/test_run.py -k signal_dates -q`

- [ ] **Step 3: Implement minimal support**

Add `ScheduleEvaluationSpec`, parse nested `evaluation`, freeze signal-date target weights to evaluation flags in `plan_builder`, and let `run.py` continue deriving trade dates from final target-weight changes.

- [ ] **Step 4: Verify green**

Run the same focused test command and inspect output.

### Task 2: Shorting Assumptions

**Files:**
- Modify: `backtesting/specs/models.py`
- Modify: `backtesting/specs/loader.py`
- Modify: `backtesting/specs/resolve.py`
- Modify: `backtesting/specs/plan_builder.py`
- Modify: `backtesting/engine/core.py`
- Modify: `backtesting/execution/costs.py`
- Modify: `backtesting/run.py`
- Test: `tests/specs/test_loader.py`
- Test: `tests/specs/test_plan_builder.py`
- Test: `tests/specs/test_resolve.py`
- Test: `tests/engine/test_core.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write failing tests**

Add tests for:
- Parsing `shorting.enabled`, `borrow_fee_annual`, `shortable_field`, and `cash_collateral_ratio`.
- Requiring `shorting.enabled = true` for negative portfolio-shape weights.
- Zeroing short targets for symbols where `shortable_field` is false.
- Applying daily borrow fee to open short notional in engine equity.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `uv run python -m pytest tests/specs/test_loader.py -k shorting tests/specs/test_plan_builder.py -k shorting tests/specs/test_resolve.py -k shorting tests/engine/test_core.py -k borrow tests/test_run.py -k shorting -q`

- [ ] **Step 3: Implement minimal support**

Add `ShortingSpec`, feature resolution for `shortable_field`, target-weight short masking, engine borrow fee accrual, and runner plumbing from spec to cost model.

- [ ] **Step 4: Verify green**

Run the focused command and inspect output.

### Task 3: Sector-Neutral Budget Variant

**Files:**
- Modify: `backtesting/specs/models.py`
- Modify: `backtesting/specs/loader.py`
- Modify: `backtesting/construction/sector_neutral.py`
- Modify: `backtesting/specs/plan_builder.py`
- Test: `tests/specs/test_loader.py`
- Test: `tests/construction/test_rules.py`
- Test: `tests/specs/test_plan_builder.py`

- [ ] **Step 1: Write failing tests**

Add tests for `portfolio_shape.group_budget = "proportional_selected"` where groups with more selected longs/shorts receive more gross budget than smaller groups.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `uv run python -m pytest tests/specs/test_loader.py -k group_budget tests/construction/test_rules.py -k group_budget tests/specs/test_plan_builder.py -k group_budget -q`

- [ ] **Step 3: Implement minimal support**

Add `group_budget` to `PortfolioShapeSpec`, pass it to `SectorNeutralTopBottom`, and support `equal_group` plus `proportional_selected`.

- [ ] **Step 4: Verify green**

Run the focused command and inspect output.

### Task 4: Docs, Limits, and Speed Comparison

**Files:**
- Modify: `docs/openclaw/PLAYBOOK.md`
- Modify: `CONTEXT.md`

- [ ] **Step 1: Update OpenClaw docs**

Replace the old missing-limit bullets with supported spec examples and remaining advanced limits.

- [ ] **Step 2: Run full verification**

Run: `uv run python -m pytest -q`

- [ ] **Step 3: Run speed comparison**

Run a before/after style construction benchmark for long-short and sector-neutral paths where current code is compared against local baseline implementations embedded in the benchmark script.
