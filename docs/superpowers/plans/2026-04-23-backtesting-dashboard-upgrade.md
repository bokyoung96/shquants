# Backtesting Dashboard Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `backtesting/reporting` to export richer static dashboard pictures and benchmark-optional analytics without touching unrelated areas.

**Architecture:** Keep the current reporting pipeline, enrich the analytics/snapshot layer, make report profile logic explicit, and render a dashboard-first single-run report from shared analytics data. Preserve existing builder entry points while thinning internal responsibilities.

**Tech Stack:** Python, pandas, Plotly, pytest

---

## File Map

- Modify: `backtesting/reporting/analytics.py`
- Modify: `backtesting/reporting/models.py`
- Modify: `backtesting/reporting/snapshots.py`
- Modify: `backtesting/reporting/figures.py`
- Modify: `backtesting/reporting/comparison_figures.py`
- Modify: `backtesting/reporting/tables_single.py`
- Modify: `backtesting/reporting/tables_comparison.py`
- Modify: `backtesting/reporting/builder.py`
- Test: `tests/reporting/test_snapshots.py`
- Test: `tests/reporting/test_figures.py`
- Test: `tests/reporting/test_tables.py`
- Test: `tests/reporting/test_builder.py`

## Execution Tasks

### Task 1: Lock new reporting/profile behavior with tests
- [ ] Add/extend tests for benchmark-free snapshots and profile-aware output.
- [ ] Add/extend tests for richer dashboard asset keys and tables.
- [ ] Run targeted reporting tests to confirm failures before implementation.

### Task 2: Enrich analytics and profile modeling
- [ ] Add report profile model + optional benchmark semantics.
- [ ] Add richer core/benchmark/tracking metrics and rolling helpers.
- [ ] Keep APIs short and class-friendly.

### Task 3: Build richer snapshots and dashboard figures
- [ ] Expand snapshot assembly to support optional benchmark and profile-driven rendering.
- [ ] Upgrade single-run dashboard figure composition to include the new panels.
- [ ] Keep output static-export oriented.

### Task 4: Update tables/comparison outputs
- [ ] Add richer metric rows and benchmark-conditional sections.
- [ ] Preserve comparison reporting while handling missing benchmark metrics safely.

### Task 5: Verify and iterate until green
- [ ] Run targeted reporting test suite.
- [ ] Run broader dependent tests if reporting contract changes require it.
- [ ] Fix regressions until green.
