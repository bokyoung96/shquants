# Backtesting Calculation Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce dashboard-launched benchmark strategy Backtest Calculation time without changing portfolio math or strategy meaning.

**Architecture:** Use the new Backtest Timing data to optimize the largest measured stage first. The measured bottleneck is `_BenchmarkOverlayConstruction.build()`, shared by `benchmark_overlay` and `benchmark_tilt`; replace its per-date pandas `Series` construction path with an equivalent numpy array path while keeping the old scalar helper as the reference behavior in tests.

**Tech Stack:** Python, pandas, numpy, pytest

---

## Measured Baseline

Dashboard preset run with `write_report_assets=False` and `profile=True`:

| Strategy | Total | Plan build | Main bottleneck |
| --- | ---: | ---: | --- |
| `benchmark_overlay` | ~20.57s | ~18.80s | construction ~9.93s, signal ~5.84s |
| `benchmark_tilt` | ~15.53s | ~13.93s | construction ~9.21s |

Engine execution is only ~0.2-0.4s and is not the first optimization target.

## File Map

- Modify: `backtesting/strategies/benchmark_overlay.py`
- Test: `tests/strategies/test_registry.py`
- Docs: `docs/superpowers/plans/2026-05-13-backtesting-calculation-optimization.md`

## Tasks

### Task 1: Lock active overlay equivalence

- [ ] Add a test in `tests/strategies/test_registry.py` that calls the existing `_build_active_overlay()` and the new `_build_active_overlay_values()` with the same representative signal/base/sector data.
- [ ] Run only that test and verify it fails because `_build_active_overlay_values()` does not exist.

### Task 2: Add numpy active overlay helper

- [ ] Implement `_build_active_overlay_values()` in `_BenchmarkOverlayConstruction`.
- [ ] Keep `_build_active_overlay()` unchanged as the reference helper for tests and low-risk comparison.
- [ ] Run the new test and verify it passes.

### Task 3: Switch construction build to arrays

- [ ] Update `_BenchmarkOverlayConstruction.build()` to allocate numpy arrays for weights and selection masks.
- [ ] For each date, compute base weights and active overlay using `_build_active_overlay_values()`.
- [ ] Return the same `ConstructionResult` shape and columns as before.
- [ ] Run strategy tests and backtesting run tests.

### Task 4: Measure and verify

- [ ] Re-run dashboard preset timing measurement.
- [ ] Run focused verification: `uv run python -m pytest tests/strategies tests/test_run.py tests/dashboard tests/reporting`.
- [ ] If timing improves without test regressions, report the before/after by strategy.
