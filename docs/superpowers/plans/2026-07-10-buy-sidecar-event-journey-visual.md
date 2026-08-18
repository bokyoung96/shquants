# Buy-Sidecar Event Journey Visual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current sidecar timing chart with the approved Event Journey visualization while preserving all research calculations.

**Architecture:** Keep `run_study()` and every research output unchanged except for passing `holding_summary` into `plot_study()`. Rebuild `plot_study()` as a single 16:9 Matplotlib composition with a timeline, direct KPI labels, and two compact supporting comparisons.

**Tech Stack:** Python 3.11, pandas, NumPy, Matplotlib, pytest

---

### Task 1: Lock The Visual Output Contract

**Files:**
- Modify: `tests/scripts/test_run_buy_sidecar_entry_study.py`

- [ ] Add a test that supplies minimal canonical, execution, and holding summaries to `plot_study()`.
- [ ] Assert that the PNG exists, has an aspect ratio near 16:9, exceeds 1600x900, and has nonblank pixel variance.
- [ ] Run `uv run python -m pytest tests/scripts/test_run_buy_sidecar_entry_study.py -q` and confirm the new test fails against the current signature/layout.

### Task 2: Implement The Event Journey Figure

**Files:**
- Modify: `scripts/run_buy_sidecar_entry_study.py`

- [ ] Add Korean font selection with a system-font fallback and keep minus signs readable.
- [ ] Change `plot_study()` to accept `holding_summary`.
- [ ] Build the headline, context line, halt/release timeline, robust windows, A+3/R+3 markers, three KPI columns, entry comparison, holding-risk comparison, and limitation footer.
- [ ] Keep the existing output path `results/buy_sidecar_entry_study/buy_sidecar_timing_robustness.png` so downstream links do not break.
- [ ] Pass `holding_summary` from `run_study()` into `plot_study()`.
- [ ] Run the focused test and confirm it passes.

### Task 3: Regenerate And Visually Verify

**Files:**
- Regenerate: `results/buy_sidecar_entry_study/buy_sidecar_timing_robustness.png`

- [ ] Run `uv run python scripts/run_buy_sidecar_entry_study.py`.
- [ ] Compare the generated figure with the approved Event Journey reference using the visual-verdict contract.
- [ ] If the score is below 90, apply only the verdict's concrete spacing, hierarchy, or legibility corrections and rerun the verdict.
- [ ] Run `uv run python -m pytest tests/scripts/test_run_buy_sidecar_entry_study.py tests/etc/test_sidecar.py tests/etc/test_sell_sidecar_economics.py -q`.
- [ ] Run `uv run python -m py_compile scripts/run_buy_sidecar_entry_study.py tests/scripts/test_run_buy_sidecar_entry_study.py` and `git diff --check`.
