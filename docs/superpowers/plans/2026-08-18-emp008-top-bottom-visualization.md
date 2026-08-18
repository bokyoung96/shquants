# EMP008 Top/Bottom Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three Korean WI26-WICS Top/Bottom consensus heatmaps covering 2020 onward.

**Architecture:** A standalone report module reads the existing WI26 and WICS Top/Bottom workbooks by column position, normalizes their rows, and aggregates candidate appearances into signed consensus scores. Pure transformation functions are unit tested; plotting and report export consume their frames.

**Tech Stack:** Python, pandas, matplotlib, openpyxl, pytest.

---

### Task 1: Candidate-consensus calculation

**Files:**
- Create: `tests/strategies/test_emp008_factor_weight_top_bottom_plots.py`
- Create: `backtesting/strategies/emp008/reports/factor_weight_top_bottom_plots.py`

- [ ] Write a failing test where two Top appearances and one Bottom appearance produce a consensus score of +1.
- [ ] Implement workbook normalization and `Top count - Bottom count` aggregation from 2020 onward.
- [ ] Verify the focused transformation tests pass.

### Task 2: Heatmap report

**Files:**
- Modify: `backtesting/strategies/emp008/reports/factor_weight_top_bottom_plots.py`
- Modify: `tests/strategies/test_emp008_factor_weight_top_bottom_plots.py`

- [ ] Write a failing report test requiring full-period, annual-summary, and 2023-detail PNGs plus CSV and manifest outputs.
- [ ] Render WI26/WICS panels on shared stock/date axes with Korean labels and a centered diverging color scale.
- [ ] Generate the report from the saved workbooks and verify all outputs are nonempty.
- [ ] Visually inspect all three PNG files and run the focused test suite.
