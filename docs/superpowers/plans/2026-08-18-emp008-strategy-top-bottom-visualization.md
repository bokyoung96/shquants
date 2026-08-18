# EMP008 Strategy-Level Top/Bottom Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one 2020+ WI26/WICS replacement-count summary and nine strategy-specific Top/Bottom holding timelines without aggregating candidates.

**Architecture:** Add a focused report module that normalizes the existing workbook rows, validates paired strategy/date coverage, calculates set replacement counts, and renders summary/detail heatmaps. Keep the existing consensus report unchanged and write all new images and full-universe CSV matrices to a separate output directory.

**Tech Stack:** Python, pandas, NumPy, matplotlib, openpyxl, pytest

---

## File Structure

- Create `backtesting/strategies/emp008/reports/factor_weight_strategy_top_bottom_plots.py`: input normalization, pair validation, replacement counts, state matrices, plotting, CLI.
- Create `tests/strategies/test_emp008_factor_weight_strategy_top_bottom_plots.py`: calculation, isolation, validation, and report-output tests.
- Create outputs under `backtesting/strategies/emp008/tests/factor_weight_top_bottom_comparison/by_strategy/`: summary PNG/CSV, nine detail PNGs, eighteen full state CSVs, and a manifest.

### Task 1: Lock strategy-level calculations with tests

**Files:**
- Create: `tests/strategies/test_emp008_factor_weight_strategy_top_bottom_plots.py`

- [ ] **Step 1: Write failing tests for state values and replacement counts**

```python
def test_build_strategy_state_matrix_keeps_candidates_separate():
    top, bottom = sample_rows()
    matrix = build_strategy_state_matrix(top, bottom, candidate_id="s50_m30_e10_v10")
    assert matrix.loc[pd.Timestamp("2020-01-31"), "A001"] == 1
    assert "A999" not in matrix.columns


def test_calculate_replacement_counts_uses_set_changes():
    result = calculate_replacement_counts(wi26_top, wi26_bottom, wics_top, wics_bottom)
    row = result.loc[("s50_m30_e10_v10", pd.Timestamp("2020-01-31"))]
    assert row.to_dict() == {"top_replacements": 1, "bottom_replacements": 2, "total_replacements": 3}
```

- [ ] **Step 2: Write failing tests for strict pair validation**

```python
def test_validate_paired_rows_rejects_strategy_mismatch():
    with pytest.raises(ValueError, match="전략 집합 불일치"):
        validate_paired_rows(wi26_top, wi26_bottom, wics_top_without_one_strategy, wics_bottom)


def test_validate_paired_rows_rejects_date_mismatch():
    with pytest.raises(ValueError, match="리밸런싱일 불일치"):
        validate_paired_rows(wi26_top, wi26_bottom, wics_top_without_one_date, wics_bottom)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `C:\Users\CHECK\anaconda3\python.exe -m pytest tests/strategies/test_emp008_factor_weight_strategy_top_bottom_plots.py -q`

Expected: collection failure because the new report module does not exist.

### Task 2: Implement normalization, validation, and calculations

**Files:**
- Create: `backtesting/strategies/emp008/reports/factor_weight_strategy_top_bottom_plots.py`
- Test: `tests/strategies/test_emp008_factor_weight_strategy_top_bottom_plots.py`

- [ ] **Step 1: Normalize workbook rows into a stable internal schema**

```python
def load_top_bottom_rows(path: Path, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    def read(sheet: str, side: str) -> pd.DataFrame:
        raw = pd.read_excel(path, sheet_name=sheet)
        rows = raw.iloc[:, [0, 2, 7, 8, 9]].copy()
        rows.columns = ["date", "candidate_id", "rank", "ticker", "stock_name"]
        rows["date"] = pd.to_datetime(rows["date"])
        rows = rows.loc[rows["date"] >= pd.Timestamp(start)]
        rows["side"] = side
        return rows.sort_values(["candidate_id", "date", "rank"]).reset_index(drop=True)

    return read("Top 10", "top"), read("Bottom 10", "bottom")
```

- [ ] **Step 2: Validate exact strategy/date coverage and ten rows per side**

Implement `validate_paired_rows(...)` to compare candidate sets, compare date sets within every candidate, and require each `(candidate_id, date)` group in every frame to have exactly ten unique tickers. Error messages must include the mismatched strategy or date.

- [ ] **Step 3: Calculate state matrices and integer replacement counts**

```python
def build_strategy_state_matrix(top, bottom, *, candidate_id):
    selected = pd.concat([
        top.loc[top["candidate_id"].eq(candidate_id)].assign(state=1),
        bottom.loc[bottom["candidate_id"].eq(candidate_id)].assign(state=-1),
    ])
    return selected.pivot(index="date", columns="ticker", values="state").fillna(0).astype("int8")


def replacement_count(left: set[str], right: set[str]) -> int:
    return 10 - len(left & right)
```

Build one row per candidate/date containing `top_replacements`, `bottom_replacements`, and their sum `total_replacements`.

- [ ] **Step 4: Run calculation tests**

Run: `C:\Users\CHECK\anaconda3\python.exe -m pytest tests/strategies/test_emp008_factor_weight_strategy_top_bottom_plots.py -q`

Expected: calculation and validation tests pass.

### Task 3: Render summary and strategy details

**Files:**
- Modify: `backtesting/strategies/emp008/reports/factor_weight_strategy_top_bottom_plots.py`
- Modify: `tests/strategies/test_emp008_factor_weight_strategy_top_bottom_plots.py`

- [ ] **Step 1: Add a failing end-to-end output test**

Create paired temporary workbooks containing two strategies and four dates. Call `generate_strategy_top_bottom_report(...)` and assert:

```python
assert outputs["summary_png"].stat().st_size > 0
assert outputs["replacement_csv"].stat().st_size > 0
assert len(outputs["detail_pngs"]) == 2
assert len(outputs["state_csvs"]) == 4
assert outputs["manifest_json"].stat().st_size > 0
```

- [ ] **Step 2: Render the replacement-count summary**

Pivot `total_replacements` to candidate-by-date, use a sequential white-to-dark-red colormap fixed to 0–20, label years on the x-axis, draw year boundaries, and show full candidate IDs plus Korean factor-weight descriptions on the y-axis.

- [ ] **Step 3: Select deterministic detail tickers**

For each strategy, combine WI26 and WICS matrices. Rank positive and negative appearances separately by total occurrence count, break ties by ticker, select fifteen from each side, remove duplicates, and align both panels to the same rows and dates.

- [ ] **Step 4: Render nine detail images**

For each strategy, render WI26 and WICS side by side with a fixed discrete scale: blue `-1`, white `0`, red `+1`. Use Korean stock names with ticker codes, year ticks, and a subtitle spelling out the four factor weights.

- [ ] **Step 5: Write complete CSVs and manifest**

Write `replacement_counts.csv`, two full-universe state matrices per strategy, and `manifest.json` containing start/end dates, strategy count, strategy labels, definitions, and all output paths. Use UTF-8 with BOM for CSVs and UTF-8 for JSON.

- [ ] **Step 6: Run the new test file**

Run: `C:\Users\CHECK\anaconda3\python.exe -m pytest tests/strategies/test_emp008_factor_weight_strategy_top_bottom_plots.py -q`

Expected: all tests pass.

### Task 4: Generate and verify the real report

**Files:**
- Generate: `backtesting/strategies/emp008/tests/factor_weight_top_bottom_comparison/by_strategy/*`

- [ ] **Step 1: Run the report CLI on the WI26 and WICS workbooks**

Run: `C:\Users\CHECK\anaconda3\python.exe -m backtesting.strategies.emp008.reports.factor_weight_strategy_top_bottom_plots`

Expected: JSON listing one summary PNG, nine detail PNGs, nineteen CSVs, and one manifest JSON.

- [ ] **Step 2: Inspect rendered images**

Open the summary image and at least the highest-difference and lowest-difference strategy detail images. Confirm Korean labels render, WI26/WICS axes align, legends do not overlap, and 2020–2026 is visible.

- [ ] **Step 3: Verify artifacts and ranges**

Confirm all output files are non-empty, the manifest has nine strategies, the date range begins in 2020, and `total_replacements` stays in `[0, 20]`.

- [ ] **Step 4: Run related regression tests**

Run: `C:\Users\CHECK\anaconda3\python.exe -m pytest tests/strategies/test_emp008_factor_weight_strategy_top_bottom_plots.py tests/strategies/test_emp008_factor_weight_top_bottom_plots.py tests/strategies/test_emp008_factor_weight_grid_comparison.py tests/strategies/test_emp008_factor_weight_grid_search.py -q`

Expected: all related tests pass with zero failures.
