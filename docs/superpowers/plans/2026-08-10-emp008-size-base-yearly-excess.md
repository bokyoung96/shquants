# EMP008 Size Base and Yearly Excess Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Size-only EMP008 portfolio to all value-measure comparison outputs and generate KOSPI200-relative yearly excess-return subplots containing the existing, first-adjustment, second-adjustment, and four value-measure portfolios.

**Architecture:** Extend the immutable EMP008 factor registry with one Size-only factor set, then keep the experiment runner responsible for the four directly-run portfolios. Load the three historical comparison series only for the new yearly visualization, align all inputs on common dates, and compute exact relative wealth against KOSPI200 with an annual zero reset.

**Tech Stack:** Python, pandas, matplotlib, pytest, Ruff, existing EMP008 backtest runner.

---

### Task 1: Register the Size-only EMP008 factor set

**Files:**
- Modify: `backtesting/strategies/emp008/factor_registry.py`
- Modify: `tests/strategies/test_emp008_factor_registry.py`

- [ ] **Step 1: Write the failing registry test**

Add `FactorSetId.SIZE_ONLY` to the expected enum order and assert the definition is exactly:

```python
assert get_factor_set_definition(FactorSetId.SIZE_ONLY) == FactorSetDefinition(
    id=FactorSetId.SIZE_ONLY,
    factors=(FactorId.LN_MARKET_CAP,),
    constrain_expected_alpha_to_direction=True,
)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py -q`

Expected: failure because `FactorSetId.SIZE_ONLY` is not defined.

- [ ] **Step 3: Implement the registry definition**

Add the enum and immutable mapping entry:

```python
class FactorSetId(StrEnum):
    # existing members
    SIZE_ONLY = "size_only"

_FACTOR_SET_DEFINITIONS = {
    # existing definitions
    FactorSetId.SIZE_ONLY: FactorSetDefinition(
        id=FactorSetId.SIZE_ONLY,
        factors=(FactorId.LN_MARKET_CAP,),
        constrain_expected_alpha_to_direction=True,
    ),
}
```

- [ ] **Step 4: Run registry tests**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the isolated registry change**

Stage only the registry and its test. Use a Lore-format commit explaining that Size-only is an additional portfolio, not a benchmark.

### Task 2: Add Size-only to the comparison and implement yearly reset frames

**Files:**
- Modify: `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`
- Modify: `tests/strategies/test_emp008_size_value_measure_comparison.py`

- [ ] **Step 1: Write failing comparison tests**

Extend `DEFAULT_VARIANTS` expectations to start with `size_only`. Add a yearly-frame test using two years and assert every annual series starts at zero:

```python
yearly = _yearly_cumulative_excess_bp_frames(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
)
assert tuple(yearly) == (2024, 2025)
for frame in yearly.values():
    assert frame.iloc[0].eq(0.0).all()
```

Add an output test asserting `yearly_excess_returns.png` is generated and included in the payload. Assert the historical input requires these columns: `기존 EMP008`, `1차수정 EMP008`, `2차수정 EMP008`, and `KOSPI200 BM`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: failures for the absent Size-only variant and yearly helper/output.

- [ ] **Step 3: Add Size-only display configuration**

Update the variant and plotting maps:

```python
DEFAULT_VARIANTS = (
    "size_only",
    "size_value_fcf_tev",
    "size_value_dividend_fy0",
    "size_value_dividend_ttm",
)

_DISPLAY_LABELS["size_only"] = "Size-only Base"
```

Use a distinct neutral purple color for Size-only while preserving the existing three colors.

- [ ] **Step 4: Implement historical-series loading and annual reset calculation**

Add a default historical comparison path and validate its required columns. Build one aligned return frame containing the four directly-run portfolios and three historical EMP008 portfolios. For each year, normalize portfolio and benchmark wealth by the first common date and compute:

```python
relative = portfolio_wealth.div(benchmark_wealth, axis=0).sub(1.0).mul(10_000.0)
relative = relative.sub(relative.iloc[0])
```

The subtraction guarantees an exact zero start even when the first annual row contains a return.

- [ ] **Step 5: Implement the yearly subplot image**

Create a 4x2 subplot grid for 2020–2026, remove the unused eighth axis, use a global shared y-range, add a zero line to every panel, and use one figure-level legend. Write `yearly_excess_returns.png` and add it to the returned output payload and interpretation document.

- [ ] **Step 6: Update cumulative outputs**

Regenerate cumulative returns, exact KOSPI200-relative cumulative excess returns, the dashboard, manifest, CSV, and Excel outputs with Size-only included as the fourth directly-run portfolio.

- [ ] **Step 7: Run comparison tests**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit the comparison change**

Stage only the experiment runner and its focused test. Use a Lore-format commit recording KOSPI200 as the invariant denominator.

### Task 3: Execute the cost-aligned run and verify artifacts

**Files:**
- Regenerate: `backtesting/strategies/emp008/tests/size_value_measure_comparison/**`

- [ ] **Step 1: Run the comparison through 2026-06-30**

Run:

```powershell
uv run python -m backtesting.strategies.emp008.experiments.size_value_measure_comparison --end 2026-06-30
```

Expected: Size-only weights/backtest run once; the other three portfolios reuse compatible weights and cached costed returns.

- [ ] **Step 2: Verify manifest and dates**

Assert `manifest.json` records end `2026-06-30`, fee `0.0002`, sell tax `0.0015`, slippage `0.0005`, benchmark `IKS200`, and four variants including `size_only`.

- [ ] **Step 3: Run static and focused verification**

Run:

```powershell
uv run ruff check backtesting/strategies/emp008/factor_registry.py backtesting/strategies/emp008/experiments/size_value_measure_comparison.py tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py
uv run pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py -q
```

Expected: Ruff passes and all focused tests pass.

- [ ] **Step 4: Inspect all generated images**

Visually inspect `cumulative_returns.png`, `cumulative_excess_returns.png`, `performance_dashboard.png`, and `yearly_excess_returns.png`. Verify labels are readable, no values are clipped, annual panels share a y-range, and every annual line starts on the zero axis.

- [ ] **Step 5: Report results**

Provide links to all four images and the updated performance workbook. Report Size-only performance metrics and its exact KOSPI200-relative cumulative excess alongside the other portfolios.

### Task 4: Localize the yearly chart and export yearly tables

**Files:**
- Modify: `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`
- Modify: `tests/strategies/test_emp008_size_value_measure_comparison.py`
- Regenerate: `backtesting/strategies/emp008/tests/size_value_measure_comparison/**`

- [ ] **Step 1: Write failing localization and table tests**

Assert the yearly output payload contains `yearly_returns_csv`, `yearly_excess_returns_csv`, and `yearly_performance_xlsx`. Build a two-year fixture and assert:

```python
returns, excess = _yearly_performance_tables(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
)
assert returns.index.tolist() == [2024, 2025]
assert excess["KOSPI200 BM"].eq(0.0).all()
assert excess.loc[2025, "Size-only Base"] == pytest.approx(yearly_frames[2025]["size_only"].iloc[-1])
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: failures for missing yearly table helper and output paths.

- [ ] **Step 3: Implement localized labels and yearly tables**

Use Korean figure strings:

```python
fig.suptitle("KOSPI200 대비 연도별 누적 초과성과 (매년 0bp 재설정)")
fig.supxlabel("날짜")
fig.supylabel("누적 초과성과 (bp)")
```

Map historical labels to `기존 EMP008`, `1차수정 EMP008`, and `2차수정 EMP008`. Compute annual returns from annually normalized wealth and compute excess endpoints from `_yearly_cumulative_excess_bp_frames`, adding a zero `KOSPI200 BM` column.

- [ ] **Step 4: Export CSV and Excel artifacts**

Write `yearly_returns_pct.csv`, `yearly_excess_returns_bp.csv`, and `yearly_performance.xlsx` with sheets `yearly_returns_pct` and `yearly_excess_bp`. Add all paths to the output payload.

- [ ] **Step 5: Verify and regenerate**

Run the focused tests, Ruff, and the comparison command through `2026-06-30`. Inspect the Korean yearly image and verify CSV/Excel values match subplot endpoints.
