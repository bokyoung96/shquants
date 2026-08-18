# EMP008 Size + Value Measure Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and persist a like-for-like EMP008 comparison of Size paired separately with FCF/TEV, FY0 dividend yield, and TTM dividend yield.

**Architecture:** Register three explicit two-factor sets so the normal EMP008 loader, preprocessing, risk model, and optimizer remain authoritative. Add a focused experiment runner that executes the same no-cost backtest for each set, aggregates portfolio and excess-return metrics, and writes reproducible artifacts under `backtesting/strategies/emp008/tests/size_value_measure_comparison/`.

**Tech Stack:** Python 3.11+, pandas, matplotlib, openpyxl, pytest, existing EMP008 optimizer and BacktestRunner.

---

## File Structure

- Modify `backtesting/strategies/emp008/factor_registry.py`: declare the three production-valid two-factor sets and their snapshot/direction policies.
- Create `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`: orchestration, metric aggregation, charts, manifest, Markdown interpretation, and CLI.
- Modify `backtesting/strategies/emp008/README.md`: document the experiment command and output contract.
- Modify `tests/strategies/test_emp008_factor_registry.py`: lock exact membership, datasets, and public CLI visibility of the new sets while preserving current uncommitted registry work.
- Create `tests/strategies/test_emp008_size_value_measure_comparison.py`: test validation, aggregation, report ranking, output paths, and orchestration with fakes.
- Create at runtime `backtesting/strategies/emp008/tests/size_value_measure_comparison/`: empirical weights, backtests, daily series, tables, plots, manifest, and interpretation.

### Task 1: Register the isolated two-factor portfolios

**Files:**
- Modify: `tests/strategies/test_emp008_factor_registry.py`
- Modify: `backtesting/strategies/emp008/factor_registry.py`

- [ ] **Step 1: Write the failing registry tests**

Extend the expected `FactorSetId` sequence and add an exact contract test:

```python
@pytest.mark.parametrize(
    ("factor_set", "value_factor", "snapshot_forward_days"),
    [
        (FactorSetId.SIZE_VALUE_FCF_TEV, FactorId.VALUE, 0),
        (FactorSetId.SIZE_VALUE_DIVIDEND_FY0, FactorId.DIVIDEND_YIELD_FY0, 7),
        (FactorSetId.SIZE_VALUE_DIVIDEND_TTM, FactorId.DIVIDEND_YIELD_TTM, 0),
    ],
)
def test_size_value_comparison_sets_isolate_one_value_measure(
    factor_set: FactorSetId,
    value_factor: FactorId,
    snapshot_forward_days: int,
) -> None:
    definition = get_factor_set_definition(factor_set)

    assert definition.factors == (FactorId.LN_MARKET_CAP, value_factor)
    assert definition.rank_transform_factors == ()
    assert definition.neutralize_large_benchmark_weight_factors == ()
    assert definition.constrain_expected_alpha_to_direction is True
    assert definition.snapshot_forward_days == snapshot_forward_days
    assert definition.diagnostics_only is False
```

Also update `factor_set_values()`, `strategy_factor_set_values()`, and the invalid-value error expectation to include the three new names in enum order.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py -q`

Expected: collection or assertion failure because the three enum members do not exist.

- [ ] **Step 3: Implement the three registry entries**

Add to `FactorSetId`:

```python
SIZE_VALUE_FCF_TEV = "size_value_fcf_tev"
SIZE_VALUE_DIVIDEND_FY0 = "size_value_dividend_fy0"
SIZE_VALUE_DIVIDEND_TTM = "size_value_dividend_ttm"
```

Add to `_FACTOR_SET_DEFINITIONS` before `ALL_FACTORS`:

```python
FactorSetId.SIZE_VALUE_FCF_TEV: FactorSetDefinition(
    id=FactorSetId.SIZE_VALUE_FCF_TEV,
    factors=(FactorId.LN_MARKET_CAP, FactorId.VALUE),
    constrain_expected_alpha_to_direction=True,
),
FactorSetId.SIZE_VALUE_DIVIDEND_FY0: FactorSetDefinition(
    id=FactorSetId.SIZE_VALUE_DIVIDEND_FY0,
    factors=(FactorId.LN_MARKET_CAP, FactorId.DIVIDEND_YIELD_FY0),
    constrain_expected_alpha_to_direction=True,
    snapshot_forward_days=7,
),
FactorSetId.SIZE_VALUE_DIVIDEND_TTM: FactorSetDefinition(
    id=FactorSetId.SIZE_VALUE_DIVIDEND_TTM,
    factors=(FactorId.LN_MARKET_CAP, FactorId.DIVIDEND_YIELD_TTM),
    constrain_expected_alpha_to_direction=True,
),
```

Do not rank-transform or large-benchmark-neutralize Size; this keeps the same Origin-style Size semantics across all three sets.

- [ ] **Step 4: Run the registry and data-loader tests and verify GREEN**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_factor_pipeline.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit only Task 1 paths with a Lore-format message**

Stage only the registry and registry-test paths. Preserve the pre-existing `adjust`/rank-transform edits in those files and describe the combined current diff honestly.

### Task 2: Build pure comparison aggregation and reporting

**Files:**
- Create: `tests/strategies/test_emp008_size_value_measure_comparison.py`
- Create: `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`

- [ ] **Step 1: Write failing tests for variant validation and metric aggregation**

Use deterministic daily return frames:

```python
def test_validate_variants_rejects_unknown_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unknown variants"):
        validate_variants(("not_registered",))
    with pytest.raises(ValueError, match="duplicate variants"):
        validate_variants(("size_value_fcf_tev", "size_value_fcf_tev"))


def test_build_comparison_tables_aligns_returns_and_ranks_excess() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    benchmark = pd.Series([0.0, 0.001, -0.001, 0.0], index=dates, name="benchmark")
    returns = {
        "size_value_fcf_tev": benchmark + 0.001,
        "size_value_dividend_fy0": benchmark - 0.001,
        "size_value_dividend_ttm": benchmark + 0.0005,
    }
    active_share = {name: {"mean_pct": float(i + 1)} for i, name in enumerate(returns)}

    summary, daily = build_comparison_tables(
        returns_by_variant=returns,
        benchmark_returns=benchmark,
        active_share_by_variant=active_share,
    )

    assert summary["variant"].tolist()[0] == "size_value_fcf_tev"
    assert set(daily) == {
        "benchmark_return",
        "size_value_fcf_tev_return",
        "size_value_fcf_tev_excess_return",
        "size_value_dividend_fy0_return",
        "size_value_dividend_fy0_excess_return",
        "size_value_dividend_ttm_return",
        "size_value_dividend_ttm_excess_return",
    }
    assert summary["annualized_excess_bp"].notna().all()
    assert summary["mean_active_share_pct"].tolist() == [1.0, 3.0, 2.0]
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: import failure because the experiment module does not exist.

- [ ] **Step 3: Implement constants, validation, and pure aggregation**

In the new module define:

```python
DEFAULT_VARIANTS = (
    "size_value_fcf_tev",
    "size_value_dividend_fy0",
    "size_value_dividend_ttm",
)
DEFAULT_OUTPUT_DIR = Path("backtesting/strategies/emp008/tests/size_value_measure_comparison")


def validate_variants(variants: tuple[str, ...]) -> tuple[FactorSetId, ...]:
    duplicates = tuple(name for name in dict.fromkeys(variants) if variants.count(name) > 1)
    if duplicates:
        raise ValueError(f"duplicate variants: {duplicates}")
    allowed = set(DEFAULT_VARIANTS)
    unknown = tuple(name for name in variants if name not in allowed)
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return tuple(parse_factor_set(name) for name in variants)
```

Implement `build_comparison_tables(...) -> tuple[pd.DataFrame, pd.DataFrame]` by:

1. sorting and intersecting the benchmark and every strategy index;
2. raising `ValueError("no common return dates")` when the intersection is empty;
3. calling existing `performance_metrics(..., periods_per_year=252)`;
4. calling existing `excess_summary_bps(..., periods_per_year=252)`;
5. adding `mean_active_share_pct` from the supplied summary;
6. sorting descending by `annualized_excess_bp`, then `information_ratio`.

Implement `write_comparison_outputs(...)` to write:

```text
performance_summary.csv
performance_summary.xlsx
daily_returns.csv
cumulative_returns.png
cumulative_excess_returns.png
interpretation.md
```

The Markdown report must state the shared assumptions, list all three variants, show the ranked metric table, and identify the top annualized-excess and top information-ratio variants without claiming statistical significance.

- [ ] **Step 4: Run the new unit tests and verify GREEN**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: all pure comparison tests pass.

- [ ] **Step 5: Commit only Task 2 paths with a Lore-format message**

Record that reporting reuses the existing EMP008 performance and excess-return metric functions.

### Task 3: Add reproducible EMP008 orchestration and CLI

**Files:**
- Modify: `tests/strategies/test_emp008_size_value_measure_comparison.py`
- Modify: `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`

- [ ] **Step 1: Write a failing orchestration test with fakes**

Monkeypatch a module-level `run_portfolio_variant` and `_benchmark_returns`, then assert:

```python
def test_run_size_value_measure_comparison_writes_manifest_and_shared_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")

    def fake_run_portfolio_variant(**kwargs: object) -> VariantResult:
        factor_set = FactorSetId(str(kwargs["factor_set"]))
        offset = DEFAULT_VARIANTS.index(factor_set.value) * 0.0001
        return VariantResult(
            factor_set=factor_set,
            returns=pd.Series([0.001 + offset] * 4, index=dates),
            returns_csv=tmp_path / factor_set.value / "returns.csv",
            weights_dir=tmp_path / factor_set.value / "weights",
            active_share={"mean_pct": 2.0 + offset},
        )

    monkeypatch.setattr(MODULE, "run_portfolio_variant", fake_run_portfolio_variant)
    monkeypatch.setattr(MODULE, "_benchmark_returns", lambda *args, **kwargs: pd.Series(0.0, index=dates))

    payload = run_size_value_measure_comparison(
        parquet_dir=Path("parquet"),
        output_dir=tmp_path,
        start="2024-01-02",
        end="2024-01-05",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
        variants=DEFAULT_VARIANTS,
    )

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "performance_summary.csv").exists()
    assert payload["variants"] == list(DEFAULT_VARIANTS)
```

- [ ] **Step 2: Run the orchestration test and verify RED**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py::test_run_size_value_measure_comparison_writes_manifest_and_shared_outputs -q`

Expected: failure because orchestration types/functions do not exist.

- [ ] **Step 3: Implement one real portfolio run**

Add a frozen `VariantResult` dataclass and `run_portfolio_variant(...)`. The function must:

1. build `Emp008Config` with the selected factor set, shared risk model, and tracking-error budget;
2. run `run_emp008(...)` into `<output>/<variant>/weights` unless complete cached weights exist;
3. write `target_weights.csv` and `active_share.parquet/csv`;
4. build a no-cost close-fill `TargetWeightSpec` and run `BacktestRunner` into `<output>/<variant>/backtests`;
5. persist `<output>/<variant>/backtest_metadata.json` containing factor set, serialized config, and return-series path;
6. return the daily returns and active-share summary;
7. reuse valid cached outputs unless `force=True`.

Use the existing helpers `build_emp008_config`, `write_target_weights_csv`, `build_target_weight_spec`, `write_active_share`, and `active_share_summary`; do not duplicate their formulas.

- [ ] **Step 4: Implement the three-variant orchestrator and manifest**

`run_size_value_measure_comparison(...)` must validate the variants, run each with identical settings, load `IKS200` through `_benchmark_returns(parquet_dir / "qw_BM.parquet", "IKS200")`, call the pure aggregation/output functions, and write `manifest.json` with:

```python
{
    "start": start,
    "end": end,
    "tracking_error_annual": tracking_error_annual,
    "risk_model": risk_model,
    "fill_mode": "close",
    "costs": {"fee": 0.0, "sell_tax": 0.0, "slippage": 0.0},
    "benchmark": "IKS200",
    "variants": [
        {
            "factor_set": factor_set.value,
            "factors": [factor.value for factor in get_factor_set_definition(factor_set).factors],
            "datasets": sorted(dataset.value for definition in factor_definitions_for_set(factor_set) for dataset in definition.datasets),
        }
        for factor_set in factor_sets
    ],
}
```

The CLI defaults are `parquet`, `2020-01-31`, the latest common end across all selected variants, annual TE `0.007`, `factor_idio`, and `DEFAULT_OUTPUT_DIR`. It exposes `--force` and optional `--variants`.

- [ ] **Step 5: Run the complete new test module and verify GREEN**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit only Task 3 paths with a Lore-format message**

Document the cache contract and no-cost comparison constraint.

### Task 4: Document and execute the empirical comparison

**Files:**
- Modify: `backtesting/strategies/emp008/README.md`
- Runtime output: `backtesting/strategies/emp008/tests/size_value_measure_comparison/`

- [ ] **Step 1: Add the README command and interpretation contract**

Document:

```powershell
uv run python -m backtesting.strategies.emp008.experiments.size_value_measure_comparison `
  --start 2020-01-31 `
  --tracking-error-annual 0.007 `
  --risk-model factor_idio
```

Explain the exact two factors in each variant, the separate FY0/TTM datasets, the Origin-style Size semantics, the no-cost primary comparison, and every root-level output file.

- [ ] **Step 2: Run the real comparison**

Run the command above without `--force`. If a partial cache is found, rerun with `--force` only for the affected experiment directory.

Expected: exit code 0 and all three variant directories plus the six root comparison artifacts and `manifest.json`.

- [ ] **Step 3: Inspect the empirical artifacts**

Check:

```powershell
uv run python -c "import json, pathlib, pandas as pd; p=pathlib.Path('backtesting/strategies/emp008/tests/size_value_measure_comparison'); s=pd.read_csv(p/'performance_summary.csv'); m=json.loads((p/'manifest.json').read_text(encoding='utf-8')); assert len(s)==3; assert set(s['variant'])=={'size_value_fcf_tev','size_value_dividend_fy0','size_value_dividend_ttm'}; assert s.select_dtypes('number').notna().all().all(); assert len(m['variants'])==3; print(s[['variant','annualized_excess_bp','annualized_tracking_error_bp','information_ratio','mean_active_share_pct']].to_string(index=False))"`
```

Expected: three finite rows and an interpretable ranking table.

- [ ] **Step 4: Commit documentation and lightweight empirical summaries**

Stage the README plus manifest, summary CSV/XLSX, daily returns, plots, and interpretation. Do not stage bulky per-run engine caches if repository ignore rules exclude them; retain them locally under the requested experiment directory.

### Task 5: Full verification and completion audit

**Files:**
- Verify all changed files and empirical outputs.

- [ ] **Step 1: Run focused EMP008 tests**

Run:

```powershell
uv run pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_factor_pipeline.py tests/strategies/test_emp008_factor_quantiles.py tests/strategies/test_emp008_experiments.py tests/strategies/test_emp008_size_value_measure_comparison.py -q
```

Expected: all pass.

- [ ] **Step 2: Run broader static and test verification**

Run:

```powershell
uv run ruff check backtesting/strategies/emp008 tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py
uv run pytest tests/strategies tests/scripts/test_run_emp008_full.py -q
```

Expected: zero lint errors and all selected EMP008-related tests pass. If repo-wide unrelated failures exist, isolate and report them with evidence rather than modifying unrelated user work.

- [ ] **Step 3: Audit every requested artifact**

Prove:

- Size is identical in all three registered sets;
- exactly one of FCF/TEV, FY0 dividend, or TTM dividend is paired with Size;
- all three complete real EMP008 runs use identical dates/risk/TE/cost settings;
- outputs are inside `backtesting/strategies/emp008/tests/`;
- portfolio performance is available in tabular, time-series, chart, and interpretation forms;
- no required metric is missing or non-finite.

- [ ] **Step 4: Review the final diff without disturbing unrelated edits**

Run: `git status --short` and `git diff --check`.

Expected: only intentional task files plus clearly identified pre-existing user changes; no whitespace errors in task files.
