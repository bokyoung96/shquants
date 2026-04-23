# Backtesting Dashboard Reporting Design

## Summary

Expand the existing `backtesting/reporting` pipeline into a dashboard-first reporting toolkit that exports static post-backtest pictures and supporting tables/pages. The system should share one analytics core across notebook/research usage and saved report output.

## Hard Constraints

- This is **not** a web UI project.
- Main deliverable is a static exported dashboard picture after backtesting.
- Benchmark is optional by default.
- Reports must degrade gracefully when benchmark data is absent.
- Support three report profiles:
  - `alpha`
  - `index`
  - `absolute`
- Default behavior should auto-detect a sensible profile and allow user override.
- Keep scope inside `backtesting/reporting` unless a minimal supporting edit is unavoidable.

## Architecture

### User-facing surface

Add a thin facade class that gives a simple entry point for post-backtest analysis/report generation. Use short parameter names and class-friendly methods.

### Internal layers

1. **Analytics core**
   - Computes scalar metrics and plot-ready series.
   - Shared by notebook/research users and report builders.
2. **Profile/mode resolution**
   - Decides `alpha` / `index` / `absolute`.
   - Auto-detects by default, allows override.
3. **Snapshot assembly**
   - Builds one enriched render-ready snapshot from a `SavedRun`.
4. **Dashboard rendering**
   - Exports static figures/pages, especially a dashboard-first executive image.
5. **Tables/comparison reports**
   - Mirrors the same analytics/profile logic instead of recomputing independently.

## Dashboard Shape

The default single-run report should lead with an executive dashboard containing:

1. Performance overview
   - cumulative return
   - cumulative return vs benchmark when benchmark exists
2. Drawdown / recovery
   - underwater curve
   - max drawdown / duration / recovery stats
3. Key metrics panel
   - core metrics always
   - benchmark-relative metrics only when benchmark exists
   - tracking-oriented emphasis for index mode
4. Rolling diagnostics
   - rolling Sharpe
   - rolling volatility
   - rolling beta / alpha / information ratio / correlation when benchmark exists
   - benchmark-free replacements when benchmark does not exist
5. Calendar / distribution / exposure
   - monthly heatmap
   - yearly returns
   - daily and monthly return distributions
   - turnover / holdings count / exposure summary

## Metric Model

### Core metrics

Always available:

- cumulative return
- CAGR
- annual volatility
- Sharpe
- Sortino
- Calmar
- max drawdown
- final equity
- avg turnover
- downside deviation
- VaR / CVaR
- win rate
- payoff ratio
- profit factor
- skew / kurtosis
- best/worst day
- best/worst month
- best/worst year
- longest drawdown duration
- recovery time
- month/year hit ratios

### Benchmark-relative metrics

Only when benchmark exists and overlaps:

- alpha
- beta
- tracking error
- information ratio
- correlation
- upside/downside capture
- active return / active risk
- yearly excess return
- rolling benchmark-relative diagnostics

### Tracking/index emphasis

For `index` profile, prioritize:

- tracking error
- active return
- active risk
- correlation
- beta
- rolling tracking diagnostics

De-emphasize alpha-first storytelling for index-like strategies.

## Rendering Rules

### With benchmark

- Show benchmark-relative panels/metrics.
- Use benchmark-aware rolling diagnostics.

### Without benchmark

- Do not synthesize fake zero benchmark metrics.
- Replace benchmark panels with benchmark-free diagnostics.
- Keep the dashboard visually full and intentional.

### Index profile

- Show strategy vs benchmark.
- Emphasize tracking quality over alpha.

## Code Organization

Primary files to expand:

- `backtesting/reporting/analytics.py`
- `backtesting/reporting/models.py`
- `backtesting/reporting/snapshots.py`
- `backtesting/reporting/figures.py`
- `backtesting/reporting/comparison_figures.py`
- `backtesting/reporting/plots.py`
- `backtesting/reporting/tables_single.py`
- `backtesting/reporting/tables_comparison.py`
- `backtesting/reporting/builder.py`

Testing surface:

- `tests/reporting/*`

## Acceptance Criteria

1. Single-run reports export a richer executive dashboard picture.
2. New metrics appear in analytics/tables without requiring a benchmark.
3. Benchmark-relative metrics are omitted cleanly when benchmark is missing.
4. Index-like reports emphasize tracking metrics.
5. Existing report generation flow still works for benchmark-backed runs.
6. Reporting tests pass after implementation.
