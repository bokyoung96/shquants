# Graph Report - backtesting  (2026-04-23)

## Corpus Check
- 75 files · ~60,851 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 499 nodes · 1287 edges · 14 communities detected
- Extraction: 63% EXTRACTED · 37% INFERRED · 0% AMBIGUOUS · INFERRED: 472 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 14|Community 14]]

## God Nodes (most connected - your core abstractions)
1. `Public validation exports.` - 41 edges
2. `PerformanceSnapshotFactory` - 27 edges
3. `SignalBundle` - 25 edges
4. `ReportBuilder` - 24 edges
5. `DataCatalog` - 20 edges
6. `ConstructionResult` - 20 edges
7. `PositionPlan` - 20 edges
8. `TearsheetFigureBuilder` - 20 edges
9. `ComposableStrategy` - 18 edges
10. `PerformanceSnapshot` - 17 edges

## Surprising Connections (you probably didn't know these)
- `RunConfig` --uses--> `DataCatalog`  [INFERRED]
  backtesting/run.py → backtesting/catalog/catalog.py
- `RunConfig` --uses--> `PositionPlan`  [INFERRED]
  backtesting/run.py → backtesting/policy/base.py
- `RunReport` --uses--> `DataCatalog`  [INFERRED]
  backtesting/run.py → backtesting/catalog/catalog.py
- `RunReport` --uses--> `PositionPlan`  [INFERRED]
  backtesting/run.py → backtesting/policy/base.py
- `RunReport` --uses--> `RunWriter`  [INFERRED]
  backtesting/run.py → backtesting/reporting/writer.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (35): ABC, build_signal(), ConstructionResult, PositionPlan, PositionPolicy, RegisteredStrategy, SignalBundle, Breakout52WeekNearnessSignalProducer (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (40): annualized_downside_deviation(), annualized_sharpe(), annualized_volatility(), build_monthly_heatmap(), build_return_distribution(), build_yearly_excess_returns(), capture_ratio(), conditional_value_at_risk() (+32 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (28): BaseStrategy, CrossSectionalStrategy, TimeSeriesStrategy, validate_positive(), BacktestEngine, _normalize_quantity(), _schedule(), _tradable() (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (32): default(), default_repositories_for_universe(), from_frame(), from_historical_excel(), _load_default_frame(), _load_display_name_maps(), _normalize_symbol_key(), _read_historical_sector_frame() (+24 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (28): _benchmark_label(), _comparison_metric_strip(), ComparisonComposer, ComparisonRenderContext, CoverContext, _format_metric_value(), _format_table_cell(), _format_value() (+20 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (18): _annualized_metric_note(), _compact_number(), _cumulative_returns(), _drawdown(), _format_date_axis(), _monthly_heatmap(), _monthly_returns(), _safe_label() (+10 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (15): summarize_perf(), list_strategies(), BacktestRunner, _build_parser(), main(), _resolve_effective_config(), _resolve_load_start(), RunConfig (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (17): _build_notes(), ReportBuilder, _write_tables(), _comparison_summary_lines(), ComparisonFigureBuilder, _cumulative_returns(), _largest_holding(), _line() (+9 more)

### Community 8 - "Community 8"
Cohesion: 0.16
Nodes (17): main(), ReportArgumentParser, ReportCli, _validate_report_args(), BenchmarkConfig, default_kospi200(), ReportKind, ReportProfile (+9 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (13): _write_legacy_table(), ReportBundle, _line_trace(), _monthly_heatmap_trace(), _monthly_returns(), PlotLibrary, _vertical_spacing(), build_appendix_table() (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.28
Nodes (8): DataCatalog, default(), _spec(), Enum, DatasetGroup, DatasetId, DatasetGroups, DatasetSpec

### Community 11 - "Community 11"
Cohesion: 0.36
Nodes (9): build_drawdown_episodes_table(), build_performance_summary_table(), build_sector_weights_table(), build_top_holdings_table(), build_validation_appendix_table(), _metric_label(), _metric_order(), _ordered_columns() (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.53
Nodes (4): _covers_index(), _has_sparse_row(), _unique_sorted(), ValidationSession

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): Core shared type definitions for the backtesting package.

## Knowledge Gaps
- **1 isolated node(s):** `Core shared type definitions for the backtesting package.`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 14`** (2 nodes): `types.py`, `Core shared type definitions for the backtesting package.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Public validation exports.` connect `Community 2` to `Community 0`, `Community 6`, `Community 10`, `Community 12`, `Community 13`?**
  _High betweenness centrality (0.273) - this node is a cross-community bridge._
- **Why does `DataCatalog` connect `Community 10` to `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `PositionPlan` connect `Community 0` to `Community 2`, `Community 6`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 40 inferred relationships involving `str` (e.g. with `main()` and `find_raw_path()`) actually correct?**
  _`str` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `Public validation exports.` (e.g. with `DataCatalog` and `DatasetGroup`) actually correct?**
  _`Public validation exports.` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `PerformanceSnapshotFactory` (e.g. with `ReportBuilder` and `DrawdownStats`) actually correct?**
  _`PerformanceSnapshotFactory` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `SignalBundle` (e.g. with `LongOnlyTopN` and `LongShortTopBottom`) actually correct?**
  _`SignalBundle` has 24 INFERRED edges - model-reasoned connections that need verification._