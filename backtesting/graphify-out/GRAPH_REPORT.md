# Graph Report - backtesting  (2026-04-22)

## Corpus Check
- Corpus is ~15,195 words - fits in a single context window. You may not need a graph.

## Summary
- 448 nodes · 1128 edges · 14 communities detected
- Extraction: 63% EXTRACTED · 37% INFERRED · 0% AMBIGUOUS · INFERRED: 416 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Strategy Policies|Strategy Policies]]
- [[_COMMUNITY_Report Figures|Report Figures]]
- [[_COMMUNITY_Data Ingest|Data Ingest]]
- [[_COMMUNITY_Engine Validation|Engine Validation]]
- [[_COMMUNITY_Performance Snapshots|Performance Snapshots]]
- [[_COMMUNITY_HTML Composition|HTML Composition]]
- [[_COMMUNITY_Strategy Foundations|Strategy Foundations]]
- [[_COMMUNITY_Run Orchestration|Run Orchestration]]
- [[_COMMUNITY_Report CLI|Report CLI]]
- [[_COMMUNITY_Dataset Catalog|Dataset Catalog]]
- [[_COMMUNITY_Run Writer|Run Writer]]
- [[_COMMUNITY_Tearsheet Tables|Tearsheet Tables]]
- [[_COMMUNITY_Comparison Tables|Comparison Tables]]
- [[_COMMUNITY_Shared Types|Shared Types]]

## God Nodes (most connected - your core abstractions)
1. `Public validation exports.` - 41 edges
2. `ReportBuilder` - 24 edges
3. `PerformanceSnapshotFactory` - 24 edges
4. `SignalBundle` - 22 edges
5. `DataCatalog` - 20 edges
6. `ConstructionResult` - 20 edges
7. `PositionPlan` - 20 edges
8. `PassThroughPolicy` - 16 edges
9. `SectorRepository` - 16 edges
10. `HtmlRenderer` - 16 edges

## Surprising Connections (you probably didn't know these)
- `RunConfig` --uses--> `DataCatalog`  [INFERRED]
  backtesting/run.py → backtesting/catalog/catalog.py
- `RunConfig` --uses--> `PositionPlan`  [INFERRED]
  backtesting/run.py → backtesting/policy/base.py
- `RunReport` --uses--> `DataCatalog`  [INFERRED]
  backtesting/run.py → backtesting/catalog/catalog.py
- `RunReport` --uses--> `PositionPlan`  [INFERRED]
  backtesting/run.py → backtesting/policy/base.py
- `RunWriter` --uses--> `RunReport`  [INFERRED]
  backtesting/reporting/writer.py → backtesting/run.py

## Communities

### Community 0 - "Strategy Policies"
Cohesion: 0.06
Nodes (29): ABC, build_signal(), ConstructionResult, PositionPlan, PositionPolicy, RegisteredStrategy, SignalBundle, _Breakout52WeekConstructionRule (+21 more)

### Community 1 - "Report Figures"
Cohesion: 0.07
Nodes (26): _build_notes(), ReportBuilder, _write_legacy_table(), _write_tables(), ComparisonFigureBuilder, _largest_holding(), _line(), _line() (+18 more)

### Community 2 - "Data Ingest"
Cohesion: 0.09
Nodes (23): default(), default_repositories_for_universe(), from_frame(), from_historical_excel(), _load_default_frame(), _load_display_name_maps(), _normalize_symbol_key(), _read_historical_sector_frame() (+15 more)

### Community 3 - "Engine Validation"
Cohesion: 0.08
Nodes (23): BacktestEngine, _normalize_quantity(), _schedule(), _tradable(), CostModel, TradeCost, fill_prices(), Public validation exports. (+15 more)

### Community 4 - "Performance Snapshots"
Cohesion: 0.11
Nodes (21): annualized_sharpe(), build_monthly_heatmap(), build_return_distribution(), build_yearly_excess_returns(), DrawdownStats, ExposureSnapshot, monthly_return_series(), PerformanceMetrics (+13 more)

### Community 5 - "HTML Composition"
Cohesion: 0.15
Nodes (26): _comparison_metric_strip(), ComparisonComposer, ComparisonRenderContext, CoverContext, _format_metric_value(), _format_table_cell(), _format_value(), _is_internal_column() (+18 more)

### Community 6 - "Strategy Foundations"
Cohesion: 0.09
Nodes (13): BaseStrategy, CrossSectionalStrategy, TimeSeriesStrategy, validate_positive(), RankLongOnly, RankLongShort, CrossSectionalStrategy, LongOnlyTopN (+5 more)

### Community 7 - "Run Orchestration"
Cohesion: 0.11
Nodes (15): summarize_perf(), list_strategies(), BacktestRunner, _build_parser(), main(), _resolve_effective_config(), _resolve_load_start(), RunConfig (+7 more)

### Community 8 - "Report CLI"
Cohesion: 0.15
Nodes (17): main(), ReportArgumentParser, ReportCli, _validate_report_args(), BenchmarkConfig, default_kospi200(), ReportKind, ReportSpec (+9 more)

### Community 9 - "Dataset Catalog"
Cohesion: 0.18
Nodes (10): BenchmarkRepository, BenchmarkSeries, DataCatalog, default(), _spec(), Enum, DatasetGroup, DatasetId (+2 more)

### Community 10 - "Run Writer"
Cohesion: 0.33
Nodes (8): _bucket_ledger(), _drawdown(), _latest_qty(), _latest_weights(), _monthly_returns(), _plot_series(), RunWriter, _write_json()

### Community 11 - "Tearsheet Tables"
Cohesion: 0.36
Nodes (9): build_drawdown_episodes_table(), build_performance_summary_table(), build_sector_weights_table(), build_top_holdings_table(), build_validation_appendix_table(), _metric_label(), _metric_order(), _ordered_columns() (+1 more)

### Community 12 - "Comparison Tables"
Cohesion: 0.54
Nodes (6): build_benchmark_relative_table(), build_holdings_turnover_table(), build_ranked_summary_table(), build_sector_comparison_table(), ComparisonTableBuilder, _ordered_columns()

### Community 13 - "Shared Types"
Cohesion: 1.0
Nodes (1): Core shared type definitions for the backtesting package.

## Knowledge Gaps
- **1 isolated node(s):** `Core shared type definitions for the backtesting package.`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Shared Types`** (2 nodes): `types.py`, `Core shared type definitions for the backtesting package.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Public validation exports.` connect `Engine Validation` to `Strategy Policies`, `Dataset Catalog`, `Strategy Foundations`, `Run Orchestration`?**
  _High betweenness centrality (0.301) - this node is a cross-community bridge._
- **Why does `DataCatalog` connect `Dataset Catalog` to `Data Ingest`, `Engine Validation`, `Performance Snapshots`, `Run Orchestration`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `PositionPlan` connect `Strategy Policies` to `Engine Validation`, `Run Orchestration`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 32 inferred relationships involving `Public validation exports.` (e.g. with `DataCatalog` and `DatasetGroup`) actually correct?**
  _`Public validation exports.` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `str` (e.g. with `main()` and `find_raw_path()`) actually correct?**
  _`str` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `ReportBuilder` (e.g. with `BenchmarkRepository` and `SectorRepository`) actually correct?**
  _`ReportBuilder` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `PerformanceSnapshotFactory` (e.g. with `ReportBuilder` and `DrawdownStats`) actually correct?**
  _`PerformanceSnapshotFactory` has 12 INFERRED edges - model-reasoned connections that need verification._