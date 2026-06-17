# ETF Index Research

This package separates ETF universe collection from index-provider methodology work.

## Boundaries

- `research.py` collects the ETF universe and domestic-sector classifications.
- Provider-specific methodology pipelines live in their own subpackages.
- Do not add provider-specific crawlers or rule parsers at the `etfs/` root.
- Provider-neutral holdings and cap-candidate helpers live under `etfs/common/`. Keep broader methodology logic inside provider/engine boundaries until a second provider proves the abstraction is reusable.

## FnGuide

`etfs/fnguide/` is the first provider-specific pipeline and acts as the reference layout for future index families.

FnGuide currently has more files than KRX, S&P Global, Nasdaq, and MSCI because it is implemented deeper: PDF discovery, rule extraction, evidence extraction, spec promotion, audit, validation, engine artifacts, and offline orchestration are all present. That does not mean every file is inherently FnGuide-only. Keep the boundary explicit:

| File | Current role | Classification | Refactor direction |
| --- | --- | --- | --- |
| `methodology.py` | FnGuide source/PDF discovery | FnGuide-specific | Keep under `etfs/fnguide/` |
| `index_methodology.py` | FnGuide rule extraction from methodology text | FnGuide-specific | Keep under `etfs/fnguide/` |
| `methodology_extraction.py` | FnGuide PDF evidence extraction patterns | FnGuide-specific adapter | Keep provider parsing here; move only generic evidence containers if another provider needs them |
| `methodology_specs.py` | Builds canonical specs from FnGuide extraction fields | Mostly FnGuide-specific adapter | Keep here until another provider emits the same extraction field contract |
| `rules.py` | FnGuide rule dataclasses and conversion helpers | FnGuide-specific | Keep under `etfs/fnguide/` |
| `pipeline.py` | FnGuide offline artifact orchestration | FnGuide-specific orchestration | Keep here; call `etfs/common/` only for active shared primitives |
| `coverage.py` | FnGuide readiness/coverage report | Provider report over common concepts | Potential later shared report shape, but no move until a second provider needs it |
| `data_requirements.py` | FnGuide rule-to-data requirement report | Provider report over common concepts | Potential later shared report shape |
| `data_inventory.py` | Provider-wide replication data inventory | Provider report over common concepts | Potential later shared inventory renderer |
| `methodology_audit.py` | Spec readiness and blocker categorization | Mixed | Candidate for `etfs/methodology_audit.py` after another provider uses canonical specs |
| `methodology_engine.py` | Target-weight calculation plus FnGuide engine reports | Mixed; too much still lives here | Next major split: move generic target-weight engine to root, keep FnGuide report/CLI wrapper here |
| `validation.py` | Holdings workbook parsing and validation fixture writing | Mixed; not truly FnGuide-specific | Candidate for root `etfs.validation`/`etfs.holdings_io`; keep only provider-specific ingestion hooks here if needed |

Provider-neutral modules live under `etfs/common/`:

| File | Shared responsibility |
| --- | --- |
| `common/holdings.py` | Holdings fixture, snapshot, and holding models/loaders |
| `common/cap.py` | Generic cap policy/candidate reporting |

FnGuide wrappers that only re-exported shared modules were removed: `fnguide/selection.py`, `fnguide/replication.py`, and the KSS-only `fnguide/replication_data.py`. The short-lived common `selection`, `weighting`, and `replication` modules were also removed because they were not part of the current cap-candidate path. Target-weight redistribution remains local to the methodology engine, where it is only used when explicit methodology inputs are supplied. Do not add provider wrappers around `etfs/common/` modules unless they add provider-specific behavior.

Pipeline:

```powershell
python -m etfs.research
python -m etfs.families
python -m etfs.sources
python -m etfs.fnguide.methodology
python -m etfs.fnguide.index_methodology
python -m etfs.fnguide.data_requirements
python -m etfs.fnguide.coverage
python -m etfs.fnguide.methodology_extraction
python -m etfs.fnguide.methodology_specs --canonical
python -m etfs.fnguide.methodology_audit
python -m etfs.refresh.holdings_refresh --template etfs/refresh/pdf.xlsx --output-dir etfs/output/files
python -m etfs.fnguide.validation --input <holdings.xlsx> --index-map '{"<etf_code>":"<index_code>"}' --write-results
python -m etfs.common.cap
python -m etfs.fnguide.methodology_engine --write-requirements
python -m etfs.fnguide.methodology_engine --write-template
python -m etfs.fnguide.methodology_engine --write-replication-report
python -m etfs.fnguide.pipeline
python -m etfs.krx.methodology
python -m etfs.spglobal.methodology
python -m etfs.nasdaq.methodology
python -m etfs.msci.methodology
```

Outputs:

- `output/files/holdings_<ticker>.parquet`: DataGuide6-refreshed ETF holdings rows from the workbook template, stored per ETF ticker
- `output/validation/validation_fixtures.json`, `output/validation/validation_results.json`: ETF holdings validation fixtures and count/cash checks
- `output/validation/cap_candidates.json`, `output/validation/cap_candidates.md`: latest-holdings cap breach candidates with current weight, quantity, market value, cap, and excess weight
- `output/methodology/fnguide/*.json` and `*.md`: FnGuide methodology PDF manifests, extracted rules, specs, audit, data inventory, and optional target-weight diagnostics
- `output/validation/target_weight_validation.json`: target weights compared with ETF holdings snapshots, written only when target weights exist
- `output/methodology/{krx,spglobal,nasdaq,msci}/`: optional methodology probes for non-FnGuide providers

`python -m etfs.fnguide.pipeline` reruns the offline artifact chain from existing FnGuide rule/PDF outputs through extraction, canonical specs, audit, validation, engine input requirements, a fillable engine-input template, and a scoped target-weight diagnostics report. It deliberately skips target-weight generation unless `output/methodology/fnguide/engine_inputs.json` exists, because ETF holdings files are validation evidence, not methodology calculation inputs. When target weights do exist, the pipeline also writes a strict target-vs-holdings comparison.

Target-weight diagnostics are secondary to the current cap workflow and intentionally scoped. They prove only that engine-ready methodology specs can execute target-weight formulas from explicit calculation inputs. They do not claim full provider methodology replication until issuer universe construction, bucket selection, and official rebalance target comparisons are also available.

Cap and target-weight checks are provider-neutral. A specific ETF workbook, such as a SOL AI semiconductor holdings file, is a validation fixture only: it helps test parsing, current holdings, cap breaches, and target-vs-holdings drift after the generic pipeline is working. It is not a production default and does not define the pipeline's scope. The cap candidate report is intentionally holdings-first: it identifies securities already above a conservative security-level cap and carries quantity and market value forward for later market-impact analysis.

`python -m etfs.refresh.holdings_refresh` uses FnGuide methodology specs to derive ETF tickers, writes each code into the DataGuide workbook template's `B3` cell as `Axxxxxx`, follows workbook hyperlinks whose ScreenTip is `DataGuide6`, then stores refreshed `A:H` rows from row 7 onward as ticker-level parquet files under `output/files/`. The DataGuide template and ticker workbook live beside the refresh code as `etfs/refresh/pdf.xlsx` and `etfs/refresh/ticker.xlsx`. This follows FnGuide's DataGuide6 auto-refresh macro guidance, but keeps the automation in a script instead of requiring a permanent `.xlsm` macro workbook. Excel runs visible by default because DataGuide6 refresh events are more reliable that way; use `--hidden` only after confirming it works locally. Use `--tickers 0167A0 --limit 1` for a narrow live refresh test, or `--refresh-mode parse-only` to convert the current workbook contents without opening Excel.

`python -m etfs.refresh.refresh` is the project-level entry point for this workflow. It reads the visible filtered rows in `etfs/refresh/ticker.xlsx`, uses `etfs/refresh/pdf.xlsx` as the DataGuide6 template, and writes final parquet outputs to `etfs/output/files/`. `output/files/` is reserved for `holdings_<ticker>.parquet` files only. Transient Excel copies and `refresh.ps1` are created under `etfs/refresh/work/` during a run and removed after a successful run unless `--keep-work` is passed. The refresh command defaults to `--mode batch`, which opens Excel and the template once per chunk, refreshes selected tickers sequentially, and saves one workbook copy per ticker before parquet conversion. This avoids most per-ticker PowerShell/Excel/workbook startup cost while keeping DataGuide6 refresh itself sequential. Existing ticker parquet files are skipped by default for faster resumable runs; pass `--force` to refresh them again. Use `--mode single` only when diagnosing one-ticker workbook behavior.

Full index reconstruction is gated by the data inventory. Official bucket assignments, official target weights, and ETF holdings are validation evidence only: they can prove or challenge a calculated result, but they are not inputs for constructing unknown index constituents.

`families` is provider-agnostic. It separates the current FnGuide domestic-sector reference lane from future product-family lanes such as foreign/global equity, domestic broad-market, and fixed-income/cash/commodity/derivative products.

`sources` is also provider-agnostic. It assigns explicit methodology-source candidates when the ETF name includes one, such as MSCI, S&P, Nasdaq, KRX, FnGuide, iSelect, or KIS. When the name does not identify a source, it assigns a low-confidence product-family probe target so the next crawler can be planned by family instead of by individual ETF.

`etfs/krx/` is the second provider-specific package. Its first stage is a probe manifest because KRX broad-market ETFs usually identify the KRX-managed underlying index by name; the next crawler can use the manifest to query KRX index summary and constituent data.

FnGuide methodology PDFs under `etfs/output/methodologies/` are treated as a local cache. They are source methodology documents, not DataGuide holdings refresh outputs. `output/methodology/fnguide/pdfs.json` records the source URL, local cache path, byte size, and SHA-256 digest needed to verify or recreate the cache.

`etfs/spglobal/` maps S&P source candidates to S&P DJI methodology-library documents. This stage records official methodology URL candidates before any PDF downloader is added.

`etfs/nasdaq/` maps Nasdaq source candidates to Nasdaq methodology-library documents such as Nasdaq-100, Nasdaq Biotechnology, Nasdaq Clean Edge Green Energy, and Dividend Achievers.

`etfs/msci/` maps MSCI source candidates to MSCI methodology-library documents such as the Global Investable Market Indexes, Universal, ESG Leaders, and US REIT methodology documents. This is the last provider added in the current expansion pass.

## Adding Another Provider

Create a new provider package under `etfs/<provider>/` and keep its public stages aligned with the FnGuide shape:

- `methodology.py`: discover or download source methodology documents
- `index_methodology.py`: extract provider-specific rule profiles
- `rules.py`: typed rule objects for that provider
- `data_requirements.py`: map rules to required calculation data
- `coverage.py`: summarize current readiness and blockers

Provider packages may use different extraction logic because each index provider publishes methodology documents differently. Shared abstractions should be introduced only after two provider packages need the same behavior.

