# ETF Index Research

This package separates ETF universe collection from index-provider methodology work.

## Boundaries

- `research.py` collects the ETF universe and domestic-sector classifications.
- Provider-specific methodology pipelines live in their own subpackages.
- Do not add provider-specific crawlers or rule parsers at the `etfs/` root.

## FnGuide

`etfs/fnguide/` is the first provider-specific pipeline and acts as the reference layout for future index families.

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
python -m etfs.fnguide.validation --write-results
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

- `output/universe/all.csv`, `output/universe/sector.csv`, `output/universe/universe.json`: ETF universe
- `output/classification/families.csv`, `output/classification/families.json`, `output/classification/families.md`: broad index-product family inventory
- `output/sources/sources.csv`, `output/sources/sources.json`, `output/sources/sources.md`: methodology source candidates for each family
- `output/providers/fnguide/pdfs.csv`, `output/providers/fnguide/pdfs.json`: FnGuide methodology PDF manifest
- `output/providers/fnguide/rules.csv`, `output/providers/fnguide/rules.json`: extracted FnGuide index rules
- `output/providers/fnguide/requirements.csv`, `output/providers/fnguide/requirements.json`, `output/providers/fnguide/requirements.md`: data requirements
- `output/providers/fnguide/fnguide.csv`, `output/providers/fnguide/fnguide.json`, `output/providers/fnguide/fnguide.md`: FnGuide coverage and next actions
- `output/extractions/fnguide/methodology_extractions.json`, `output/extractions/fnguide/methodology_extractions.md`: PDF evidence extracted for methodology specs
- `output/extractions/fnguide/draft_specs.json`, `output/extractions/fnguide/methodology_specs.json`: draft and canonical methodology specs
- `output/extractions/fnguide/methodology_audit.json`, `output/extractions/fnguide/methodology_review_queue.json`: engine-readiness gate and review queue
- `output/validation/validation_fixtures.json`, `output/validation/validation_results.json`: ETF holdings validation fixtures and count/cash checks
- `output/engine/fnguide/engine_input_requirements.json`: exact bucket and field inputs required by engine-ready specs
- `output/engine/fnguide/engine_inputs.template.json`: fillable request template for producing `engine_inputs.json`
- `output/engine/fnguide/engine_support_matrix.json`, `output/engine/fnguide/engine_support_matrix.md`: engine-ready, review-only, evidence-blocked, and unsupported methodology classification
- `output/engine/fnguide/engine_promotion_candidates.json`, `output/engine/fnguide/engine_promotion_candidates.md`: specs whose methodology is engine-supported but still needs PDF evidence/status review before promotion
- `output/engine/fnguide/methodology_replication_report.json`, `output/engine/fnguide/methodology_replication_report.md`: smoke-tested target-weight replication status for engine-ready specs and explicit full-methodology replication limits
- `output/replication/fnguide/kss_data_requirements.json`: KSS full-replication data contract and currently available datasets
- `output/replication/fnguide/data_inventory.json`, `output/replication/fnguide/data_inventory.md`: provider-wide full-replication data inventory and readiness report
- `output/replication/fnguide/kss_data_inventory.json`, `output/replication/fnguide/kss_data_inventory.md`: focused SOL/KSS data inventory showing available local inputs and missing official evidence
- `output/replication/fnguide/kss_selected_buckets.json`: selected KSS top2, momentum, and market-cap-fill buckets when a source snapshot is supplied
- `output/replication/fnguide/kss_target_weights.json`: KSS target weights generated from selected buckets
- `output/replication/fnguide/kss_replication_validation.json`, `output/replication/fnguide/kss_replication_validation.md`: KSS validation diff against official targets or secondary ETF holdings
- `output/engine/fnguide/target_weights.json`: target weights, written only when `engine_inputs.json` supplies explicit bucket constituents and float-market-cap inputs
- `output/validation/target_weight_validation.json`: target weights compared with ETF holdings snapshots, written only when target weights exist
- `output/engine/fnguide/offline_pipeline_manifest.json`: reproducible offline pipeline manifest
- `output/providers/krx/krx.csv`, `output/providers/krx/krx.json`, `output/providers/krx/krx.md`: KRX methodology probe manifest
- `output/providers/spglobal/spglobal.csv`, `output/providers/spglobal/spglobal.json`, `output/providers/spglobal/spglobal.md`: S&P Global methodology probe manifest
- `output/providers/nasdaq/nasdaq.csv`, `output/providers/nasdaq/nasdaq.json`, `output/providers/nasdaq/nasdaq.md`: Nasdaq methodology probe manifest
- `output/providers/msci/msci.csv`, `output/providers/msci/msci.json`, `output/providers/msci/msci.md`: MSCI methodology probe manifest

`python -m etfs.fnguide.pipeline` reruns the offline artifact chain from existing FnGuide rule/PDF outputs through extraction, canonical specs, audit, validation, engine input requirements, a fillable engine-input template, and a methodology replication report. It deliberately skips target-weight generation unless `output/engine/fnguide/engine_inputs.json` exists, because ETF holdings files are validation evidence, not methodology calculation inputs. When target weights do exist, the pipeline also writes a strict target-vs-holdings comparison.

The replication report is intentionally scoped. It proves that engine-ready methodology specs can execute target-weight formulas from explicit calculation inputs. It does not claim full provider methodology replication until issuer universe construction, bucket selection, and official rebalance target comparisons are also available.

KSS replication is the first full-methodology tracer bullet. The pipeline always writes its data-requirement artifact and only calculates selected buckets and target weights when `output/replication/fnguide/kss_snapshot.json` supplies a dated source snapshot. Missing snapshots are reported as a skip, not as a successful replication.

Full replication work is gated by the data inventory. KSS/SOL calculation remains `missing_calculation_inputs` until the missing calculation inputs, such as sales momentum, composite score inputs, and corporate-action history, are available. Official bucket assignments, official target weights, and ETF holdings are validation evidence only: they can prove or challenge the calculated result, but they are not inputs for constructing unknown index constituents.

`families` is provider-agnostic. It separates the current FnGuide domestic-sector reference lane from future product-family lanes such as foreign/global equity, domestic broad-market, and fixed-income/cash/commodity/derivative products.

`sources` is also provider-agnostic. It assigns explicit methodology-source candidates when the ETF name includes one, such as MSCI, S&P, Nasdaq, KRX, FnGuide, iSelect, or KIS. When the name does not identify a source, it assigns a low-confidence product-family probe target so the next crawler can be planned by family instead of by individual ETF.

`etfs/krx/` is the second provider-specific package. Its first stage is a probe manifest because KRX broad-market ETFs usually identify the KRX-managed underlying index by name; the next crawler can use the manifest to query KRX index summary and constituent data.

FnGuide methodology PDFs under `etfs/raw/methodologies/` are treated as a local cache. They are intentionally not tracked because the current cache is large; `output/providers/fnguide/pdfs.json` records the source URL, local cache path, byte size, and SHA-256 digest needed to verify or recreate the cache.

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
