# PDF Ingestion Review Notes

## Scope reviewed
- `README.md`
- `src/analysts/extraction.py`
- `src/analysts/pipeline.py`
- `src/analysts/cli.py`
- `.omx/plans/2026-04-15-pdf-ingestion-agent-plan.md`
- `.omx/plans/prd-pdf-ingestion-agent.md`
- `.omx/plans/test-spec-pdf-ingestion-agent.md`

## Current-state gap analysis
- **Extraction is still summary-packet first.** `SummaryReadyExtractor` writes `*-raw-text.txt` and `*-summary-input.json`, but it does not yet emit page-level extraction metadata, image metadata, chunks, or embeddings-ready records.
- **Pipeline orchestration is single-report focused.** `ArasPipeline.summarize_report()` writes summary artifacts directly after packet creation. There is no dedicated `pdf_ingest` orchestration layer yet.
- **CLI coverage is partial for the target surface.** `summarize-latest` exists today, but `ingest-pdf` and `summarize-recent` still need to be added before the planned interface is complete.
- **Traceability stops at the excerpt.** Current summarizer inputs are capped by `max_input_chars` and do not preserve chunk ids, page ranges, or image references that a richer PDF flow will need.

## Review decisions
1. **Keep raw PDFs authoritative.** Every derived artifact should be reproducible from `data/raw/*.pdf` without mutating the original download.
2. **Prefer stable per-report filenames.** The ingestion path should keep deterministic report slugs so `fulltext`, `extraction`, `chunks`, and `summary` artifacts are easy to inspect side-by-side.
3. **Chunk before summarization.** Route-aware summary agents should receive chunk selections plus metadata instead of a blind whole-document prompt.
4. **Expose a command-oriented interface.** OpenClaw should call stable CLI commands and read JSON/Markdown outputs instead of reaching into repo internals.
5. **Make degraded extraction explicit.** If OCR is still unavailable, artifact JSON should say so clearly rather than silently falling back to title/caption text.

## Merge checklist for the ingestion lanes
- `pdf_text` produces `*-fulltext.txt` plus extraction-quality metadata
- `pdf_images` records image/page metadata in a machine-readable artifact
- chunking output includes stable chunk ids, order, and page ranges
- embeddings-ready output is present even when embeddings stay deferred
- summary input records reference chunk ids instead of a lone excerpt
- `ingest-pdf`, `summarize-latest`, and `summarize-recent` share one stable artifact layout
- README examples match the final CLI behavior exactly
- end-to-end verification proves all expected processed artifacts are written

## Recommended verification after merge
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/pytest tests/test_pdf_text.py tests/test_pdf_images.py tests/test_page_selection.py tests/test_chunking.py tests/test_embeddings.py tests/test_pipeline.py -q
PYTHONPATH=src ../.venv/bin/pytest tests -q
PYTHONPATH=src ../.venv/bin/python -m analysts.cli summarize-latest --channel DOC_POOL
PYTHONPATH=src ../.venv/bin/python -m analysts.cli summarize-recent --channel DOC_POOL --limit 10
```
