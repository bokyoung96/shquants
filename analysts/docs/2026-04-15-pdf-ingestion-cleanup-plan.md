# PDF Ingestion Cleanup / Refactor Plan

## Why this pass exists
After the new PDF ingestion pipeline lands, the code should be restructured for clarity so extraction, chunking, embeddings, and summarization each live behind focused boundaries rather than accreting into `pipeline.py` or a few oversized modules.

## Refactor goals
- Keep orchestration thin.
- Move PDF-specific logic out of generic pipeline glue.
- Separate pure transforms from side-effecting I/O.
- Keep CLI wiring shallow and command-focused.
- Preserve behavior with tests before any cleanup edits.

## Intended boundaries
- `pdf_ingest.py` — top-level orchestration only
- `pdf_text.py` — text extraction adapters and quality scoring
- `pdf_images.py` — image metadata extraction
- `chunking.py` — deterministic chunk creation only
- `embeddings.py` — embeddings-ready payload generation / persistence helpers
- `summarizer.py` — route-aware lane planning, prompt assembly, and output normalization
- `summary_outputs.py` — persisted JSON/Markdown summary artifacts
- `pipeline.py` — fetch + delegate orchestration only
- `cli.py` — command parsing and reporting only

## Cleanup constraints
- No cleanup pass before regression tests are green.
- One smell-focused pass at a time.
- Prefer moving/deleting code over adding extra abstraction layers.
- Keep raw artifact contracts unchanged unless explicitly re-tested.

## Smells to remove after feature lands
1. Summary artifact writing mixed into high-level pipeline flow.
2. Extraction fallback logic spread across parser + extraction modules.
3. Route/lane selection mixed with prompt-building concerns.
4. CLI output formatting duplicated across commands.
5. Storage helpers growing beyond report/state persistence concerns.

## Required pre-cleanup verification
- `cd analysts && PYTHONPATH=src ../.venv/bin/pytest tests -q`
- smoke one stored/live report
- smoke one recent-N batch

## Required post-cleanup verification
- same full test suite
- same live/stored smoke checks
- manual inspection of one fulltext/chunks/summary artifact set
