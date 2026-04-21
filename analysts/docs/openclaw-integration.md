# OpenClaw Integration Surface for PDF Ingestion

## Goal
Keep the `analysts/` PDF-ingestion workflow callable from future OpenClaw automation without requiring OpenClaw to know repo-internal modules.

## Stable command surface

### 1) Ingest one local PDF
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli ingest-pdf --path data/raw/example.pdf
```

Expected result:
- writes processed artifacts for the PDF
- returns/report logs the report slug or message id used for artifact naming

### 2) Summarize the latest stored report for a channel
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli summarize-latest --channel DOC_POOL
```

Expected result:
- reuses the latest stored raw PDF for `DOC_POOL`
- writes summary input, summary JSON, and summary Markdown artifacts

### 3) Summarize a recent batch
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli summarize-recent --channel DOC_POOL --limit 10
```

Expected result:
- processes the most recent `N` stored reports for the channel
- returns concrete counts for processed files and summaries

## Artifact contract
For each processed report, OpenClaw should be able to read:
- `data/processed/*-fulltext.txt`
- `data/processed/*-extraction.json`
- `data/processed/*-images.json`
- `data/processed/*-chunks.json`
- `data/processed/*-summary-input.json`
- `data/processed/*-summary.json`
- `data/processed/*-summary.md`

Raw PDFs remain in `data/raw/` and are never rewritten by downstream steps.

## JSON expectations

### Extraction metadata
- extraction method
- quality/degraded reason
- page count and per-page stats
- image metadata summary

### Chunk artifact
- chunk id
- chunk order
- page range
- char/token counts
- chunk text payload

### Summary JSON
- headline
- executive summary
- key points
- risks
- confidence
- follow-up questions
- route/lane metadata

## Failure cases OpenClaw should expect
- missing raw PDF path
- unreadable or OCR-needed PDF
- no stored report found for a requested channel
- summarizer provider/config missing
- partial ingestion failure where extraction metadata exists but summary outputs do not

OpenClaw should treat the machine-readable JSON artifacts as the primary integration surface and Markdown as a human-readable companion output.
