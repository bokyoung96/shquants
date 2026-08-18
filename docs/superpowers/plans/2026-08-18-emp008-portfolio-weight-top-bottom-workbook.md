# EMP008 Portfolio-Weight Top/Bottom Workbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one formatted Excel workbook containing target-weight and Active-weight Top/Bottom rankings for all WI26/WICS strategies and rebalance dates.

**Architecture:** Read the two existing Active Top/Bottom workbooks plus each candidate's target and active weight files, normalize them into five long-form ranking tables, then author a standalone `.xlsx` with `@oai/artifact-tool`. Keep all calculations typed and auditable, and export only the final workbook under `outputs/emp008_portfolio_weight_top_bottom/`.

**Tech Stack:** Node.js, `@oai/artifact-tool`, Python/pandas for read-only source extraction, Excel `.xlsx`

---

### Task 1: Extract and validate ranking data

**Files:**
- Create: `.tmp/emp008_portfolio_weight_top_bottom/extract.py`
- Create: `.tmp/emp008_portfolio_weight_top_bottom/data.json`

- [ ] Read the WI26/WICS candidate lists and existing Active Top/Bottom sheets.
- [ ] Read every `target_weights.csv` and `active_weights.parquet`, align dates and tickers, and calculate `BM weight = target weight - active weight`.
- [ ] Produce deterministic target Top 10, literal Bottom 10, and material Bottom 10 using ticker ascending as the tie-breaker.
- [ ] Validate 14,220 rows per ranking table, ten rows per classification/date/candidate group, exact Active ranking agreement with the existing workbooks, and the material-weight threshold.
- [ ] Write one UTF-8 JSON intermediate containing metadata, candidates, and all five ranking tables.

Run: `C:\Users\CHECK\anaconda3\python.exe .tmp\emp008_portfolio_weight_top_bottom\extract.py`

Expected: prints five row counts of `14220`, zero Active mismatches, and zero material-threshold violations.

### Task 2: Build the formatted workbook

**Files:**
- Create: `.tmp/emp008_portfolio_weight_top_bottom/build.mjs`
- Create: `outputs/emp008_portfolio_weight_top_bottom/emp008_weight_top_bottom_wi26_wics.xlsx`

- [ ] Load the workspace-provided spreadsheet dependencies and create a temporary dependency junction.
- [ ] Run `mark_artifact_operation_started.mjs` exactly once for one `.xlsx` create operation.
- [ ] Create the seven approved sheets with typed dates and numeric percentages.
- [ ] Apply dark navy headers, classification fills, percentage/date formats, filters, frozen headers, bounded widths, and Top/Bottom color accents.
- [ ] Export exactly one workbook to the requested output directory.

Run: `node .tmp\emp008_portfolio_weight_top_bottom\build.mjs`

Expected: one non-empty `.xlsx` file and no support artifact exported beside it.

### Task 3: Verify values and layout

**Files:**
- Verify: `outputs/emp008_portfolio_weight_top_bottom/emp008_weight_top_bottom_wi26_wics.xlsx`

- [ ] Inspect the guide, candidate list, and representative data ranges for values and formulas.
- [ ] Scan the workbook for Excel formula errors.
- [ ] Render every sheet or representative ranges from every sheet and inspect for clipped headers, unreadable text, empty tables, and broken filters.
- [ ] Reconcile workbook row counts, group sizes, Active rankings, and `BM + Active = target` within tolerance.
- [ ] Export once more only if a material layout defect requires repair.

Expected: seven populated sheets, five ranking sheets with 14,220 rows each, no formula errors, and a legible visual render.
