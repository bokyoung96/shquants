# KIND Announcement Time Recovery

This package recovers visible KIND disclosure times for the `timeframe` sheet in
`kind/announcements.xlsx` without editing the source workbook and without
guessing ambiguous rows.

## Setup

```powershell
uv sync
uv run python -m playwright install chromium
```

## Run

```powershell
uv run python -m kind.pipeline `
  --input kind/announcements.xlsx `
  --output-dir kind/outputs `
  --cache-dir kind/cache `
  --log-dir kind/logs
```

Use `--refresh` to refetch dates even when a valid cache exists. The default
client uses one Playwright request context, `--concurrency 4`, `--min-delay
0.75`, `--timeout 30`, and `--max-attempts 3`.

## Input Contract

The source workbook is read-only. The `timeframe` sheet must keep the observed
wide layout: ticker, company, disclosure type, item code, item name, aggregation
period, quarter reference dates, and YYYYMMDD announcement-date cells. The
reader fails closed if metadata changes or duplicate ticker-quarter observations
appear.

## Cache

Each date is cached under:

```text
kind/cache/YYYY-MM-DD/page-0001.html
kind/cache/YYYY-MM-DD/manifest.json
```

The manifest records the source URL, form fields, page count, parser schema
version, page names, SHA-256 hashes, and fetch timestamp. Page files are written
atomically and the manifest is written last. A corrupt or stale manifest causes a
refetch; an interrupted run does not publish a valid manifest for partial data.

## Confidence Values

- `EXACT_MATCH`: one eligible disclosure matched the workbook company after
  trimming and issuer checks.
- `NORMALIZED_MATCH`: no exact company match existed, but one eligible
  disclosure matched after conservative legal-form and punctuation
  normalization.
- `MULTIPLE_MATCH`: deterministic filters still left multiple candidates, or a
  duplicate receipt carried conflicting evidence.
- `NO_MATCH`: no eligible unambiguous disclosure was found.

When both separate and consolidated statements are present, the
`연결재무제표기준` title is preferred. Disclosures whose title identifies
`자회사의 주요경영사항` are excluded. If the announcement date has no match,
the same company is checked on each date up to three days before and after;
the audit records the selected disclosure date and day offset.

Only `EXACT_MATCH` and `NORMALIZED_MATCH` rows receive `announcement_datetime`.
Unresolved rows stay `NaT`. The CLI returns `0` only when validation is strictly
complete; otherwise it writes all artifacts and returns `2`.

## Outputs And Reports

`kind/outputs/`:

- `announcements_with_time.xlsx`
- `announcements_with_time.csv`
- `match_audit.csv`

`kind/logs/`:

- `missing_match.csv`
- `duplicate_match.csv`
- `multiple_candidate.csv`
- `schema_error.csv`
- `validation_summary.json`

`match_audit.csv` keeps receipt IDs, issuer IDs, candidate receipts/titles,
page positions, normalized company text, and rejection reasons so unresolved
rows can be inspected without re-running the crawler.

## Recovery

Rerun the same command after interruption. Valid cached dates are reused without
network requests. Use `--refresh` only when KIND pages or parser expectations
need to be revalidated. If KIND changes its HTML shape, affected pages are
reported in `schema_error.csv`; the corresponding workbook rows remain present
and unresolved rather than being dropped.
