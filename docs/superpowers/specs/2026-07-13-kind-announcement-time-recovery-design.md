# KIND Announcement Time Recovery Design

## Goal

Build a reproducible pipeline under `kind/` that reads the `timeframe` sheet in
`kind/announcements.xlsx`, retrieves the corresponding KIND disclosure pages,
and adds the actual disclosure time without guessing. The pipeline must retain
every input announcement-date row, distinguish certain matches from ambiguous
or missing matches, and emit auditable warning reports.

The source workbook is immutable input. The pipeline writes new long-format
Excel and CSV artifacts.

## Observed Input Contract

`kind/announcements.xlsx` contains visible `value` and `timeframe` sheets. Both
are 41 rows by 648 columns and use the same wide matrix layout:

- Rows 1-8 contain workbook metadata and settings.
- Row 9 contains 647 ticker codes, with an `A` prefix.
- Row 10 contains company names.
- Rows 11-14 contain statement type, item code, item name, and aggregation
  period.
- Rows 15-41 contain reference quarter dates in column A and per-company values
  across columns B:XX.
- The `timeframe` values are announcement dates encoded as `YYYYMMDD` integers.

There are 11,495 non-null timeframe observations over 720 distinct announcement
dates from 2020-04-07 through 2026-07-08. Company names are not unique and one
ticker contains a letter, so ticker parsing must preserve arbitrary text after
the leading `A` rather than enforce six numeric digits.

The normalized input row is:

```text
ticker, company, quarter, announcement_date
```

`quarter` is derived from the calendar quarter of the reference date in column
A, not from the announcement date.

## Chosen Collection Approach

Use Playwright's asynchronous request context. The client first visits the KIND
main page to establish the same session boundary as the browser, then submits
the observed `searchTodayDisclosureSub` form for each announcement date. Results
are requested in 100-row pages and pagination is followed until the final page.

This approach was selected over full Chromium UI navigation because KIND mixes
form submissions and page replacement in a way that produces navigation-event
timeouts across hundreds of dates. It was selected over a bare HTTP client
because the requested implementation explicitly prefers Playwright and the
initial browser-compatible session boundary is useful if KIND changes its
session behavior.

Every successful response is cached before parsing:

```text
kind/cache/YYYY-MM-DD/page-0001.html
kind/cache/YYYY-MM-DD/manifest.json
```

The manifest records the source URL, form fields that affect the response,
fetch timestamp, page count, response hashes, and parser schema version. A
normal rerun reads valid cached pages. `--refresh` refetches them. Cache writes
use a temporary file followed by an atomic replace so interrupted runs do not
leave valid-looking partial pages.

## Components

### Workbook reader

`kind/workbook.py` validates the workbook layout and converts the timeframe
matrix into a deterministically sorted long DataFrame. It rejects missing
sheets, changed header labels, mismatched ticker/name axes, duplicate normalized
input keys, invalid announcement dates, and unsupported layout changes.

### KIND client

`kind/client.py` owns Playwright, form submission, pagination, retry, timeout,
rate limiting, and the date cache. Independent dates may run concurrently, but
all outgoing requests pass through one global rate limiter. Defaults are two
concurrent dates, a 750 ms minimum interval between requests, a 30-second
request timeout, and three attempts with bounded exponential backoff.

### Selector and response schema

`kind/selectors.py` contains the endpoint, form-field defaults, expected table
structure, title patterns, and attribute patterns. Network and parser code do
not embed selectors or title strings elsewhere.

`kind/parser.py` parses cached HTML with BeautifulSoup's built-in `html.parser`.
It extracts time, displayed company name, disclosure title, submitter, KIND
issuer identifier, receipt identifier, and page position. It validates the
five-column disclosure table, `HH:MM` time format, pagination metadata, company
link, and disclosure link. A schema mismatch fails closed for the affected date
and is logged; the parser never infers fields from positional text outside the
validated disclosure row.

### Candidate filtering and matching

`kind/matching.py` first restricts disclosures to titles matching the provisional
operating-results family, including connected-statement variants. Forecast
disclosures are excluded. Correction disclosures and subsidiary-major-event
disclosures remain visible to the audit layer but are lower-priority candidates
than direct, non-correction disclosures.

Company matching has two stages:

1. `EXACT_MATCH`: trimmed source and KIND company names are identical.
2. `NORMALIZED_MATCH`: Unicode NFKC normalization, case folding, whitespace
   removal, legal-form removal such as `(주)` and `주식회사`, and punctuation
   removal produce identical non-empty names.

When KIND exposes an issuer identifier, it is used as a safety guard against
historical duplicate company names. A conflicting issuer identifier invalidates
the candidate; it is never ignored merely because the names match.

Candidate selection is fail-closed:

- A single eligible candidate produces its actual KIND time.
- An original direct disclosure supersedes a matching correction or subsidiary
  disclosure only when that leaves exactly one candidate.
- Two or more plausible candidates with different receipt identifiers remain
  `MULTIPLE_MATCH`; no time is written.
- No plausible candidate is `NO_MATCH`; no time is written.
- No nearest-name, nearest-time, value-comparison, or inferred-time heuristic is
  permitted.

The confidence vocabulary is exactly:

```text
EXACT_MATCH
NORMALIZED_MATCH
MULTIPLE_MATCH
NO_MATCH
```

### Validation and reporting

`kind/validation.py` runs validation before outputs are finalized:

1. Normalized input row count equals final result row count.
2. Every row has ticker, company, quarter, and announcement date. Matched rows
   have a valid `announcement_datetime`; unmatched rows retain `NaT` and are
   reported rather than dropped or filled.
3. More than one discovered time for the same ticker and quarter is reported.
4. Every row has one of the four confidence values, and datetime presence is
   consistent with the confidence.
5. Every warning is written under `kind/logs/`.

Required reports are:

```text
kind/logs/missing_match.csv
kind/logs/duplicate_match.csv
kind/logs/multiple_candidate.csv
kind/logs/schema_error.csv
kind/logs/run.log
kind/logs/validation_summary.json
```

The pipeline writes outputs even when unmatched rows exist so the audit is
inspectable, but the CLI exits non-zero when strict validation finds missing or
ambiguous times. A non-zero exit never causes guessed data to be substituted.

## Output Contract

The final artifacts are:

```text
kind/outputs/announcements_with_time.xlsx
kind/outputs/announcements_with_time.csv
kind/outputs/match_audit.csv
```

The primary dataset contains, in this order:

```text
ticker
company
quarter
announcement_date
announcement_datetime
confidence
```

`announcement_date` and `announcement_datetime` are pandas datetime columns in
memory and typed Excel datetimes in the workbook. CSV values use ISO formats.
Rows are sorted by quarter, ticker, and announcement date so identical input,
cache, and parser versions produce byte-stable CSV content. `match_audit.csv`
contains the selected or rejected receipt identifiers, titles, displayed names,
normalization results, cache pages, and rejection reasons without cluttering the
primary dataset.

## Command Surface

The checkout-local command is:

```powershell
uv run python -m kind.pipeline `
  --input kind/announcements.xlsx `
  --output-dir kind/outputs `
  --cache-dir kind/cache `
  --log-dir kind/logs
```

Optional operational flags are limited to `--refresh`, `--concurrency`,
`--min-delay`, `--timeout`, and `--max-attempts`. Defaults favor accuracy and
polite request volume over speed.

`kind/README.md` documents setup, Chromium installation, command usage,
component boundaries, cache semantics, confidence meanings, validation behavior,
and recovery from interrupted runs or KIND schema changes.

## Test Strategy

Development follows red-green-refactor. Deterministic tests live under
`tests/kind/` and cover:

- Wide workbook validation and wide-to-long conversion.
- Date and ticker preservation, including alphanumeric tickers.
- Exact and normalized Korean/English company-name matching.
- Correction, subsidiary, duplicate, and multiple-candidate handling.
- HTML parsing from minimal representative KIND fixtures.
- Pagination, retry, timeout, cache hit, refresh, and atomic-cache behavior using
  an injected fake transport rather than the live site.
- Row-count, missing-time, duplicate-time, confidence, and log validations.
- End-to-end output generation from a small workbook and cached HTML fixture.

A live smoke check then retrieves bounded historical dates through Playwright and
compares parsed rows with the visible KIND table. Live results are evidence for
the integration boundary, not a substitute for deterministic tests.

## Repository Changes

The `kind` package is added to setuptools package discovery. `beautifulsoup4` is
added as the only new dependency because the user explicitly allowed
BeautifulSoup when needed and it permits deterministic parsing of cached HTML
without launching a new browser for every page. Generated `kind/cache/`,
`kind/logs/`, and `kind/outputs/` directories and Excel owner files are ignored;
source, tests, documentation, and the original input workbook remain visible to
Git.

## Completion Criteria

The work is complete only when:

- The pipeline and README exist under `kind/` and execute from the repository
  checkout.
- Unit and integration tests, syntax checks, and project tests pass.
- A live Playwright smoke check proves the current KIND boundary and selectors.
- The full 11,495-row input is processed with equal output row count.
- Excel and CSV outputs contain typed datetime data and one confidence per row.
- Every missing, duplicate, multiple, or schema warning is materialized under
  `kind/logs/`.
- No output time came from inference or an unresolved candidate set.
- A second cached run produces the same result and does not refetch valid dates.
