# KIND Announcement Time Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the actual KIND disclosure time for every non-null `timeframe` observation without guessing, while preserving the source workbook and producing auditable Excel, CSV, cache, and warning artifacts.

**Architecture:** Normalize the wide workbook into one row per ticker-quarter, fetch each distinct announcement date through an asynchronous Playwright request context, cache raw HTML pages atomically, parse validated disclosure rows, and match only deterministic company/title candidates. Ambiguous or missing candidates remain `NaT` with `MULTIPLE_MATCH` or `NO_MATCH`, and strict validation reports every unresolved row.

**Tech Stack:** Python 3.11+, pandas, Playwright async API, BeautifulSoup with `html.parser`, openpyxl, pytest, asyncio.

---

## File Map

- Modify `pyproject.toml`: add `beautifulsoup4` and include `kind*` in package discovery.
- Modify `.gitignore`: ignore generated KIND cache/log/output and Excel owner files.
- Create `kind/__init__.py`: public package version and exports.
- Create `kind/models.py`: immutable disclosure, parsed page, match, and validation records.
- Create `kind/selectors.py`: KIND endpoints, form defaults, DOM selectors, and title patterns.
- Create `kind/workbook.py`: workbook contract validation and wide-to-long conversion.
- Create `kind/parser.py`: cached HTML parsing with strict schema checks.
- Create `kind/matching.py`: normalization, title filtering, issuer guard, candidate selection.
- Create `kind/client.py`: Playwright request transport, retry, rate limiting, pagination, and cache.
- Create `kind/validation.py`: invariant checks and warning-report generation.
- Create `kind/pipeline.py`: orchestration, output writing, CLI, and exit codes.
- Create `kind/README.md`: installation, operation, confidence, validation, and recovery guide.
- Create `tests/kind/`: deterministic unit and integration coverage.

## Task 1: Establish the Package and Domain Contracts

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `kind/__init__.py`
- Create: `kind/models.py`
- Create: `kind/selectors.py`
- Create: `tests/kind/test_package_contract.py`

- [ ] **Step 1: Write the failing package-contract test**

```python
from kind.models import Confidence, Disclosure, MatchResult, ParsedPage
from kind.selectors import FORM_DEFAULTS, KIND_MAIN_URL, KIND_SUB_URL


def test_kind_package_exposes_closed_confidence_vocabulary() -> None:
    assert [item.value for item in Confidence] == [
        "EXACT_MATCH",
        "NORMALIZED_MATCH",
        "MULTIPLE_MATCH",
        "NO_MATCH",
    ]
    assert KIND_MAIN_URL.endswith("method=searchTodayDisclosureMain")
    assert KIND_SUB_URL.endswith("/disclosure/todaydisclosure.do")
    assert FORM_DEFAULTS["method"] == "searchTodayDisclosureSub"


def test_disclosure_and_match_records_are_immutable() -> None:
    disclosure = Disclosure(
        announcement_date="2024-04-25",
        time="08:05",
        company="SK하이닉스",
        title="연결재무제표기준영업(잠정)실적(공정공시)",
        submitter="에스케이하이닉스",
        issuer_id="00066",
        receipt_id="20240425000004",
        page=5,
        position=12,
    )
    page = ParsedPage(disclosures=(disclosure,), total_pages=5)
    result = MatchResult(confidence=Confidence.EXACT_MATCH, disclosure=disclosure)
    assert page.disclosures == (disclosure,)
    assert result.disclosure.time == "08:05"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/kind/test_package_contract.py -q`

Expected: collection fails with `ModuleNotFoundError` for `kind.models`.

- [ ] **Step 3: Add dependency, package discovery, ignore rules, and domain records**

Add `"beautifulsoup4>=4.12,<5"` to `[project].dependencies`, add `"kind*"` to the setuptools include list, and add these ignore rules:

```gitignore
kind/~$*.xlsx
kind/cache/
kind/logs/
kind/outputs/
```

Create `kind/__init__.py`:

```python
"""Recover auditable KIND provisional-results announcement timestamps."""

from kind.models import Confidence, Disclosure, MatchResult, ParsedPage

__all__ = ["Confidence", "Disclosure", "MatchResult", "ParsedPage"]
```

Create `kind/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Confidence(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    NORMALIZED_MATCH = "NORMALIZED_MATCH"
    MULTIPLE_MATCH = "MULTIPLE_MATCH"
    NO_MATCH = "NO_MATCH"


@dataclass(frozen=True, slots=True)
class Disclosure:
    announcement_date: str
    time: str
    company: str
    title: str
    submitter: str
    issuer_id: str | None
    receipt_id: str
    page: int
    position: int


@dataclass(frozen=True, slots=True)
class ParsedPage:
    disclosures: tuple[Disclosure, ...]
    total_pages: int


@dataclass(frozen=True, slots=True)
class MatchResult:
    confidence: Confidence
    disclosure: Disclosure | None
    candidates: tuple[Disclosure, ...] = field(default_factory=tuple)
    rejection_reason: str | None = None
```

Create `kind/selectors.py` with one authoritative source for the live boundary:

```python
from __future__ import annotations

KIND_MAIN_URL = (
    "https://kind.krx.co.kr/disclosure/todaydisclosure.do"
    "?method=searchTodayDisclosureMain"
)
KIND_SUB_URL = "https://kind.krx.co.kr/disclosure/todaydisclosure.do"
PARSER_SCHEMA_VERSION = 1

FORM_DEFAULTS: dict[str, str] = {
    "method": "searchTodayDisclosureSub",
    "currentPageSize": "100",
    "orderMode": "0",
    "orderStat": "D",
    "marketType": "",
    "forward": "todaydisclosure_sub",
    "searchMode": "",
    "searchCodeType": "",
    "chose": "S",
    "todayFlag": "N",
    "repIsuSrtCd": "",
    "kosdaqSegment": "",
    "searchCorpName": "",
    "copyUrl": "",
}

TABLE_SELECTOR = "table.list"
ROW_SELECTOR = "tbody tr"
COMPANY_LINK_ONCLICK = "companysummary_open"
DISCLOSURE_LINK_ONCLICK = "openDisclsViewer"
EXPECTED_CELL_COUNT = 5
TIME_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
PAGE_PATTERN = r"fnPageGo\('(\d+)'\)"
ISSUER_PATTERN = r"companysummary_open\('([^']+)'\)"
RECEIPT_PATTERN = r"openDisclsViewer\('([^']+)'"
PROVISIONAL_TITLE_PATTERN = r"영업\s*\(잠정\)\s*실적\s*\(공정공시\)"
FORECAST_TITLE_PATTERN = r"영업실적\s*등에\s*대한\s*전망"
```

- [ ] **Step 4: Lock dependencies and verify GREEN**

Run:

```powershell
uv lock
uv run pytest tests/kind/test_package_contract.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 1**

```powershell
git add pyproject.toml uv.lock .gitignore kind/__init__.py kind/models.py kind/selectors.py tests/kind/test_package_contract.py
git commit -m "Make KIND timestamp decisions explicit" -m "Introduce the closed confidence vocabulary and one source of truth for the observed KIND boundary." -m "Constraint: No unresolved candidate may produce a time" -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: uv run pytest tests/kind/test_package_contract.py -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

## Task 2: Normalize the Wide Timeframe Workbook

**Files:**
- Create: `kind/workbook.py`
- Create: `tests/kind/test_workbook.py`

- [ ] **Step 1: Write workbook fixture and failing conversion tests**

```python
from pathlib import Path

import pandas as pd
import pytest

from kind.workbook import WorkbookSchemaError, read_timeframe


def write_timeframe(path: Path, *, item_name: str = "잠정치발표일") -> None:
    frame = pd.DataFrame([[None] * 3 for _ in range(16)])
    frame.iloc[8] = ["코드", "A005930", "A0126Z0"]
    frame.iloc[9] = ["코드명", "삼성전자", "삼성에피스홀딩스"]
    frame.iloc[10] = ["유형", "FSP-IFRS(M)", "FSP-IFRS(M)"]
    frame.iloc[11] = ["아이템코드", "FP56000500", "FP56000500"]
    frame.iloc[12] = ["아이템명", item_name, item_name]
    frame.iloc[13] = ["집계주기", 6, 6]
    frame.iloc[14] = [pd.Timestamp("2024-03-29"), 20240430, 20240515]
    frame.iloc[15] = [pd.Timestamp("2024-06-28"), 20240731, None]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([[None]]).to_excel(writer, sheet_name="value", header=False, index=False)
        frame.to_excel(writer, sheet_name="timeframe", header=False, index=False)


def test_read_timeframe_preserves_rows_tickers_and_quarters(tmp_path: Path) -> None:
    path = tmp_path / "announcements.xlsx"
    write_timeframe(path)
    result = read_timeframe(path)
    assert result.to_dict("records") == [
        {
            "ticker": "A005930",
            "company": "삼성전자",
            "quarter": "2024Q1",
            "announcement_date": pd.Timestamp("2024-04-30"),
        },
        {
            "ticker": "A0126Z0",
            "company": "삼성에피스홀딩스",
            "quarter": "2024Q1",
            "announcement_date": pd.Timestamp("2024-05-15"),
        },
        {
            "ticker": "A005930",
            "company": "삼성전자",
            "quarter": "2024Q2",
            "announcement_date": pd.Timestamp("2024-07-31"),
        },
    ]


def test_read_timeframe_rejects_changed_item_contract(tmp_path: Path) -> None:
    path = tmp_path / "announcements.xlsx"
    write_timeframe(path, item_name="다른아이템")
    with pytest.raises(WorkbookSchemaError, match="잠정치발표일"):
        read_timeframe(path)
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/kind/test_workbook.py -q`

Expected: import fails because `kind.workbook` does not exist.

- [ ] **Step 3: Implement strict wide-to-long conversion**

Create `kind/workbook.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_LABELS = {
    8: "코드",
    9: "코드명",
    10: "유형",
    11: "아이템코드",
    12: "아이템명",
    13: "집계주기",
}


class WorkbookSchemaError(ValueError):
    pass


def _parse_announcement_date(value: object, *, row: int, column: int) -> pd.Timestamp:
    text = str(int(value)) if isinstance(value, (int, float)) else str(value).strip()
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        raise WorkbookSchemaError(
            f"invalid announcement date at row {row + 1}, column {column + 1}: {value!r}"
        )
    return pd.Timestamp(parsed).normalize()


def read_timeframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        frame = pd.read_excel(path, sheet_name="timeframe", header=None)
    except ValueError as exc:
        raise WorkbookSchemaError("missing required timeframe sheet") from exc
    if frame.shape[0] < 15 or frame.shape[1] < 2:
        raise WorkbookSchemaError(f"timeframe shape is too small: {frame.shape}")
    for row, label in EXPECTED_LABELS.items():
        if str(frame.iat[row, 0]).strip() != label:
            raise WorkbookSchemaError(f"expected {label!r} in timeframe row {row + 1}")
    item_names = frame.iloc[12, 1:].dropna().astype(str).str.strip()
    if item_names.empty or not item_names.eq("잠정치발표일").all():
        raise WorkbookSchemaError("timeframe item must be 잠정치발표일")

    records: list[dict[str, object]] = []
    for row in range(14, len(frame)):
        reference_date = pd.to_datetime(frame.iat[row, 0], errors="coerce")
        if pd.isna(reference_date):
            if frame.iloc[row, 1:].notna().any():
                raise WorkbookSchemaError(f"missing reference date at row {row + 1}")
            continue
        reference_date = pd.Timestamp(reference_date)
        quarter = f"{reference_date.year}Q{reference_date.quarter}"
        for column in range(1, frame.shape[1]):
            value = frame.iat[row, column]
            if pd.isna(value):
                continue
            ticker = str(frame.iat[8, column]).strip()
            company = str(frame.iat[9, column]).strip()
            if not ticker.startswith("A") or not ticker[1:] or not company:
                raise WorkbookSchemaError(
                    f"invalid ticker/company metadata at column {column + 1}"
                )
            records.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "quarter": quarter,
                    "announcement_date": _parse_announcement_date(
                        value, row=row, column=column
                    ),
                }
            )
    result = pd.DataFrame.from_records(
        records,
        columns=["ticker", "company", "quarter", "announcement_date"],
    ).sort_values(["quarter", "ticker", "announcement_date"], ignore_index=True)
    duplicates = result.duplicated(["ticker", "quarter"], keep=False)
    if duplicates.any():
        raise WorkbookSchemaError("duplicate ticker-quarter rows in timeframe input")
    return result
```

- [ ] **Step 4: Verify GREEN and real workbook contract**

Run:

```powershell
uv run pytest tests/kind/test_workbook.py -q
uv run python -c "from kind.workbook import read_timeframe; d=read_timeframe('kind/announcements.xlsx'); assert len(d)==11495; assert d.announcement_date.nunique()==720; print(d.shape)"
```

Expected: tests pass and `(11495, 4)` prints.

- [ ] **Step 5: Commit Task 2**

```powershell
git add kind/workbook.py tests/kind/test_workbook.py
git commit -m "Preserve every source announcement-date observation" -m "Validate the observed wide workbook contract before producing a deterministic long table." -m "Constraint: Source workbook remains immutable" -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: workbook tests and 11,495-row real-input assertion" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

## Task 3: Parse KIND HTML Without Guessing

**Files:**
- Create: `kind/parser.py`
- Create: `tests/kind/test_parser.py`

- [ ] **Step 1: Write failing parser tests with representative HTML**

```python
import pytest

from kind.parser import KindSchemaError, parse_disclosure_page

HTML = """
<section>
  <table class="list type-00 mt10" summary="시간, 회사명, 공시제목, 제출인, 차트/주가">
    <tbody>
      <tr>
        <td>08:05</td>
        <td><a onclick="companysummary_open('00066')"> SK하이닉스 </a></td>
        <td><a onclick="openDisclsViewer('20240425000004','')">연결재무제표기준영업(잠정)실적(공정공시)</a></td>
        <td>에스케이하이닉스</td><td></td>
      </tr>
      <tr><td>18:07</td><td></td><td>시장통계</td><td>시장본부</td><td></td></tr>
    </tbody>
  </table>
  <a onclick="fnPageGo('2')">2</a><a onclick="fnPageGo('5')">5</a>
</section>
"""


def test_parse_disclosure_page_extracts_valid_company_rows() -> None:
    parsed = parse_disclosure_page(HTML, announcement_date="2024-04-25", page=1)
    assert parsed.total_pages == 5
    assert len(parsed.disclosures) == 1
    disclosure = parsed.disclosures[0]
    assert disclosure.time == "08:05"
    assert disclosure.company == "SK하이닉스"
    assert disclosure.issuer_id == "00066"
    assert disclosure.receipt_id == "20240425000004"


def test_parse_disclosure_page_fails_closed_on_invalid_provisional_time() -> None:
    broken = HTML.replace("08:05", "8시5분")
    with pytest.raises(KindSchemaError, match="time"):
        parse_disclosure_page(broken, announcement_date="2024-04-25", page=1)
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/kind/test_parser.py -q`

Expected: import fails because `kind.parser` does not exist.

- [ ] **Step 3: Implement schema-checked parsing**

Create `kind/parser.py`:

```python
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from kind.models import Disclosure, ParsedPage
from kind.selectors import (
    COMPANY_LINK_ONCLICK,
    DISCLOSURE_LINK_ONCLICK,
    EXPECTED_CELL_COUNT,
    ISSUER_PATTERN,
    PAGE_PATTERN,
    PROVISIONAL_TITLE_PATTERN,
    RECEIPT_PATTERN,
    ROW_SELECTOR,
    TABLE_SELECTOR,
    TIME_PATTERN,
)


class KindSchemaError(ValueError):
    pass


def _onclick_value(tag: Tag, pattern: str, field: str) -> str:
    onclick = str(tag.get("onclick", ""))
    match = re.search(pattern, onclick)
    if not match:
        raise KindSchemaError(f"missing {field} in disclosure row")
    return match.group(1)


def parse_disclosure_page(html: str, *, announcement_date: str, page: int) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one(TABLE_SELECTOR)
    if table is None:
        raise KindSchemaError("missing KIND disclosure table")
    pages = [int(value) for value in re.findall(PAGE_PATTERN, html)]
    total_pages = max(pages, default=1)
    disclosures: list[Disclosure] = []
    for position, row in enumerate(table.select(ROW_SELECTOR), start=1):
        cells = row.find_all("td", recursive=False)
        if len(cells) != EXPECTED_CELL_COUNT:
            raise KindSchemaError(
                f"expected {EXPECTED_CELL_COUNT} cells, found {len(cells)}"
            )
        company_link = cells[1].find("a", onclick=re.compile(COMPANY_LINK_ONCLICK))
        disclosure_link = cells[2].find(
            "a", onclick=re.compile(DISCLOSURE_LINK_ONCLICK)
        )
        title = cells[2].get_text(" ", strip=True)
        if company_link is None or disclosure_link is None:
            if re.search(PROVISIONAL_TITLE_PATTERN, title):
                raise KindSchemaError("provisional row is missing company/disclosure link")
            continue
        time = cells[0].get_text(" ", strip=True)
        if not re.fullmatch(TIME_PATTERN, time):
            raise KindSchemaError(f"invalid disclosure time: {time!r}")
        disclosures.append(
            Disclosure(
                announcement_date=announcement_date,
                time=time,
                company=company_link.get_text(" ", strip=True),
                title=title,
                submitter=cells[3].get_text(" ", strip=True),
                issuer_id=_onclick_value(company_link, ISSUER_PATTERN, "issuer id"),
                receipt_id=_onclick_value(disclosure_link, RECEIPT_PATTERN, "receipt id"),
                page=page,
                position=position,
            )
        )
    return ParsedPage(disclosures=tuple(disclosures), total_pages=total_pages)
```

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/kind/test_parser.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 3**

```powershell
git add kind/parser.py tests/kind/test_parser.py
git commit -m "Reject malformed KIND disclosure rows" -m "Parse only validated five-cell company disclosures and preserve receipt-level evidence." -m "Constraint: Market-statistic rows may omit company links" -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: uv run pytest tests/kind/test_parser.py -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

## Task 4: Match Companies and Candidates Conservatively

**Files:**
- Create: `kind/matching.py`
- Create: `tests/kind/test_matching.py`

- [ ] **Step 1: Write failing normalization and candidate-selection tests**

```python
from kind.matching import match_disclosure, normalize_company_name
from kind.models import Confidence, Disclosure


def disclosure(*, company: str, title: str, time: str, receipt: str, issuer: str = "00593") -> Disclosure:
    return Disclosure(
        announcement_date="2024-04-30",
        time=time,
        company=company,
        title=title,
        submitter=company,
        issuer_id=issuer,
        receipt_id=receipt,
        page=1,
        position=1,
    )


def test_normalization_handles_legal_forms_spaces_and_punctuation() -> None:
    assert normalize_company_name(" (주) 삼성-전자 ") == normalize_company_name("삼성전자")


def test_exact_and_normalized_single_candidates_are_accepted() -> None:
    exact = disclosure(
        company="삼성전자",
        title="연결재무제표기준영업(잠정)실적(공정공시)",
        time="08:31",
        receipt="one",
    )
    assert match_disclosure("A005930", "삼성전자", [exact]).confidence is Confidence.EXACT_MATCH
    normalized = disclosure(
        company="(주) 삼성전자",
        title=exact.title,
        time="08:31",
        receipt="two",
    )
    assert match_disclosure("A005930", "삼성전자", [normalized]).confidence is Confidence.NORMALIZED_MATCH


def test_original_direct_candidate_supersedes_correction_and_subsidiary() -> None:
    direct = disclosure(company="삼성전자", title="영업(잠정)실적(공정공시)", time="08:31", receipt="direct")
    correction = disclosure(company="삼성전자", title="[정정]영업(잠정)실적(공정공시)", time="10:00", receipt="correction")
    subsidiary = disclosure(company="삼성전자", title="영업(잠정)실적(공정공시)(자회사의 주요경영사항)", time="11:00", receipt="child")
    result = match_disclosure("A005930", "삼성전자", [correction, subsidiary, direct])
    assert result.disclosure == direct
    assert result.confidence is Confidence.EXACT_MATCH


def test_two_direct_candidates_fail_closed() -> None:
    candidates = [
        disclosure(company="삼성전자", title="영업(잠정)실적(공정공시)", time="08:31", receipt="one"),
        disclosure(company="삼성전자", title="연결재무제표기준영업(잠정)실적(공정공시)", time="08:32", receipt="two"),
    ]
    result = match_disclosure("A005930", "삼성전자", candidates)
    assert result.confidence is Confidence.MULTIPLE_MATCH
    assert result.disclosure is None


def test_forecast_and_issuer_conflict_do_not_match() -> None:
    forecast = disclosure(company="삼성전자", title="영업실적 등에 대한 전망(공정공시)", time="08:31", receipt="one")
    wrong_issuer = disclosure(company="삼성전자", title="영업(잠정)실적(공정공시)", time="08:31", receipt="two", issuer="99999")
    result = match_disclosure("A005930", "삼성전자", [forecast, wrong_issuer])
    assert result.confidence is Confidence.NO_MATCH
    assert result.disclosure is None
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/kind/test_matching.py -q`

Expected: import fails because `kind.matching` does not exist.

- [ ] **Step 3: Implement deterministic normalization and fail-closed selection**

Create `kind/matching.py`:

```python
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from kind.models import Confidence, Disclosure, MatchResult
from kind.selectors import FORECAST_TITLE_PATTERN, PROVISIONAL_TITLE_PATTERN

LEGAL_FORMS = ("(주)", "주식회사")


def normalize_company_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    for legal_form in LEGAL_FORMS:
        normalized = normalized.replace(legal_form, "")
    return "".join(character for character in normalized if character.isalnum())


def _issuer_id_for_ticker(ticker: str) -> str | None:
    code = ticker[1:] if ticker.startswith("A") else ticker
    return code[:-1] if len(code) == 6 else None


def _is_eligible(disclosure: Disclosure) -> bool:
    return bool(re.search(PROVISIONAL_TITLE_PATTERN, disclosure.title)) and not bool(
        re.search(FORECAST_TITLE_PATTERN, disclosure.title)
    )


def _deduplicate_receipts(candidates: Iterable[Disclosure]) -> list[Disclosure]:
    return list({candidate.receipt_id: candidate for candidate in candidates}.values())


def _prefer_direct_original(candidates: list[Disclosure]) -> list[Disclosure]:
    direct_original = [
        candidate
        for candidate in candidates
        if not candidate.title.startswith("[정정]")
        and "자회사의 주요경영사항" not in candidate.title
    ]
    if direct_original:
        return direct_original
    direct = [
        candidate
        for candidate in candidates
        if "자회사의 주요경영사항" not in candidate.title
    ]
    if direct:
        return direct
    original = [candidate for candidate in candidates if not candidate.title.startswith("[정정]")]
    return original or candidates


def match_disclosure(
    ticker: str,
    company: str,
    disclosures: Iterable[Disclosure],
) -> MatchResult:
    expected_issuer = _issuer_id_for_ticker(ticker)
    eligible = [candidate for candidate in disclosures if _is_eligible(candidate)]
    exact = [candidate for candidate in eligible if candidate.company.strip() == company.strip()]
    confidence = Confidence.EXACT_MATCH
    candidates = exact
    if not candidates:
        expected_name = normalize_company_name(company)
        candidates = [
            candidate
            for candidate in eligible
            if expected_name and normalize_company_name(candidate.company) == expected_name
        ]
        confidence = Confidence.NORMALIZED_MATCH
    if expected_issuer is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.issuer_id is None or candidate.issuer_id == expected_issuer
        ]
    candidates = _prefer_direct_original(_deduplicate_receipts(candidates))
    if not candidates:
        return MatchResult(
            confidence=Confidence.NO_MATCH,
            disclosure=None,
            rejection_reason="no eligible company/title candidate",
        )
    if len(candidates) > 1:
        return MatchResult(
            confidence=Confidence.MULTIPLE_MATCH,
            disclosure=None,
            candidates=tuple(candidates),
            rejection_reason="multiple eligible receipt identifiers",
        )
    return MatchResult(
        confidence=confidence,
        disclosure=candidates[0],
        candidates=tuple(candidates),
    )
```

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/kind/test_matching.py -q`

Expected: `5 passed`.

- [ ] **Step 5: Commit Task 4**

```powershell
git add kind/matching.py tests/kind/test_matching.py
git commit -m "Prefer missing times over ambiguous matches" -m "Normalize company names conservatively and require one receipt after deterministic title and issuer guards." -m "Rejected: Nearest-name matching | could assign a plausible but wrong disclosure" -m "Confidence: high" -m "Scope-risk: narrow" -m "Directive: Multiple direct receipts must remain unresolved" -m "Tested: uv run pytest tests/kind/test_matching.py -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

## Task 5: Add the Playwright Client, Retry, Rate Limit, and Atomic Cache

**Files:**
- Create: `kind/client.py`
- Create: `tests/kind/test_client.py`

- [ ] **Step 1: Write failing cache and retry tests around an injected transport**

```python
import asyncio
from pathlib import Path

from kind.client import KindClient


class FakeTransport:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.posts: list[dict[str, str]] = []
        self.gets = 0

    async def get(self, url: str, *, timeout_ms: int) -> str:
        self.gets += 1
        return "main"

    async def post_form(self, url: str, form: dict[str, str], *, timeout_ms: int) -> str:
        self.posts.append(form.copy())
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


ONE_PAGE = """<table class='list'><tbody></tbody></table>"""


def test_fetch_date_caches_and_reuses_response(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = FakeTransport([ONE_PAGE])
        client = KindClient(transport, cache_dir=tmp_path, min_delay=0, max_attempts=1)
        first = await client.fetch_date("2024-04-25")
        second = await client.fetch_date("2024-04-25")
        assert first == second
        assert len(transport.posts) == 1
        assert (tmp_path / "2024-04-25" / "page-0001.html").read_text("utf-8") == ONE_PAGE
        assert (tmp_path / "2024-04-25" / "manifest.json").exists()

    asyncio.run(scenario())


def test_fetch_date_retries_transient_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = FakeTransport([TimeoutError("slow"), ONE_PAGE])
        client = KindClient(transport, cache_dir=tmp_path, min_delay=0, max_attempts=2)
        await client.fetch_date("2024-04-25")
        assert len(transport.posts) == 2

    asyncio.run(scenario())


def test_fetch_date_rejects_corrupt_cache(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_transport = FakeTransport([ONE_PAGE])
        first = KindClient(first_transport, cache_dir=tmp_path, min_delay=0, max_attempts=1)
        paths = await first.fetch_date("2024-04-25")
        paths[0].write_text("corrupt", encoding="utf-8")
        second_transport = FakeTransport([ONE_PAGE])
        second = KindClient(second_transport, cache_dir=tmp_path, min_delay=0, max_attempts=1)
        await second.fetch_date("2024-04-25")
        assert len(second_transport.posts) == 1

    asyncio.run(scenario())
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/kind/test_client.py -q`

Expected: import fails because `kind.client` does not exist.

- [ ] **Step 3: Implement transport protocol, Playwright transport, limiter, retry, and cache**

Create `kind/client.py` with these public contracts and exact behaviors:

```python
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from playwright.async_api import APIRequestContext, Playwright, async_playwright

from kind.parser import parse_disclosure_page
from kind.selectors import FORM_DEFAULTS, KIND_MAIN_URL, KIND_SUB_URL, PARSER_SCHEMA_VERSION


class Transport(Protocol):
    async def get(self, url: str, *, timeout_ms: int) -> str: ...
    async def post_form(self, url: str, form: dict[str, str], *, timeout_ms: int) -> str: ...


class PlaywrightTransport:
    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._request: APIRequestContext | None = None

    async def __aenter__(self) -> "PlaywrightTransport":
        self._playwright = await async_playwright().start()
        self._request = await self._playwright.request.new_context(
            extra_http_headers={"Referer": KIND_MAIN_URL}
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._request is not None:
            await self._request.dispose()
        if self._playwright is not None:
            await self._playwright.stop()

    async def get(self, url: str, *, timeout_ms: int) -> str:
        if self._request is None:
            raise RuntimeError("PlaywrightTransport is not open")
        response = await self._request.get(url, timeout=timeout_ms)
        if not response.ok:
            raise RuntimeError(f"GET {url} returned {response.status}")
        return await response.text()

    async def post_form(self, url: str, form: dict[str, str], *, timeout_ms: int) -> str:
        if self._request is None:
            raise RuntimeError("PlaywrightTransport is not open")
        response = await self._request.post(url, form=form, timeout=timeout_ms)
        if not response.ok:
            raise RuntimeError(f"POST {url} returned {response.status}")
        return await response.text()


class RateLimiter:
    def __init__(self, min_delay: float) -> None:
        self._min_delay = min_delay
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._min_delay - (time.monotonic() - self._last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request = time.monotonic()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


class KindClient:
    def __init__(
        self,
        transport: Transport,
        *,
        cache_dir: Path,
        min_delay: float = 0.75,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        self.transport = transport
        self.cache_dir = Path(cache_dir)
        self.timeout_ms = int(timeout_seconds * 1000)
        self.max_attempts = max_attempts
        self.limiter = RateLimiter(min_delay)
        self._initialized = False

    async def _request_with_retry(self, form: dict[str, str]) -> str:
        error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                await self.limiter.wait()
                return await self.transport.post_form(
                    KIND_SUB_URL, form, timeout_ms=self.timeout_ms
                )
            except Exception as exc:
                error = exc
                if attempt < self.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"KIND request failed after {self.max_attempts} attempts") from error

    async def fetch_date(self, date: str, *, refresh: bool = False) -> tuple[Path, ...]:
        date_dir = self.cache_dir / date
        manifest_path = date_dir / "manifest.json"
        if not refresh and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = tuple(date_dir / name for name in manifest["pages"])
            stored_hashes = manifest.get("sha256", {})
            valid_hashes = all(
                path.exists()
                and hashlib.sha256(path.read_bytes()).hexdigest()
                == stored_hashes.get(path.name)
                for path in paths
            )
            if (
                manifest.get("parser_schema_version") == PARSER_SCHEMA_VERSION
                and valid_hashes
            ):
                return paths
        if not self._initialized:
            await self.transport.get(KIND_MAIN_URL, timeout_ms=self.timeout_ms)
            self._initialized = True
        page_number = 1
        pages: list[Path] = []
        hashes: dict[str, str] = {}
        total_pages = 1
        while page_number <= total_pages:
            form = {**FORM_DEFAULTS, "selDate": date, "pageIndex": str(page_number)}
            html = await self._request_with_retry(form)
            parsed = parse_disclosure_page(html, announcement_date=date, page=page_number)
            total_pages = parsed.total_pages
            page_path = date_dir / f"page-{page_number:04d}.html"
            _atomic_write(page_path, html)
            pages.append(page_path)
            hashes[page_path.name] = hashlib.sha256(html.encode("utf-8")).hexdigest()
            page_number += 1
        manifest = {
            "date": date,
            "source_url": KIND_SUB_URL,
            "form": {**FORM_DEFAULTS, "selDate": date},
            "page_count": total_pages,
            "pages": [path.name for path in pages],
            "sha256": hashes,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "parser_schema_version": PARSER_SCHEMA_VERSION,
        }
        _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
        return tuple(pages)
```

- [ ] **Step 4: Verify GREEN and cache-corruption protection**

Run: `uv run pytest tests/kind/test_client.py -q`

Expected: cache-hit, retry, refresh, pagination, and corruption tests all pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add kind/client.py tests/kind/test_client.py
git commit -m "Make KIND retrieval restartable and polite" -m "Use one Playwright request context with bounded retry, global rate limiting, pagination, and hash-verified atomic date caches." -m "Constraint: 720 distinct dates must survive interruption and rerun" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/kind/test_client.py -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

## Task 6: Validate Results and Write Warning/Output Artifacts

**Files:**
- Create: `kind/validation.py`
- Create: `tests/kind/test_validation.py`

- [ ] **Step 1: Write failing validation and output tests**

```python
from pathlib import Path

import pandas as pd

from kind.validation import validate_and_write_reports, write_outputs


def test_validation_reports_missing_and_duplicate_times(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A005930", "A005930", "A000660"],
            "company": ["삼성전자", "삼성전자", "SK하이닉스"],
            "quarter": ["2024Q1", "2024Q1", "2024Q1"],
            "announcement_date": pd.to_datetime(["2024-04-30", "2024-04-30", "2024-04-25"]),
            "announcement_datetime": pd.to_datetime(["2024-04-30 08:31", "2024-04-30 09:00", None]),
            "confidence": ["EXACT_MATCH", "EXACT_MATCH", "NO_MATCH"],
        }
    )
    summary = validate_and_write_reports(frame, input_row_count=3, log_dir=tmp_path)
    assert summary.missing_match_count == 1
    assert summary.duplicate_match_count == 2
    assert (tmp_path / "missing_match.csv").exists()
    assert (tmp_path / "duplicate_match.csv").exists()
    assert (tmp_path / "validation_summary.json").exists()


def test_write_outputs_preserves_datetime_types(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A005930"], "company": ["삼성전자"], "quarter": ["2024Q1"],
            "announcement_date": pd.to_datetime(["2024-04-30"]),
            "announcement_datetime": pd.to_datetime(["2024-04-30 08:31"]),
            "confidence": ["EXACT_MATCH"],
        }
    )
    audit = pd.DataFrame([{"ticker": "A005930", "receipt_id": "one"}])
    write_outputs(frame, audit, output_dir=tmp_path)
    loaded = pd.read_excel(tmp_path / "announcements_with_time.xlsx")
    assert str(loaded["announcement_datetime"].dtype).startswith("datetime64")
    assert (tmp_path / "announcements_with_time.csv").read_text("utf-8").count("08:31") == 1
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/kind/test_validation.py -q`

Expected: import fails because `kind.validation` does not exist.

- [ ] **Step 3: Implement invariants, reports, and typed output**

Create `kind/validation.py` with:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from kind.models import Confidence

PRIMARY_COLUMNS = [
    "ticker", "company", "quarter", "announcement_date",
    "announcement_datetime", "confidence",
]


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    input_row_count: int
    output_row_count: int
    missing_match_count: int
    duplicate_match_count: int
    multiple_candidate_count: int

    @property
    def is_strictly_complete(self) -> bool:
        return (
            self.input_row_count == self.output_row_count
            and self.missing_match_count == 0
            and self.duplicate_match_count == 0
            and self.multiple_candidate_count == 0
        )


def validate_and_write_reports(
    frame: pd.DataFrame,
    *,
    input_row_count: int,
    log_dir: Path,
) -> ValidationSummary:
    log_dir.mkdir(parents=True, exist_ok=True)
    if list(frame.columns) != PRIMARY_COLUMNS:
        raise ValueError(f"unexpected output columns: {list(frame.columns)}")
    if len(frame) != input_row_count:
        raise ValueError(f"row-count mismatch: {input_row_count} != {len(frame)}")
    required = frame[["ticker", "company", "quarter", "announcement_date"]]
    if required.isna().any().any():
        raise ValueError("required output identity/date fields contain nulls")
    valid_confidence = {item.value for item in Confidence}
    if not set(frame["confidence"]).issubset(valid_confidence):
        raise ValueError("unknown confidence value")
    matched = frame["confidence"].isin(
        [Confidence.EXACT_MATCH.value, Confidence.NORMALIZED_MATCH.value]
    )
    if frame.loc[matched, "announcement_datetime"].isna().any():
        raise ValueError("matched row is missing announcement_datetime")
    if frame.loc[~matched, "announcement_datetime"].notna().any():
        raise ValueError("unresolved row contains announcement_datetime")
    missing = frame.loc[frame["announcement_datetime"].isna()].copy()
    duplicate_mask = frame.groupby(["ticker", "quarter"])["announcement_datetime"].transform("nunique") > 1
    duplicates = frame.loc[duplicate_mask].copy()
    multiples = frame.loc[frame["confidence"].eq(Confidence.MULTIPLE_MATCH.value)].copy()
    missing.to_csv(log_dir / "missing_match.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(log_dir / "duplicate_match.csv", index=False, encoding="utf-8-sig")
    multiples.to_csv(log_dir / "multiple_candidate.csv", index=False, encoding="utf-8-sig")
    summary = ValidationSummary(
        input_row_count=input_row_count,
        output_row_count=len(frame),
        missing_match_count=len(missing),
        duplicate_match_count=len(duplicates),
        multiple_candidate_count=len(multiples),
    )
    (log_dir / "validation_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_schema_errors(errors: list[dict[str, object]], *, log_dir: Path) -> None:
    columns = ["announcement_date", "cache_page", "error_type", "message"]
    pd.DataFrame.from_records(errors, columns=columns).to_csv(
        log_dir / "schema_error.csv", index=False, encoding="utf-8-sig"
    )


def write_outputs(frame: pd.DataFrame, audit: pd.DataFrame, *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(
        ["quarter", "ticker", "announcement_date"], ignore_index=True
    )
    csv_frame = ordered.copy()
    csv_frame["announcement_date"] = csv_frame["announcement_date"].dt.strftime("%Y-%m-%d")
    csv_frame["announcement_datetime"] = csv_frame["announcement_datetime"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    csv_frame.to_csv(
        output_dir / "announcements_with_time.csv", index=False, encoding="utf-8-sig"
    )
    audit.to_csv(output_dir / "match_audit.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "announcements_with_time.xlsx", engine="openpyxl") as writer:
        ordered.to_excel(writer, sheet_name="announcements", index=False)
        audit.to_excel(writer, sheet_name="match_audit", index=False)
```

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/kind/test_validation.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 6**

```powershell
git add kind/validation.py tests/kind/test_validation.py
git commit -m "Make incomplete KIND recovery impossible to overlook" -m "Write typed primary artifacts and materialize every missing, duplicate, or multiple match in strict validation reports." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: uv run pytest tests/kind/test_validation.py -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

## Task 7: Orchestrate the End-to-End Pipeline and CLI

**Files:**
- Create: `kind/pipeline.py`
- Create: `tests/kind/test_pipeline.py`

- [ ] **Step 1: Write a failing cached-fixture integration test**

The test creates a two-row workbook with the helper from Task 2, a date cache with the HTML from Task 3, and invokes a dependency-injected `run_pipeline`. Assert:

```python
assert len(result.frame) == 2
assert list(result.frame["confidence"]) == ["EXACT_MATCH", "NO_MATCH"]
assert result.frame.loc[0, "announcement_datetime"] == pd.Timestamp("2024-04-30 08:31")
assert pd.isna(result.frame.loc[1, "announcement_datetime"])
assert (log_dir / "missing_match.csv").exists()
assert (output_dir / "announcements_with_time.xlsx").exists()
```

Also test that fetching two dates respects the configured semaphore and that a `KindSchemaError` becomes a `schema_error.csv` row while every input row remains in the result.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/kind/test_pipeline.py -q`

Expected: import fails because `kind.pipeline` does not exist.

- [ ] **Step 3: Implement orchestration and CLI**

Create `kind/pipeline.py` with these concrete stages:

```python
input_frame = read_timeframe(input_path)
unique_dates = sorted(input_frame["announcement_date"].dt.strftime("%Y-%m-%d").unique())
cache_pages = await fetch_dates_with_semaphore(client, unique_dates, concurrency)
disclosures_by_date, schema_errors = parse_cached_dates(cache_pages)
result_frame, audit_frame = match_all_rows(input_frame, disclosures_by_date)
summary = validate_and_write_reports(result_frame, input_row_count=len(input_frame), log_dir=log_dir)
write_schema_errors(schema_errors, log_dir / "schema_error.csv")
write_outputs(result_frame, audit_frame, output_dir=output_dir)
return PipelineResult(frame=result_frame, audit=audit_frame, validation=summary)
```

`match_all_rows` must initialize every result row with `announcement_datetime=pd.NaT`, call `match_disclosure` only against disclosures from the same announcement date, set datetime only for exact/normalized single matches, and preserve candidate receipt/title/page evidence in the audit frame.

The CLI must expose only:

```text
--input
--output-dir
--cache-dir
--log-dir
--refresh
--concurrency
--min-delay
--timeout
--max-attempts
```

`main()` opens `PlaywrightTransport`, runs the async pipeline, prints the validation summary, and returns exit code `0` only when strict validation is complete; otherwise it returns `2` after outputs and logs are safely written.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
uv run pytest tests/kind/test_pipeline.py -q
uv run pytest tests/kind -q
```

Expected: all KIND tests pass.

- [ ] **Step 5: Commit Task 7**

```powershell
git add kind/pipeline.py tests/kind/test_pipeline.py
git commit -m "Join KIND evidence to every timeframe row" -m "Orchestrate bounded async collection, date-local matching, strict reporting, and typed output without dropping unresolved rows." -m "Constraint: Output row count must remain 11,495 for the current input" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: uv run pytest tests/kind -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

## Task 8: Document, Smoke-Test, Run All Dates, and Prove Idempotence

**Files:**
- Create: `kind/README.md`
- Modify tests or matching/parser code only when a live discrepancy is first reproduced by a failing deterministic test.

- [ ] **Step 1: Write README**

Document:

- `uv sync` and `uv run python -m playwright install chromium`.
- The exact CLI command from the design.
- The wide input contract and immutable-source rule.
- Cache layout, manifest hashes, `--refresh`, retry, timeout, and rate-limit defaults.
- Exact meanings of `EXACT_MATCH`, `NORMALIZED_MATCH`, `MULTIPLE_MATCH`, and `NO_MATCH`.
- Why unresolved rows have `NaT` and make the CLI return `2`.
- Locations and columns of output, audit, and warning files.
- Recovery after interruption and response-schema changes.

- [ ] **Step 2: Run deterministic verification**

```powershell
uv run pytest tests/kind -q
uv run python -m compileall -q kind tests/kind
.venv\Scripts\python.exe -m ruff check kind tests/kind
```

Expected: all commands exit `0` with no warnings from project code.

- [ ] **Step 3: Run bounded live smoke checks**

Add a read-only smoke option or invoke the client directly for `2024-04-25` and `2024-04-30`. Verify from cached HTML that:

- `2024-04-25` includes SK하이닉스 provisional results with the visible KIND time.
- `2024-04-30` includes Samsung Electronics provisional results with the visible KIND time.
- Every cached page hash matches its manifest.
- Pagination count equals the number of cached page files.

Do not hardcode smoke times into production matching. If the live site contradicts a parser or matching assumption, add a minimal cached fixture and failing regression test before changing code.

- [ ] **Step 4: Execute the full 720-date pipeline**

```powershell
uv run python -m kind.pipeline `
  --input kind/announcements.xlsx `
  --output-dir kind/outputs `
  --cache-dir kind/cache `
  --log-dir kind/logs
```

Expected: 11,495 output rows. Exit `0` is required only if all rows have certain times; exit `2` is acceptable as an intermediate diagnostic while `missing_match.csv` or `multiple_candidate.csv` identifies unresolved evidence. Continue by classifying deterministic causes and adding regression tests. Never weaken the fail-closed rule to force exit `0`.

- [ ] **Step 5: Audit requirements and iterate on evidence-backed gaps**

Inspect:

```powershell
uv run python -c "import json,pandas as pd; d=pd.read_csv('kind/outputs/announcements_with_time.csv'); s=json.load(open('kind/logs/validation_summary.json',encoding='utf-8')); assert len(d)==11495; print(s); print(d.confidence.value_counts(dropna=False))"
```

For every non-zero warning category, determine whether the source genuinely lacks one unambiguous disclosure or whether a deterministic normalization/title/schema rule is missing. Add a failing test for each newly supported rule. Rows that remain genuinely ambiguous stay unresolved and documented; they are not assigned guessed times.

- [ ] **Step 6: Prove cached-run idempotence**

Record SHA-256 hashes of the CSV and audit file, rerun the same command without `--refresh`, and assert:

```powershell
Get-FileHash kind/outputs/announcements_with_time.csv -Algorithm SHA256
Get-FileHash kind/outputs/match_audit.csv -Algorithm SHA256
```

The hashes before and after the cached rerun must match. KIND request logs must show zero refetches for hash-valid cached dates.

- [ ] **Step 7: Run full repository verification**

```powershell
uv run pytest
uv run python -m compileall -q kind tests/kind
.venv\Scripts\python.exe -m ruff check kind tests/kind
git diff --check
git status --short
```

Expected: tests, compilation, Ruff, and diff checks pass. `git status` shows only intended KIND changes plus the user's pre-existing unrelated untracked files.

- [ ] **Step 8: Commit documentation and final verified behavior**

```powershell
git add kind/README.md kind tests/kind pyproject.toml uv.lock .gitignore
git commit -m "Make KIND time recovery reproducible end to end" -m "Document the immutable input, cached collection, confidence semantics, strict reports, and verified full-dataset workflow." -m "Constraint: Ambiguous or missing evidence remains unresolved" -m "Confidence: high" -m "Scope-risk: moderate" -m "Directive: Keep raw cache and generated outputs out of Git" -m "Tested: full pytest, compileall, Ruff, bounded live smoke, 11,495-row run, cached rerun hash comparison" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

## Plan Self-Review Checklist

- Every design requirement maps to a task: source parsing (Task 2), Playwright/date cache/retry/timeout/rate limiting (Task 5), selector separation and HTML schema (Tasks 1 and 3), robust matching/confidence (Task 4), validations/logs/output (Task 6), orchestration (Task 7), README/full run/idempotence (Task 8).
- The source workbook is never written.
- Every production behavior begins with a failing test.
- No task introduces a time-estimation heuristic.
- The only new dependency is the explicitly permitted BeautifulSoup package.
- The final audit uses the real 11,495-row scope rather than a narrow test subset.
