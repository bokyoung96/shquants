# Upbit Listing Short Event Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable Playwright-based Upbit listing-event collector and a Binance USD-M one-minute event study that ranks cost-adjusted short returns at the first announcement and original scheduled-support timestamps.

**Architecture:** A focused `backtesting.strategies.upbit_listing_event` package separates immutable domain records, Upbit collection/parsing, Binance archive retrieval/matching, event-return calculation, orchestration, and reporting. Raw Upbit JSON and Binance ZIP/checksum inputs are cached before parsing; every unresolved candidate or missing horizon is materialized instead of guessed. A thin script wires the components together, while all research math remains in the package.

**Tech Stack:** Python 3.11+, Playwright async request context, httpx, pandas, numpy, matplotlib, openpyxl, BeautifulSoup, pytest, ruff, mypy, Binance official USD-M public-data archives and public REST endpoints.

---

## Scope and Execution Notes

- Implement in a dedicated worktree because the primary checkout already contains unrelated user changes.
- Do not modify or delete the existing EMP008 work visible in the primary checkout.
- Do not add dependencies; every required package is already declared in `pyproject.toml`.
- Keep all timestamps timezone-aware. Store KST source values and UTC join values explicitly.
- Keep `ANNOUNCEMENT` and `SCHEDULED_SUPPORT` as independent rows even when they share a notice UUID.
- Keep every coin in a multi-asset notice as an independent observation.
- Do not add stop-loss, take-profit, leverage, p-values, confidence intervals, or clustered inference.
- Every task uses red-green-refactor and ends with a Lore-protocol commit.

## File Map

Create the following production files:

```text
backtesting/strategies/upbit_listing_event/__init__.py
backtesting/strategies/upbit_listing_event/models.py
backtesting/strategies/upbit_listing_event/upbit_parser.py
backtesting/strategies/upbit_listing_event/upbit_client.py
backtesting/strategies/upbit_listing_event/matching.py
backtesting/strategies/upbit_listing_event/binance_client.py
backtesting/strategies/upbit_listing_event/study.py
backtesting/strategies/upbit_listing_event/pipeline.py
backtesting/strategies/upbit_listing_event/report.py
backtesting/strategies/upbit_listing_event/README.md
scripts/run_upbit_listing_event_study.py
```

Create these tests and deterministic fixtures:

```text
tests/strategies/upbit_listing_event/__init__.py
tests/strategies/upbit_listing_event/fixtures/upbit_trade_page.json
tests/strategies/upbit_listing_event/fixtures/upbit_prom_detail.json
tests/strategies/upbit_listing_event/fixtures/upbit_dos_updated_detail.json
tests/strategies/upbit_listing_event/fixtures/upbit_multi_asset_detail.json
tests/strategies/upbit_listing_event/fixtures/binance_prom_1m.csv
tests/strategies/upbit_listing_event/fixtures/binance_btcusdt_1m.csv
tests/strategies/upbit_listing_event/test_models.py
tests/strategies/upbit_listing_event/test_upbit_parser.py
tests/strategies/upbit_listing_event/test_upbit_client.py
tests/strategies/upbit_listing_event/test_matching.py
tests/strategies/upbit_listing_event/test_binance_client.py
tests/strategies/upbit_listing_event/test_study.py
tests/strategies/upbit_listing_event/test_pipeline.py
tests/strategies/upbit_listing_event/test_report.py
tests/strategies/upbit_listing_event/test_live_smoke.py
tests/scripts/test_run_upbit_listing_event_study.py
```

Generated runtime data stays under ignored paths:

```text
raw/upbit_listing_event/upbit/
raw/upbit_listing_event/binance/
results/upbit_listing_event_study/
```

## Task 1: Lock the Domain and Timestamp Contract

**Files:**
- Create: `backtesting/strategies/upbit_listing_event/__init__.py`
- Create: `backtesting/strategies/upbit_listing_event/models.py`
- Create: `tests/strategies/upbit_listing_event/__init__.py`
- Create: `tests/strategies/upbit_listing_event/test_models.py`

- [ ] **Step 1: Write failing model-invariant tests**

Create `tests/strategies/upbit_listing_event/test_models.py`:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from backtesting.strategies.upbit_listing_event.models import (
    CostAssumptions,
    EventRecord,
    EventType,
    NoticeAsset,
    SourceConfidence,
)


KST = ZoneInfo("Asia/Seoul")


def test_notice_asset_requires_timezone_aware_source_times() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        NoticeAsset(
            notice_id=6466,
            notice_uuid="1408465020",
            title="?꾨＼(PROM) KRW, USDT 留덉폆 ?붿????먯궛 異붽?",
            asset_name="?꾨＼",
            upbit_ticker="PROM",
            upbit_markets=("KRW", "USDT"),
            network="Ethereum",
            first_listed_at=datetime(2026, 8, 12, 11, 11, 38),
            listed_at=datetime(2026, 8, 12, 11, 11, 38, tzinfo=KST),
            original_scheduled_at=None,
            original_scheduled_text=None,
            source_confidence=SourceConfidence.UNMODIFIED_CURRENT,
            detail_snapshot_sha256="a" * 64,
        )


def test_event_record_preserves_notice_asset_event_type_identity() -> None:
    event = EventRecord(
        notice_uuid="510217693",
        ticker="AIOZ",
        event_type=EventType.ANNOUNCEMENT,
        event_at=datetime(2026, 8, 10, 11, 5, 54, tzinfo=KST),
    )

    assert event.identity == ("510217693", "AIOZ", "ANNOUNCEMENT")
    assert event.event_at_utc.isoformat() == "2026-08-10T02:05:54+00:00"


def test_cost_assumptions_are_per_side_and_nonnegative() -> None:
    assumptions = CostAssumptions(fee_bps_per_side=5.0, slippage_bps_per_side=5.0)

    assert assumptions.fee_round_trip == pytest.approx(0.001)
    assert assumptions.total_round_trip == pytest.approx(0.002)

    with pytest.raises(ValueError, match="nonnegative"):
        CostAssumptions(fee_bps_per_side=-1.0, slippage_bps_per_side=5.0)
```

- [ ] **Step 2: Run the model tests and verify the import failure**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_models.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError` for `backtesting.strategies.upbit_listing_event`.

- [ ] **Step 3: Add the immutable domain records and enums**

Create `backtesting/strategies/upbit_listing_event/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class EventType(str, Enum):
    ANNOUNCEMENT = "ANNOUNCEMENT"
    SCHEDULED_SUPPORT = "SCHEDULED_SUPPORT"


class SourceConfidence(str, Enum):
    UNMODIFIED_CURRENT = "UNMODIFIED_CURRENT"
    RECONSTRUCTED_ORIGINAL = "RECONSTRUCTED_ORIGINAL"
    FIRST_SEEN_SNAPSHOT = "FIRST_SEEN_SNAPSHOT"
    UNRESOLVED_ORIGINAL = "UNRESOLVED_ORIGINAL"


class ExclusionReason(str, Enum):
    NOT_A_LISTING = "NOT_A_LISTING"
    ASSET_PARSE_FAILED = "ASSET_PARSE_FAILED"
    ORIGINAL_SCHEDULE_UNRESOLVED = "ORIGINAL_SCHEDULE_UNRESOLVED"
    BINANCE_SYMBOL_UNRESOLVED = "BINANCE_SYMBOL_UNRESOLVED"
    BINANCE_PERPETUAL_NOT_ACTIVE = "BINANCE_PERPETUAL_NOT_ACTIVE"
    PRE_EVENT_CANDLE_MISSING = "PRE_EVENT_CANDLE_MISSING"
    ENTRY_CANDLE_MISSING = "ENTRY_CANDLE_MISSING"
    EXIT_CANDLE_MISSING = "EXIT_CANDLE_MISSING"
    BINANCE_SCHEMA_ERROR = "BINANCE_SCHEMA_ERROR"
    UPBIT_SCHEMA_ERROR = "UPBIT_SCHEMA_ERROR"


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class NoticeSummary:
    notice_id: int
    notice_uuid: str
    title: str
    category: str
    first_listed_at: datetime
    listed_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.first_listed_at, "first_listed_at")
        _require_aware(self.listed_at, "listed_at")


@dataclass(frozen=True, slots=True)
class NoticeAsset:
    notice_id: int
    notice_uuid: str
    title: str
    asset_name: str
    upbit_ticker: str
    upbit_markets: tuple[str, ...]
    network: str | None
    first_listed_at: datetime
    listed_at: datetime
    original_scheduled_at: datetime | None
    original_scheduled_text: str | None
    source_confidence: SourceConfidence
    detail_snapshot_sha256: str

    def __post_init__(self) -> None:
        _require_aware(self.first_listed_at, "first_listed_at")
        _require_aware(self.listed_at, "listed_at")
        if self.original_scheduled_at is not None:
            _require_aware(self.original_scheduled_at, "original_scheduled_at")
        if not self.upbit_ticker:
            raise ValueError("upbit_ticker must not be empty")


@dataclass(frozen=True, slots=True)
class EventRecord:
    notice_uuid: str
    ticker: str
    event_type: EventType
    event_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.event_at, "event_at")

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.notice_uuid, self.ticker, self.event_type.value

    @property
    def event_at_utc(self) -> datetime:
        return self.event_at.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class CostAssumptions:
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 5.0

    def __post_init__(self) -> None:
        if self.fee_bps_per_side < 0 or self.slippage_bps_per_side < 0:
            raise ValueError("cost assumptions must be nonnegative")

    @property
    def fee_round_trip(self) -> float:
        return 2.0 * self.fee_bps_per_side / 10_000.0

    @property
    def total_round_trip(self) -> float:
        return 2.0 * (
            self.fee_bps_per_side + self.slippage_bps_per_side
        ) / 10_000.0


@dataclass(frozen=True, slots=True)
class SymbolMatch:
    ticker: str
    symbol: str | None
    reason: ExclusionReason | None


@dataclass(frozen=True, slots=True)
class Exclusion:
    notice_uuid: str | None
    ticker: str | None
    event_type: EventType | None
    reason: ExclusionReason
    detail: str
    horizon_minutes: int | None = None
```

Create `backtesting/strategies/upbit_listing_event/__init__.py` with a package docstring and exports for `CostAssumptions`, `EventRecord`, `EventType`, `Exclusion`, `ExclusionReason`, `NoticeAsset`, `NoticeSummary`, `SourceConfidence`, and `SymbolMatch`. Create an empty `tests/strategies/upbit_listing_event/__init__.py`.

- [ ] **Step 4: Run focused tests and static checks**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_models.py -q
uv run ruff check backtesting/strategies/upbit_listing_event tests/strategies/upbit_listing_event/test_models.py
uv run mypy --ignore-missing-imports backtesting/strategies/upbit_listing_event/models.py
```

Expected: all model tests pass; ruff and mypy exit 0.

- [ ] **Step 5: Commit the domain contract**

```powershell
git add backtesting/strategies/upbit_listing_event/__init__.py backtesting/strategies/upbit_listing_event/models.py tests/strategies/upbit_listing_event/__init__.py tests/strategies/upbit_listing_event/test_models.py
git commit -m "Keep listing-event identity point-in-time safe" -m "Introduce immutable notice, coin-event, cost, match, and exclusion records before any external collection code." -m "Constraint: Announcement and scheduled-support events must remain independent coin observations." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: model pytest, ruff, mypy"
```

## Task 2: Parse Original Upbit Listing Content Without Revision Leakage

**Files:**
- Create: `backtesting/strategies/upbit_listing_event/upbit_parser.py`
- Create: `tests/strategies/upbit_listing_event/fixtures/upbit_trade_page.json`
- Create: `tests/strategies/upbit_listing_event/fixtures/upbit_prom_detail.json`
- Create: `tests/strategies/upbit_listing_event/fixtures/upbit_dos_updated_detail.json`
- Create: `tests/strategies/upbit_listing_event/fixtures/upbit_multi_asset_detail.json`
- Create: `tests/strategies/upbit_listing_event/test_upbit_parser.py`

- [ ] **Step 1: Save minimal real-shape fixtures and write failing parser tests**

The trade-page fixture must contain a valid `success/data/total_pages/total_count/notices` object with PROM, DOS, a multi-asset listing notice, and a non-listing trade warning. Preserve the observed ISO timestamps and fields `id`, `uuid`, `title`, `category`, `first_listed_at`, and `listed_at`.

The PROM detail fixture body must contain this exact listing table:

```html
<table><thead><tr><th>?붿????먯궛</th><th>留덉폆</th><th>?ㅽ듃?뚰겕</th><th>嫄곕옒吏??媛쒖떆 ?쒖젏</th></tr></thead>
<tbody><tr><td>?꾨＼(PROM)</td><td>KRW, USDT</td><td>Ethereum</td><td>8??12??12??30遺??덉젙</td></tr></tbody></table>
```

The DOS fixture must preserve update blocks for `15:00 -> 16:00` and `14:00 -> 15:00`, followed by the original table whose scheduled text is `8??11??14???덉젙`. The multi-asset fixture must contain separate rows for CYS, ICNT, XAN, EDEN, AIOZ, and ALLO.

Create `tests/strategies/upbit_listing_event/test_upbit_parser.py`:

```python
import json
from pathlib import Path

import pytest

from backtesting.strategies.upbit_listing_event.models import SourceConfidence
from backtesting.strategies.upbit_listing_event.upbit_parser import (
    UpbitSchemaError,
    parse_notice_assets,
    parse_notice_page,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_notice_page_preserves_first_and_latest_listing_times() -> None:
    page = parse_notice_page(_fixture("upbit_trade_page.json"), requested_page=1)
    dos = next(item for item in page.notices if item.notice_uuid == "1478681589")

    assert page.total_pages == 38
    assert dos.first_listed_at.isoformat() == "2026-08-11T11:30:36+09:00"
    assert dos.listed_at.isoformat() == "2026-08-11T14:55:12+09:00"


def test_unmodified_notice_extracts_original_schedule() -> None:
    assets = parse_notice_assets(_fixture("upbit_prom_detail.json"))

    assert len(assets) == 1
    assert assets[0].upbit_ticker == "PROM"
    assert assets[0].upbit_markets == ("KRW", "USDT")
    assert assets[0].original_scheduled_at.isoformat() == "2026-08-12T12:30:00+09:00"
    assert assets[0].source_confidence is SourceConfidence.UNMODIFIED_CURRENT


def test_updated_notice_parses_preserved_original_not_latest_schedule() -> None:
    assets = parse_notice_assets(_fixture("upbit_dos_updated_detail.json"))

    assert assets[0].upbit_ticker == "DOS"
    assert assets[0].original_scheduled_at.isoformat() == "2026-08-11T14:00:00+09:00"
    assert assets[0].original_scheduled_text == "8??11??14???덉젙"
    assert assets[0].source_confidence is SourceConfidence.RECONSTRUCTED_ORIGINAL


def test_earliest_pre_revision_snapshot_is_labeled_first_seen() -> None:
    assets = parse_notice_assets(
        _fixture("upbit_prom_detail.json"),
        preserved_first_seen=True,
    )

    assert assets[0].source_confidence is SourceConfidence.FIRST_SEEN_SNAPSHOT


def test_multi_asset_notice_emits_one_independent_asset_row_per_coin() -> None:
    assets = parse_notice_assets(_fixture("upbit_multi_asset_detail.json"))

    assert [asset.upbit_ticker for asset in assets] == [
        "CYS", "ICNT", "XAN", "EDEN", "AIOZ", "ALLO"
    ]


def test_malformed_success_payload_fails_closed() -> None:
    with pytest.raises(UpbitSchemaError, match="total_pages"):
        parse_notice_page({"success": True, "data": {"notices": []}}, requested_page=1)
```

- [ ] **Step 2: Run the parser tests and verify they fail**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_upbit_parser.py -q
```

Expected: FAIL because `upbit_parser.py` does not exist.

- [ ] **Step 3: Implement strict list/detail parsing and original-body recovery**

Create `backtesting/strategies/upbit_listing_event/upbit_parser.py` with these public contracts:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
from typing import Any
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from .models import NoticeAsset, NoticeSummary, SourceConfidence


KST = ZoneInfo("Asia/Seoul")
LISTING_TITLE_PATTERNS = (
    re.compile(r"?좉퇋\s+嫄곕옒吏??s+?덈궡"),
    re.compile(r"留덉폆\s+?붿???s+?먯궛\s+異붽?"),
    re.compile(r"?곸옣"),
)
UPDATE_SEPARATOR = re.compile(r"(?:^|\n)---(?:\n|$)")
TICKER_PATTERN = re.compile(r"\(([A-Z0-9]{2,20})\)")
SCHEDULE_PATTERN = re.compile(
    r"(?P<month>\d{1,2})??s*(?P<day>\d{1,2})??s*"
    r"(?P<hour>\d{1,2})???:\s*(?P<minute>\d{1,2})遺??\s*?덉젙"
)


class UpbitSchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedNoticePage:
    total_pages: int
    total_count: int
    notices: tuple[NoticeSummary, ...]


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UpbitSchemaError(f"{context} must be an object")
    return value


def _aware_iso(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise UpbitSchemaError(f"{field} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise UpbitSchemaError(f"{field} must be timezone-aware")
    return parsed


def parse_notice_page(payload: object, *, requested_page: int) -> ParsedNoticePage:
    root = _mapping(payload, "notice page")
    if root.get("success") is not True:
        raise UpbitSchemaError("notice page success must be true")
    data = _mapping(root.get("data"), "notice page data")
    total_pages = data.get("total_pages")
    total_count = data.get("total_count")
    notices_raw = data.get("notices")
    if not isinstance(total_pages, int) or total_pages < requested_page:
        raise UpbitSchemaError("total_pages is invalid")
    if not isinstance(total_count, int) or total_count < 0:
        raise UpbitSchemaError("total_count is invalid")
    if not isinstance(notices_raw, list):
        raise UpbitSchemaError("notices must be an array")
    notices: list[NoticeSummary] = []
    for index, raw in enumerate(notices_raw):
        item = _mapping(raw, f"notice {index}")
        notices.append(
            NoticeSummary(
                notice_id=int(item["id"]),
                notice_uuid=str(item["uuid"]),
                title=str(item["title"]),
                category=str(item["category"]),
                first_listed_at=_aware_iso(item["first_listed_at"], "first_listed_at"),
                listed_at=_aware_iso(item["listed_at"], "listed_at"),
            )
        )
    return ParsedNoticePage(total_pages, total_count, tuple(notices))


def is_listing_title(title: str) -> bool:
    return any(pattern.search(title) for pattern in LISTING_TITLE_PATTERNS)


def _original_body(body: str, revised: bool) -> tuple[str, SourceConfidence]:
    if not revised:
        return body, SourceConfidence.UNMODIFIED_CURRENT
    sections = [section.strip() for section in UPDATE_SEPARATOR.split(body) if section.strip()]
    originals = [section for section in sections if "嫄곕옒吏??媛쒖떆 ?쒖젏" in section and "<table" in section]
    if len(originals) != 1:
        return "", SourceConfidence.UNRESOLVED_ORIGINAL
    return originals[0], SourceConfidence.RECONSTRUCTED_ORIGINAL


def _schedule(text: str, first_listed_at: datetime) -> datetime | None:
    match = SCHEDULE_PATTERN.fullmatch(" ".join(text.split()))
    if match is None:
        return None
    year = first_listed_at.year
    parsed = datetime(
        year,
        int(match.group("month")),
        int(match.group("day")),
        int(match.group("hour")),
        int(match.group("minute") or 0),
        tzinfo=KST,
    )
    if parsed < first_listed_at and first_listed_at.month == 12 and parsed.month == 1:
        parsed = parsed.replace(year=year + 1)
    if parsed < first_listed_at:
        return None
    return parsed


def parse_notice_assets(
    payload: object,
    *,
    preserved_first_seen: bool = False,
) -> tuple[NoticeAsset, ...]:
    root = _mapping(payload, "notice detail")
    if root.get("success") is not True:
        raise UpbitSchemaError("notice detail success must be true")
    data = _mapping(root.get("data"), "notice detail data")
    summary = NoticeSummary(
        notice_id=int(data["id"]),
        notice_uuid=str(data["uuid"]),
        title=str(data["title"]),
        category=str(data["category"]),
        first_listed_at=_aware_iso(data["first_listed_at"], "first_listed_at"),
        listed_at=_aware_iso(data["listed_at"], "listed_at"),
    )
    if summary.category != "嫄곕옒" or not is_listing_title(summary.title):
        return ()
    body = data.get("body")
    if not isinstance(body, str):
        raise UpbitSchemaError("notice detail body must be text")
    source, confidence = _original_body(body, summary.listed_at != summary.first_listed_at)
    schedule_allowed = confidence is not SourceConfidence.UNRESOLVED_ORIGINAL
    if not source:
        source = body
    if preserved_first_seen and summary.listed_at == summary.first_listed_at:
        confidence = SourceConfidence.FIRST_SEEN_SNAPSHOT
    soup = BeautifulSoup(source, "html.parser")
    listing_tables = [
        table
        for table in soup.find_all("table")
        if "嫄곕옒吏??媛쒖떆 ?쒖젏" in table.get_text(" ", strip=True)
    ]
    if len(listing_tables) != 1:
        raise UpbitSchemaError(
            f"expected one original listing table, got {len(listing_tables)}"
        )
    table = listing_tables[0]
    headers = [" ".join(cell.get_text(" ", strip=True).split()) for cell in table.find_all("th")]
    required = ["?붿????먯궛", "留덉폆", "?ㅽ듃?뚰겕", "嫄곕옒吏??媛쒖떆 ?쒖젏"]
    if not all(label in headers for label in required):
        raise UpbitSchemaError("listing table headers changed")
    body_rows = table.find("tbody")
    if body_rows is None:
        raise UpbitSchemaError("listing table body is missing")
    snapshot_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    assets: list[NoticeAsset] = []
    for row in body_rows.find_all("tr", recursive=False):
        values = [" ".join(cell.get_text(" ", strip=True).split()) for cell in row.find_all(["td", "th"], recursive=False)]
        if len(values) != 4:
            raise UpbitSchemaError("listing row must contain four cells")
        asset_text, markets_text, network, schedule_text = values
        ticker_match = TICKER_PATTERN.search(asset_text)
        if ticker_match is None:
            raise UpbitSchemaError(f"asset ticker is missing from {asset_text!r}")
        ticker = ticker_match.group(1)
        name = asset_text[: ticker_match.start()].strip()
        assets.append(
            NoticeAsset(
                notice_id=summary.notice_id,
                notice_uuid=summary.notice_uuid,
                title=summary.title,
                asset_name=name,
                upbit_ticker=ticker,
                upbit_markets=tuple(part.strip() for part in markets_text.split(",") if part.strip()),
                network=network or None,
                first_listed_at=summary.first_listed_at,
                listed_at=summary.listed_at,
                original_scheduled_at=(
                    _schedule(schedule_text, summary.first_listed_at)
                    if schedule_allowed
                    else None
                ),
                original_scheduled_text=schedule_text if schedule_allowed else None,
                source_confidence=confidence,
                detail_snapshot_sha256=snapshot_hash,
            )
        )
    return tuple(assets)
```

Import `json` for canonical snapshot hashing. Add tests for naive timestamps, backward non-rollover schedules, missing headers, non-listing titles, and an updated body with two plausible original sections but one unambiguous listing table. That last case must emit the asset with `UNRESOLVED_ORIGINAL` and no scheduled event rather than losing the eligible announcement event.

- [ ] **Step 4: Run parser tests and lint**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_upbit_parser.py -q
uv run ruff check backtesting/strategies/upbit_listing_event/upbit_parser.py tests/strategies/upbit_listing_event/test_upbit_parser.py
```

Expected: all parser tests pass and ruff exits 0.

- [ ] **Step 5: Commit original-content parsing**

```powershell
git add backtesting/strategies/upbit_listing_event/upbit_parser.py tests/strategies/upbit_listing_event/fixtures tests/strategies/upbit_listing_event/test_upbit_parser.py
git commit -m "Prevent revised listing times from entering historical events" -m "Parse strict Upbit list/detail schemas, split preserved original bodies from update blocks, and emit one row per listed coin." -m "Constraint: Historical detail responses expose the current body rather than an immutable first-publication body." -m "Rejected: Use the first time-like string in the body | update blocks appear above the original listing table" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: Upbit parser pytest and ruff"
```

## Task 3: Collect and Cache Upbit JSON Through Playwright

**Files:**
- Create: `backtesting/strategies/upbit_listing_event/upbit_client.py`
- Create: `tests/strategies/upbit_listing_event/test_upbit_client.py`

- [ ] **Step 1: Write failing tests for pagination, immutable cache, retry, and throttling**

Create a fake async transport with `get_json(url, timeout_ms)` and `close()` methods. Tests must prove:

```python
def test_fetch_all_notices_follows_reported_pages_and_deduplicates_uuid(tmp_path):
    transport = FakeTransport(page_payloads={1: page_one, 2: page_two})
    client = UpbitClient(transport, cache_dir=tmp_path, min_delay=0, max_attempts=1)
    notices = asyncio.run(client.fetch_all_trade_notices())
    assert [notice.notice_uuid for notice in notices] == ["u1", "u2", "u3"]
    assert transport.requested_pages == [1, 2]


def test_detail_refresh_creates_hash_named_snapshot_without_overwrite(tmp_path):
    transport = FakeTransport(details=[original_detail, revised_detail])
    client = UpbitClient(transport, cache_dir=tmp_path, min_delay=0, max_attempts=1)
    first = asyncio.run(client.fetch_detail_snapshots("u1", refresh=True))
    second = asyncio.run(client.fetch_detail_snapshots("u1", refresh=True))
    assert first[0] != second[-1]
    assert first[0].exists() and second[-1].exists()
    assert len(list((tmp_path / "details" / "u1").glob("*.json"))) == 2


def test_valid_cached_page_avoids_network_request(tmp_path):
    transport = FakeTransport(page_payloads={1: page_one})
    client = UpbitClient(transport, cache_dir=tmp_path, min_delay=0, max_attempts=1)
    asyncio.run(client.fetch_notice_page(1))
    asyncio.run(client.fetch_notice_page(1))
    assert transport.requested_pages == [1]
```

Add explicit tests for corrupt hashes, retry count, request timeout forwarding, a schema failure that does not write a valid manifest, and categories returning code `trade`.

- [ ] **Step 2: Run the client tests and verify they fail**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_upbit_client.py -q
```

Expected: FAIL importing `upbit_client`.

- [ ] **Step 3: Implement the Playwright transport and restartable cache**

Create `upbit_client.py` using the same lifecycle pattern as `kind/client.py`:

```python
UPBIT_NOTICE_URL = "https://www.upbit.com/service_center/notice"
UPBIT_CATEGORIES_URL = "https://pub-info.upbit.com/api/v1/categories?os=web"
UPBIT_LIST_URL = (
    "https://pub-info.upbit.com/api/v1/announcements"
    "?os=web&page={page}&per_page=20&category=trade"
)
UPBIT_DETAIL_URL = "https://pub-info.upbit.com/api/v1/announcements/{uuid}"
CACHE_SCHEMA_VERSION = 1


class JsonTransport(Protocol):
    async def get_json(self, url: str, *, timeout_ms: int) -> object: ...


class PlaywrightJsonTransport:
    async def __aenter__(self) -> "PlaywrightJsonTransport":
        self._playwright = await async_playwright().start()
        self._request = await self._playwright.request.new_context()
        response = await self._request.get(UPBIT_NOTICE_URL)
        if not response.ok:
            raise RuntimeError(f"Upbit bootstrap failed with HTTP {response.status}")
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._request.dispose()
        await self._playwright.stop()

    async def get_json(self, url: str, *, timeout_ms: int) -> object:
        response = await self._request.get(url, timeout=timeout_ms)
        if not response.ok:
            raise RuntimeError(f"Upbit GET failed with HTTP {response.status}")
        return await response.json()
```

`UpbitClient` must provide:

```python
async def fetch_categories(self, *, refresh: bool = False) -> tuple[dict[str, str], ...]
async def fetch_notice_page(self, page: int, *, refresh: bool = False) -> ParsedNoticePage
async def fetch_all_trade_notices(self, *, refresh: bool = False) -> tuple[NoticeSummary, ...]
async def fetch_detail_snapshots(self, uuid: str, *, refresh: bool = False) -> tuple[Path, ...]
```

Use `_atomic_write_bytes` with `Path.replace`, SHA-256 manifests, a single async `RateLimiter`, three bounded attempts, and backoff `min(2 ** (attempt - 1), 4)`. Page cache paths are `pages/page-0001.json` plus `pages/page-0001.manifest.json`. Detail snapshots are `details/{uuid}/{sha256}.json` plus one manifest per hash. The snapshot manifest includes `fetched_at_utc`, and `fetch_detail_snapshots` returns every valid snapshot ordered by that field. `fetch_all_trade_notices` validates that category code `trade` exists, follows the first page's `total_pages`, preserves page order, rejects duplicate UUIDs, and reconciles discovered rows against the API total count only after all pages are loaded.

- [ ] **Step 4: Run client tests, parser tests, and static checks**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_upbit_client.py tests/strategies/upbit_listing_event/test_upbit_parser.py -q
uv run ruff check backtesting/strategies/upbit_listing_event/upbit_client.py tests/strategies/upbit_listing_event/test_upbit_client.py
uv run mypy --ignore-missing-imports backtesting/strategies/upbit_listing_event/upbit_client.py
```

Expected: all tests pass; ruff and mypy exit 0.

- [ ] **Step 5: Commit the Playwright collection boundary**

```powershell
git add backtesting/strategies/upbit_listing_event/upbit_client.py tests/strategies/upbit_listing_event/test_upbit_client.py
git commit -m "Make Upbit notice history restartable and auditable" -m "Collect trade-category JSON through a bootstrapped Playwright request context and cache hash-verified list pages and immutable detail snapshots." -m "Constraint: Multi-page public collection must survive interruption without silently accepting partial responses." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: client/parser pytest, ruff, mypy"
```

## Task 4: Resolve Historical Binance Perpetual Symbols Conservatively

**Files:**
- Create: `backtesting/strategies/upbit_listing_event/matching.py`
- Create: `tests/strategies/upbit_listing_event/test_matching.py`

- [ ] **Step 1: Write failing exact, multiplier, ambiguous, and missing-match tests**

Create `test_matching.py`:

```python
from backtesting.strategies.upbit_listing_event.matching import resolve_symbol
from backtesting.strategies.upbit_listing_event.models import ExclusionReason


def test_exact_usdt_perpetual_is_preferred() -> None:
    result = resolve_symbol("AIOZ", {"AIOZUSDT", "1000AIOZUSDT"}, aliases={})
    assert result.symbol == "AIOZUSDT"
    assert result.reason is None


def test_multiplier_contract_requires_source_controlled_alias() -> None:
    unresolved = resolve_symbol("SHIB", {"1000SHIBUSDT"}, aliases={})
    resolved = resolve_symbol(
        "SHIB", {"1000SHIBUSDT"}, aliases={"SHIB": "1000SHIBUSDT"}
    )
    assert unresolved.reason is ExclusionReason.BINANCE_SYMBOL_UNRESOLVED
    assert resolved.symbol == "1000SHIBUSDT"


def test_alias_must_point_to_available_symbol() -> None:
    result = resolve_symbol("LUNA", {"LUNA2USDT"}, aliases={"LUNA": "LUNAUSDT"})
    assert result.symbol is None
    assert result.reason is ExclusionReason.BINANCE_SYMBOL_UNRESOLVED
```

Add cases for lowercase input rejection, exact symbol plus alias conflict, and two candidate rebrand symbols.

- [ ] **Step 2: Run matching tests and verify they fail**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_matching.py -q
```

Expected: FAIL importing `matching`.

- [ ] **Step 3: Implement deterministic exact/alias matching**

Create `matching.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Set

from .models import ExclusionReason, SymbolMatch


VERIFIED_MULTIPLIER_ALIASES: dict[str, str] = {
    "SHIB": "1000SHIBUSDT",
    "BONK": "1000BONKUSDT",
    "PEPE": "1000PEPEUSDT",
}


def resolve_symbol(
    ticker: str,
    available_symbols: Set[str],
    *,
    aliases: Mapping[str, str] = VERIFIED_MULTIPLIER_ALIASES,
) -> SymbolMatch:
    if not ticker or ticker != ticker.upper() or not ticker.isalnum():
        return SymbolMatch(ticker, None, ExclusionReason.BINANCE_SYMBOL_UNRESOLVED)
    exact = f"{ticker}USDT"
    if exact in available_symbols:
        return SymbolMatch(ticker, exact, None)
    alias = aliases.get(ticker)
    if alias is not None and alias in available_symbols:
        return SymbolMatch(ticker, alias, None)
    return SymbolMatch(ticker, None, ExclusionReason.BINANCE_SYMBOL_UNRESOLVED)
```

Do not add a guessed alias. Each committed alias must be accompanied by a fixture/test proving the Binance contract represents the same base asset and the Upbit event-period candle exists.

- [ ] **Step 4: Run tests and lint**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_matching.py -q
uv run ruff check backtesting/strategies/upbit_listing_event/matching.py tests/strategies/upbit_listing_event/test_matching.py
```

Expected: PASS and ruff exit 0.

- [ ] **Step 5: Commit conservative symbol resolution**

```powershell
git add backtesting/strategies/upbit_listing_event/matching.py tests/strategies/upbit_listing_event/test_matching.py
git commit -m "Keep Binance identity mismatches out of listing returns" -m "Resolve exact USDT symbols first and require explicit verified mappings for multiplier contracts." -m "Rejected: Fuzzy ticker matching | rebrands and multiplier contracts can represent different economic assets" -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: matching pytest and ruff"
```

## Task 5: Retrieve Checksum-Verified Binance USD-M One-Minute Data

**Files:**
- Create: `backtesting/strategies/upbit_listing_event/binance_client.py`
- Create: `tests/strategies/upbit_listing_event/fixtures/binance_prom_1m.csv`
- Create: `tests/strategies/upbit_listing_event/fixtures/binance_btcusdt_1m.csv`
- Create: `tests/strategies/upbit_listing_event/test_binance_client.py`

- [ ] **Step 1: Write failing archive, checksum, REST fallback, and schema tests**

The CSV fixtures must use Binance USD-M kline column order:

```text
open_time,open,high,low,close,volume,close_time,quote_volume,trade_count,taker_buy_base_volume,taker_buy_quote_volume,ignore
```

Create tests that build an in-memory ZIP from the fixture and prove:

```python
def test_parse_kline_zip_preserves_utc_open_times_and_numeric_columns():
    frame = parse_kline_zip(zip_bytes, symbol="PROMUSDT")
    assert str(frame.index.tz) == "UTC"
    assert frame.index.is_monotonic_increasing
    assert frame.loc[pd.Timestamp("2026-08-12T02:12:00Z"), "open"] == 2.0


def test_checksum_mismatch_is_rejected():
    with pytest.raises(BinanceSchemaError, match="checksum"):
        verify_checksum(b"archive", "0" * 64 + "  archive.zip")


def test_archive_404_uses_bounded_fapi_fallback(tmp_path):
    transport = FakeTransport(archive_status=404, klines=rest_rows)
    client = BinanceClient(transport, cache_dir=tmp_path)
    frame = asyncio.run(client.load_day("PROMUSDT", date(2026, 8, 12)))
    assert len(frame) == len(rest_rows)
    assert transport.rest_calls == 1


def test_duplicate_or_non_minute_open_times_fail_closed():
    with pytest.raises(BinanceSchemaError, match="open time"):
        parse_kline_rows(rows_with_duplicate_minute, symbol="PROMUSDT")
```

Add tests for checksum file grammar, ZIP containing the wrong filename, 12-column enforcement, numeric conversion, UTC-day boundary selection, current `exchangeInfo` filtering to `contractType == PERPETUAL`, `quoteAsset == USDT`, and `status == TRADING`, plus cache-hit behavior.

- [ ] **Step 2: Run Binance client tests and verify they fail**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_binance_client.py -q
```

Expected: FAIL importing `binance_client`.

- [ ] **Step 3: Implement archive-first historical retrieval**

Create `binance_client.py` with these constants and public functions:

```python
BINANCE_ARCHIVE_ROOT = "https://data.binance.vision/data/futures/um/daily/klines"
BINANCE_FAPI_ROOT = "https://fapi.binance.com"
KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
)


class BinanceSchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PublicResponse:
    status: int
    body: bytes


class PublicTransport(Protocol):
    async def get(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        timeout: float,
    ) -> PublicResponse:
        raise NotImplementedError


class HttpxPublicTransport:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(follow_redirects=True)

    async def __aenter__(self) -> "HttpxPublicTransport":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._client.aclose()

    async def get(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        timeout: float,
    ) -> PublicResponse:
        response = await self._client.get(url, params=params, timeout=timeout)
        return PublicResponse(response.status_code, response.content)


def verify_checksum(content: bytes, checksum_text: str) -> None:
    parts = checksum_text.strip().split()
    if len(parts) != 2 or len(parts[0]) != 64:
        raise BinanceSchemaError("checksum manifest grammar changed")
    if hashlib.sha256(content).hexdigest() != parts[0].lower():
        raise BinanceSchemaError("archive checksum mismatch")


def parse_kline_rows(rows: list[list[object]], *, symbol: str) -> pd.DataFrame:
    if any(len(row) != len(KLINE_COLUMNS) for row in rows):
        raise BinanceSchemaError("kline row must contain 12 fields")
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    numeric = [column for column in KLINE_COLUMNS if column not in {"open_time", "close_time", "ignore"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    frame = frame.set_index("open_time").sort_index()
    if frame.index.has_duplicates or not frame.index.floor("min").equals(frame.index):
        raise BinanceSchemaError("kline open time is duplicate or not minute-aligned")
    frame["symbol"] = symbol
    return frame
```

`parse_kline_zip(content, symbol)` must require exactly one CSV member named `{symbol}-1m-YYYY-MM-DD.csv`, decode UTF-8, accept an optional header row, and delegate to `parse_kline_rows`.

`BinanceClient` must expose:

```python
async def current_perpetual_symbols(self, *, refresh: bool = False) -> frozenset[str]
async def symbol_exists_on_day(self, symbol: str, day: date) -> bool
async def load_day(self, symbol: str, day: date, *, refresh: bool = False) -> pd.DataFrame
async def load_window(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame
```

Use the official daily URL `{root}/{symbol}/1m/{symbol}-1m-{YYYY-MM-DD}.zip` and its `.CHECKSUM`. Cache the exact ZIP, checksum, and SHA-256 manifest before parsing. When the archive returns 404, call `/fapi/v1/klines` with `interval=1m`, inclusive `startTime`, exclusive day converted to `endTime - 1`, and pages of at most 1,000 rows until the day is complete. Do not REST-fallback for checksum or schema failures. Merge all UTC dates intersecting `[start, end]`, reject duplicate open times, and slice inclusively for requested candle lookup.

`symbol_exists_on_day` first checks a valid cached/archive file for the exact candidate symbol and date, then uses the recent-data REST fallback only for an archive 404. Historical symbol availability is the union of current filtered `exchangeInfo` evidence and exact symbols whose official archive or bounded REST data contains the required day. Current `onboardDate` alone must not exclude a historical contract because Binance has documented archive/metadata inconsistencies.

- [ ] **Step 4: Run Binance tests and static checks**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_binance_client.py tests/strategies/upbit_listing_event/test_matching.py -q
uv run ruff check backtesting/strategies/upbit_listing_event/binance_client.py tests/strategies/upbit_listing_event/test_binance_client.py
uv run mypy --ignore-missing-imports backtesting/strategies/upbit_listing_event/binance_client.py
```

Expected: all tests pass; ruff and mypy exit 0.

- [ ] **Step 5: Commit auditable historical market data**

```powershell
git add backtesting/strategies/upbit_listing_event/binance_client.py tests/strategies/upbit_listing_event/fixtures/binance_* tests/strategies/upbit_listing_event/test_binance_client.py
git commit -m "Make historical short prices independently verifiable" -m "Load official Binance USD-M daily one-minute archives with published checksum verification and use a bounded public REST fallback only when an archive is not yet available." -m "Constraint: Current exchange metadata is not a complete record of delisted historical contracts." -m "Rejected: Trust onboardDate as sole history evidence | official archive history can disagree with current metadata" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: Binance client/matching pytest, ruff, mypy"
```

## Task 6: Calculate Causal Entry, Forward Short Returns, MFE, and MAE

**Files:**
- Create: `backtesting/strategies/upbit_listing_event/study.py`
- Create: `tests/strategies/upbit_listing_event/test_study.py`

- [ ] **Step 1: Write failing timing and P&L tests**

Create `test_study.py`:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtesting.strategies.upbit_listing_event.models import CostAssumptions
from backtesting.strategies.upbit_listing_event.study import (
    calculate_horizon,
    entry_minute,
    summarize_forward_returns,
)


KST = ZoneInfo("Asia/Seoul")


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (datetime(2026, 8, 12, 11, 11, 38, tzinfo=KST), "2026-08-12T02:12:00+00:00"),
        (datetime(2026, 8, 12, 12, 30, 0, tzinfo=KST), "2026-08-12T03:30:00+00:00"),
    ],
)
def test_entry_minute_rounds_up_without_delaying_exact_minutes(event, expected):
    assert entry_minute(event).isoformat() == expected


def test_horizon_uses_exact_opens_and_half_open_path_window():
    index = pd.date_range("2026-08-12T02:11:00Z", periods=7, freq="min")
    asset = pd.DataFrame(
        {
            "open": [2.00, 2.00, 1.95, 1.90, 1.85, 1.80, 1.75],
            "high": [2.01, 2.03, 2.02, 1.94, 1.90, 1.86, 1.80],
            "low": [1.99, 1.94, 1.88, 1.82, 1.78, 1.70, 1.69],
            "quote_volume": [1000.0] * 7,
            "trade_count": [100] * 7,
        },
        index=index,
    )
    btc = asset.assign(open=[100, 100, 101, 102, 103, 104, 105])

    row = calculate_horizon(
        asset,
        btc,
        entry_at=pd.Timestamp("2026-08-12T02:12:00Z"),
        horizon_minutes=5,
        costs=CostAssumptions(5.0, 5.0),
    )

    assert row["entry_price"] == 2.0
    assert row["exit_price"] == 1.75
    assert row["gross_short_return"] == pytest.approx(0.125)
    assert row["fee_only_short_return"] == pytest.approx(0.124)
    assert row["net_short_return"] == pytest.approx(0.123)
    assert row["mfe"] == pytest.approx((2.0 - 1.70) / 2.0)
    assert row["mae"] == pytest.approx((2.0 - 2.03) / 2.0)
    assert row["btc_relative_short_return"] == pytest.approx(0.05 - (-0.125))


def test_summary_is_descriptive_and_contains_no_inference_columns():
    forward = pd.DataFrame(
        {
            "event_type": ["ANNOUNCEMENT"] * 3,
            "horizon_minutes": [5] * 3,
            "net_short_return": [0.03, -0.01, 0.02],
            "mfe": [0.04, 0.01, 0.03],
            "mae": [-0.01, -0.03, -0.02],
            "btc_relative_short_return": [0.02, -0.02, 0.01],
        }
    )
    result = summarize_forward_returns(forward)
    assert result.loc[0, "events"] == 3
    assert result.loc[0, "win_rate"] == pytest.approx(2 / 3)
    assert "p_value" not in result.columns
    assert "t_stat" not in result.columns
```

Add tests for missing pre-event candle, missing entry candle, a missing 24-hour exit that preserves shorter horizons, exact interval `[entry, exit)`, zero prices, duplicate candle times, and two event types from one asset.

- [ ] **Step 2: Run study tests and verify they fail**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_study.py -q
```

Expected: FAIL importing `study`.

- [ ] **Step 3: Implement event creation and return math**

Create `study.py` with:

```python
HORIZONS = (5, 15, 30, 60, 240, 1440)


def build_events(asset: NoticeAsset) -> tuple[EventRecord, ...]:
    events = [
        EventRecord(asset.notice_uuid, asset.upbit_ticker, EventType.ANNOUNCEMENT, asset.first_listed_at)
    ]
    if asset.original_scheduled_at is not None:
        events.append(
            EventRecord(
                asset.notice_uuid,
                asset.upbit_ticker,
                EventType.SCHEDULED_SUPPORT,
                asset.original_scheduled_at,
            )
        )
    return tuple(events)


def entry_minute(event_at: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(event_at).tz_convert("UTC")
    return timestamp if timestamp == timestamp.floor("min") else timestamp.ceil("min")


def calculate_horizon(
    asset: pd.DataFrame,
    btc: pd.DataFrame,
    *,
    entry_at: pd.Timestamp,
    horizon_minutes: int,
    costs: CostAssumptions,
) -> dict[str, float | int | pd.Timestamp]:
    pre_event = entry_at - pd.Timedelta(minutes=1)
    exit_at = entry_at + pd.Timedelta(minutes=horizon_minutes)
    if pre_event not in asset.index:
        raise MissingCandle(ExclusionReason.PRE_EVENT_CANDLE_MISSING, pre_event)
    if entry_at not in asset.index:
        raise MissingCandle(ExclusionReason.ENTRY_CANDLE_MISSING, entry_at)
    if exit_at not in asset.index:
        raise MissingCandle(ExclusionReason.EXIT_CANDLE_MISSING, exit_at)
    if entry_at not in btc.index or exit_at not in btc.index:
        raise MissingCandle(ExclusionReason.EXIT_CANDLE_MISSING, exit_at)
    entry_price = float(asset.at[entry_at, "open"])
    exit_price = float(asset.at[exit_at, "open"])
    btc_entry = float(btc.at[entry_at, "open"])
    btc_exit = float(btc.at[exit_at, "open"])
    if min(entry_price, exit_price, btc_entry, btc_exit) <= 0:
        raise ValueError("entry and exit prices must be positive")
    path = asset.loc[(asset.index >= entry_at) & (asset.index < exit_at)]
    gross = (entry_price - exit_price) / entry_price
    asset_long = exit_price / entry_price - 1.0
    btc_long = btc_exit / btc_entry - 1.0
    return {
        "entry_at": entry_at,
        "exit_at": exit_at,
        "horizon_minutes": horizon_minutes,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_short_return": gross,
        "fee_only_short_return": gross - costs.fee_round_trip,
        "net_short_return": gross - costs.total_round_trip,
        "mfe": (entry_price - float(path["low"].min())) / entry_price,
        "mae": (entry_price - float(path["high"].max())) / entry_price,
        "btc_relative_short_return": btc_long - asset_long,
        "entry_quote_volume": float(asset.at[entry_at, "quote_volume"]),
        "window_quote_volume": float(path["quote_volume"].sum()),
        "window_trade_count": int(path["trade_count"].sum()),
    }
```

Define `MissingCandle` with `reason` and `timestamp` fields. Implement `calculate_event` to calculate all six horizons independently and return `(valid_rows, exclusions)`, preserving short horizons when a long exit is missing. Implement `summarize_forward_returns` grouped by `event_type/horizon_minutes`, returning exact design columns: count, mean, median, win rate, p10/p25/p75/p90, worst, best, mean MFE, mean MAE, sum of equal-notional returns, and mean BTC-relative return. Implement yearly and liquidity summaries without inferential fields.

- [ ] **Step 4: Run math tests and static checks**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_study.py -q
uv run ruff check backtesting/strategies/upbit_listing_event/study.py tests/strategies/upbit_listing_event/test_study.py
uv run mypy --ignore-missing-imports backtesting/strategies/upbit_listing_event/study.py
```

Expected: all tests pass; ruff and mypy exit 0.

- [ ] **Step 5: Commit causal event-study math**

```powershell
git add backtesting/strategies/upbit_listing_event/study.py tests/strategies/upbit_listing_event/test_study.py
git commit -m "Measure short profit only after each public event" -m "Round event timestamps causally, require exact surrounding candles, and compute cost-adjusted forward return, MFE, MAE, liquidity, and BTC-relative diagnostics." -m "Constraint: Missing exact candles must not shift entry or exit to a favorable later price." -m "Rejected: Search forward for the next candle | that changes the approved entry rule and hides outages" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: study pytest, ruff, mypy"
```

## Task 7: Orchestrate Discovery, Exclusions, Market Data, and Reconciliation

**Files:**
- Create: `backtesting/strategies/upbit_listing_event/pipeline.py`
- Create: `tests/strategies/upbit_listing_event/test_pipeline.py`

- [ ] **Step 1: Write a failing cached end-to-end pipeline test**

Use fake Upbit and Binance clients so no network is needed:

```python
def test_pipeline_keeps_multi_coin_events_and_materializes_every_exclusion(tmp_path):
    upbit = FakeUpbitClient(
        notices=(multi_notice, warning_notice),
        details={multi_notice.notice_uuid: multi_detail_path},
    )
    binance = FakeBinanceClient(
        available={"AIOZUSDT", "ALLOUSDT", "BTCUSDT"},
        frames={"AIOZUSDT": aioz_frame, "ALLOUSDT": allo_frame, "BTCUSDT": btc_frame},
    )
    result = asyncio.run(
        run_pipeline(
            upbit_client=upbit,
            binance_client=binance,
            cache_root=tmp_path / "raw",
            costs=CostAssumptions(),
        )
    )
    assert set(result.events["ticker"]) == {"AIOZ", "ALLO"}
    assert set(result.events["event_type"]) == {"ANNOUNCEMENT", "SCHEDULED_SUPPORT"}
    assert "NOT_A_LISTING" in set(result.exclusions["reason"])
    assert "BINANCE_SYMBOL_UNRESOLVED" in set(result.exclusions["reason"])
    assert len(result.forward_returns) == 2 * 2 * 6
```

Add tests that reconcile candidate counts, continue after one detail schema error, emit `ORIGINAL_SCHEDULE_UNRESOLVED` without removing an eligible announcement event, prevent duplicate event identities, and bound concurrent detail/market requests.

- [ ] **Step 2: Run the pipeline tests and verify they fail**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_pipeline.py -q
```

Expected: FAIL importing `pipeline`.

- [ ] **Step 3: Implement a dependency-injected pipeline**

Create `pipeline.py`:

```python
@dataclass(frozen=True, slots=True)
class PipelineResult:
    notice_assets: pd.DataFrame
    events: pd.DataFrame
    exclusions: pd.DataFrame
    forward_returns: pd.DataFrame
    horizon_summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    liquidity_summary: pd.DataFrame


async def run_pipeline(
    *,
    upbit_client: UpbitClientProtocol,
    binance_client: BinanceClientProtocol,
    cache_root: Path,
    costs: CostAssumptions,
    concurrency: int = 4,
    refresh_upbit: bool = False,
    refresh_binance: bool = False,
) -> PipelineResult:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
```

The implementation sequence is fixed:

1. Fetch every trade notice and classify listing-title candidates.
2. Fetch candidate detail snapshots behind one semaphore. Parse the earliest valid snapshot. When later snapshots exist and the earliest has `listed_at == first_listed_at`, pass `preserved_first_seen=True`; convert every parse failure to an exclusion.
3. Emit announcement events for assets with unambiguous identity; emit scheduled events only with an original schedule.
4. Reject duplicate `(notice_uuid, ticker, event_type)` identities.
5. Resolve exact/verified Binance symbols using current metadata plus exact-candidate `symbol_exists_on_day` probes.
6. Compute the minimum UTC window from `entry_time - 1 minute` through `entry_time + 1,440 minutes` and load each unique symbol/window once.
7. Load BTCUSDT for the union window once.
8. Calculate every horizon independently; retain horizon-specific missing-candle exclusions.
9. Reconcile discovered trade notices into `NOT_A_LISTING`, parsed candidates, or schema exclusions; reconcile every parsed asset into included event rows or explicit exclusions.
10. Sort all output frames deterministically by first event time, notice UUID, ticker, event type, and horizon.

Use explicit stable column constants for each DataFrame. `notice_assets` includes
`revision_detected = listed_at != first_listed_at`, both KST timestamps, original
schedule text/time, source confidence, and snapshot SHA-256. `events` repeats
`first_listed_at_kst` beside `event_at_kst/event_at_utc` so the point-in-time
audit in Task 10 is executable. Do not serialize inside `pipeline.py`; reporting
owns files.

- [ ] **Step 4: Run all package tests accumulated so far**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event -q
uv run ruff check backtesting/strategies/upbit_listing_event tests/strategies/upbit_listing_event
uv run mypy --ignore-missing-imports backtesting/strategies/upbit_listing_event
```

Expected: all package tests pass; ruff and mypy exit 0.

- [ ] **Step 5: Commit pipeline reconciliation**

```powershell
git add backtesting/strategies/upbit_listing_event/pipeline.py tests/strategies/upbit_listing_event/test_pipeline.py
git commit -m "Make every discovered listing candidate auditable" -m "Orchestrate point-in-time parsing, historical symbol proof, deduplicated market windows, horizon calculations, and explicit exclusions with population reconciliation." -m "Constraint: Partial source failures may write inspectable results but may not silently reduce the candidate population." -m "Confidence: high" -m "Scope-risk: broad" -m "Tested: full package pytest, ruff, mypy"
```

## Task 8: Write Profit-First Data, Workbook, Markdown, and Charts

**Files:**
- Create: `backtesting/strategies/upbit_listing_event/report.py`
- Create: `tests/strategies/upbit_listing_event/test_report.py`

- [ ] **Step 1: Write failing artifact and no-inference tests**

Create `test_report.py`:

```python
def test_write_report_bundle_creates_all_contract_artifacts(tmp_path):
    write_report_bundle(sample_result, tmp_path, CostAssumptions(5.0, 5.0))
    expected = {
        "notice_assets.csv", "events.csv", "exclusions.csv", "forward_returns.csv",
        "horizon_summary.csv", "yearly_summary.csv", "liquidity_summary.csv",
        "upbit_listing_event_study.xlsx", "report.md", "event_return_paths.png",
        "horizon_net_returns.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})


def test_markdown_leads_with_net_profit_and_contains_no_inference_language(tmp_path):
    write_report_bundle(sample_result, tmp_path, CostAssumptions(5.0, 5.0))
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "?됯퇏 ?쒖닆?섏씡瑜? in text
    assert "?뺣났 鍮꾩슜 媛?? 20.0 bps" in text
    assert "p-value" not in text
    assert "t-stat" not in text
    assert "confidence interval" not in text


def test_excel_preserves_timezone_as_iso_text_and_numeric_returns(tmp_path):
    write_report_bundle(sample_result, tmp_path, CostAssumptions())
    forward = pd.read_excel(tmp_path / "upbit_listing_event_study.xlsx", sheet_name="FORWARD_RETURNS")
    assert forward["entry_at_utc"].iloc[0].endswith("+00:00")
    assert pd.api.types.is_numeric_dtype(forward["net_short_return"])
```

Add deterministic CSV order, empty-exclusion sheet, nonempty chart, and best-horizon sorting tests.

- [ ] **Step 2: Run report tests and verify they fail**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_report.py -q
```

Expected: FAIL importing `report`.

- [ ] **Step 3: Implement the deterministic report bundle**

Create `report.py` with:

```python
CSV_OUTPUTS = {
    "notice_assets.csv": "notice_assets",
    "events.csv": "events",
    "exclusions.csv": "exclusions",
    "forward_returns.csv": "forward_returns",
    "horizon_summary.csv": "horizon_summary",
    "yearly_summary.csv": "yearly_summary",
    "liquidity_summary.csv": "liquidity_summary",
}


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        if isinstance(safe[column].dtype, pd.DatetimeTZDtype):
            safe[column] = safe[column].map(lambda value: value.isoformat() if pd.notna(value) else "")
    return safe


def write_report_bundle(
    result: PipelineResult,
    output_dir: Path,
    costs: CostAssumptions,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, attribute in CSV_OUTPUTS.items():
        getattr(result, attribute).to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "upbit_listing_event_study.xlsx", engine="openpyxl") as writer:
        for filename, attribute in CSV_OUTPUTS.items():
            sheet = filename.removesuffix(".csv").upper()[:31]
            _excel_safe(getattr(result, attribute)).to_excel(writer, sheet_name=sheet, index=False)
    _write_markdown(result, output_dir / "report.md", costs)
    _write_horizon_chart(result.horizon_summary, output_dir / "horizon_net_returns.png")
    _write_event_path_chart(result.forward_returns, output_dir / "event_return_paths.png")
```

The Markdown order is mandatory: cost assumptions, sample/exclusion counts, best mean net short horizon by event type, complete horizon table, worst-case table, MFE/MAE, liquidity diagnostics, BTC-relative diagnostics, source confidence counts, limitations. `total_equal_notional_return` must be described as a sum of independent one-unit opportunities, not as compounded portfolio return.

The horizon chart compares mean and median net returns for the two event types. The event-path chart uses the six discrete horizons and includes interquartile bands; it must not interpolate a false continuous minute path. Use Matplotlib `Agg` before importing `pyplot`.

- [ ] **Step 4: Run report and package tests**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event/test_report.py tests/strategies/upbit_listing_event/test_pipeline.py -q
uv run ruff check backtesting/strategies/upbit_listing_event/report.py tests/strategies/upbit_listing_event/test_report.py
```

Expected: PASS and ruff exit 0.

- [ ] **Step 5: Commit profit-first reporting**

```powershell
git add backtesting/strategies/upbit_listing_event/report.py tests/strategies/upbit_listing_event/test_report.py
git commit -m "Put realizable short profit at the front of event results" -m "Write deterministic CSV, Excel, Markdown, and chart artifacts centered on cost-adjusted return, loss tails, MFE, MAE, and liquidity." -m "Constraint: This release is descriptive and must not imply statistical significance or portfolio compounding." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: report/pipeline pytest and ruff"
```

## Task 9: Add the CLI, Operator Guide, and Gated Live Smoke Check

**Files:**
- Create: `scripts/run_upbit_listing_event_study.py`
- Create: `tests/scripts/test_run_upbit_listing_event_study.py`
- Create: `tests/strategies/upbit_listing_event/test_live_smoke.py`
- Create: `backtesting/strategies/upbit_listing_event/README.md`

- [ ] **Step 1: Write failing CLI and live-smoke boundary tests**

Create `tests/scripts/test_run_upbit_listing_event_study.py`:

```python
from pathlib import Path

from scripts.run_upbit_listing_event_study import parse_args


def test_parse_args_exposes_only_approved_operational_controls() -> None:
    args = parse_args(
        [
            "--output-dir", "out", "--cache-dir", "cache",
            "--fee-bps-per-side", "4", "--slippage-bps-per-side", "7",
            "--concurrency", "3", "--min-delay", "0.8", "--timeout", "20",
            "--refresh-upbit", "--refresh-binance",
        ]
    )
    assert args.output_dir == Path("out")
    assert args.cache_dir == Path("cache")
    assert args.fee_bps_per_side == 4.0
    assert args.slippage_bps_per_side == 7.0
    assert args.concurrency == 3
    assert args.refresh_upbit and args.refresh_binance
```

Create `test_live_smoke.py` with a module-level skip unless `RUN_LIVE_UPBIT_LISTING_SMOKE=1`. The single test opens `PlaywrightJsonTransport`, validates the category list contains `trade`, fetches page 1, fetches its first listing detail, then uses `BinanceClient` to load one bounded BTCUSDT UTC day. It asserts schemas and closes every client in `finally`/async context managers.

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```powershell
uv run pytest tests/scripts/test_run_upbit_listing_event_study.py tests/strategies/upbit_listing_event/test_live_smoke.py -q
```

Expected: CLI test fails because the script does not exist; live smoke is skipped by default.

- [ ] **Step 3: Implement the thin CLI and operator guide**

Create the script with this argument surface:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Upbit listing short event study.")
    parser.add_argument("--output-dir", type=Path, default=ROOT.results_path / "upbit_listing_event_study")
    parser.add_argument("--cache-dir", type=Path, default=ROOT.raw_path / "upbit_listing_event")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--min-delay", type=float, default=0.75)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--refresh-upbit", action="store_true")
    parser.add_argument("--refresh-binance", action="store_true")
    return parser.parse_args(argv)
```

`async_main` constructs one `PlaywrightJsonTransport`, one Upbit client, one httpx-backed Binance transport/client, calls `run_pipeline`, writes the report bundle, prints output path plus included/excluded counts, and returns 0 only after reconciliation succeeds. `main()` returns exit code 2 for `UpbitSchemaError`, `BinanceSchemaError`, or reconciliation failure after logging the specific error; unexpected errors propagate during tests.

The README must document:

```powershell
uv sync
uv run python -m playwright install chromium
uv run python scripts/run_upbit_listing_event_study.py
```

It must also explain both event anchors, first-publication reconstruction confidence, exact one-minute entry/exit rules, Binance archive checksums, default 20 bps round-trip cost, independent multi-coin observations, every output file, cache refresh behavior, exclusions, and explicitly deferred stops/targets/leverage.

- [ ] **Step 4: Run CLI/package checks and the opt-in live smoke**

Run deterministic checks:

```powershell
uv run pytest tests/strategies/upbit_listing_event tests/scripts/test_run_upbit_listing_event_study.py -q
uv run ruff check backtesting/strategies/upbit_listing_event scripts/run_upbit_listing_event_study.py tests/strategies/upbit_listing_event tests/scripts/test_run_upbit_listing_event_study.py
uv run mypy --ignore-missing-imports backtesting/strategies/upbit_listing_event scripts/run_upbit_listing_event_study.py
```

Expected: all deterministic tests pass; live smoke reports one skip; ruff and mypy exit 0.

Then run the explicit live integration check:

```powershell
$env:RUN_LIVE_UPBIT_LISTING_SMOKE='1'
uv run pytest tests/strategies/upbit_listing_event/test_live_smoke.py -q
Remove-Item Env:RUN_LIVE_UPBIT_LISTING_SMOKE
```

Expected: one live smoke test passes against current public Upbit and Binance boundaries.

- [ ] **Step 5: Commit the executable research surface**

```powershell
git add scripts/run_upbit_listing_event_study.py tests/scripts/test_run_upbit_listing_event_study.py tests/strategies/upbit_listing_event/test_live_smoke.py backtesting/strategies/upbit_listing_event/README.md
git commit -m "Make the listing event study reproducible from one command" -m "Expose only approved collection, cache, concurrency, and cost controls and document source confidence, outputs, and recovery." -m "Constraint: Live endpoints are integration evidence and remain gated outside deterministic test runs." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: deterministic suite, ruff, mypy, gated live Upbit/Binance smoke"
```

## Task 10: Run the Historical Study and Audit the Result Population

**Files:**
- Generated only: `raw/upbit_listing_event/`
- Generated only: `results/upbit_listing_event_study/`
- Modify if defects are found: the smallest relevant package/test file from Tasks 1-9

- [ ] **Step 1: Run the full historical collection and event study**

Run:

```powershell
uv run python scripts/run_upbit_listing_event_study.py `
  --output-dir results/upbit_listing_event_study `
  --cache-dir raw/upbit_listing_event
```

Expected: exit 0; 11 documented result artifacts exist; console prints included event and exclusion counts.

- [ ] **Step 2: Audit population reconciliation and point-in-time rules**

Run:

```powershell
@'
import pandas as pd
from pathlib import Path

root = Path("results/upbit_listing_event_study")
assets = pd.read_csv(root / "notice_assets.csv")
events = pd.read_csv(root / "events.csv")
excluded = pd.read_csv(root / "exclusions.csv")
forward = pd.read_csv(root / "forward_returns.csv")

assert not events.duplicated(["notice_uuid", "ticker", "event_type"]).any()
assert set(events["event_type"]) <= {"ANNOUNCEMENT", "SCHEDULED_SUPPORT"}
assert (events.loc[events["event_type"].eq("ANNOUNCEMENT"), "event_at_kst"] ==
        events.loc[events["event_type"].eq("ANNOUNCEMENT"), "first_listed_at_kst"]).all()
assert set(forward["horizon_minutes"]) <= {5, 15, 30, 60, 240, 1440}
assert forward["net_short_return"].notna().all()
assert excluded["reason"].notna().all()
print({"assets": len(assets), "events": len(events), "forward_rows": len(forward), "exclusions": len(excluded)})
'@ | uv run python -
```

Expected: assertions pass and counts print. Inspect all `RECONSTRUCTED_ORIGINAL` and `UNRESOLVED_ORIGINAL` rows in the workbook before accepting the study.

- [ ] **Step 3: Inspect profit sensitivity and exclusion concentration**

Run:

```powershell
@'
import pandas as pd
from pathlib import Path

root = Path("results/upbit_listing_event_study")
summary = pd.read_csv(root / "horizon_summary.csv")
excluded = pd.read_csv(root / "exclusions.csv")
print(summary.sort_values(["event_type", "mean_net_short_return"], ascending=[True, False]).to_string(index=False))
print(excluded.groupby("reason", dropna=False).size().sort_values(ascending=False).to_string())
'@ | uv run python -
```

Expected: both event types are visible where data permits; no single unexplained schema exclusion dominates without an accompanying raw-cache artifact.

- [ ] **Step 4: Run full verification after the live dataset is known**

Run:

```powershell
uv run pytest tests/strategies/upbit_listing_event tests/scripts/test_run_upbit_listing_event_study.py -q
uv run pytest -q
uv run ruff check .
uv run mypy --ignore-missing-imports backtesting/strategies/upbit_listing_event scripts/run_upbit_listing_event_study.py
git diff --check
```

Expected: targeted and full pytest pass; ruff, mypy, and diff check exit 0. If an unrelated pre-existing project failure remains, record the exact command and failure separately and prove the targeted suite passes.

- [ ] **Step 5: Commit only source/test fixes discovered by the live run**

Do not commit ignored raw caches or result artifacts. If no defect was found, skip this commit. If a defect was fixed, use:

```powershell
git add backtesting/strategies/upbit_listing_event scripts/run_upbit_listing_event_study.py tests/strategies/upbit_listing_event tests/scripts/test_run_upbit_listing_event_study.py
git commit -m "Make live listing evidence satisfy the research contract" -m "Resolve the smallest parser, market-data, or reconciliation defect exposed by the complete historical run and lock it with a regression test." -m "Constraint: Generated raw and result artifacts remain local and ignored." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: targeted suite, full pytest, ruff, mypy, historical run"
```

## Final Completion Checklist

- [ ] Both event anchors use original point-in-time values, never revised values.
- [ ] Multi-asset notices remain separate coin observations.
- [ ] Only verified Binance USD-M perpetual history enters return calculations.
- [ ] Entries and exits use exact one-minute opens; missing candles never shift.
- [ ] All six horizons report gross, fee-only, and net short returns.
- [ ] MFE, MAE, liquidity, and BTC-relative diagnostics are present.
- [ ] No stops, targets, leverage, p-values, t-statistics, confidence intervals, or clustered inference appear.
- [ ] Every omission has a stable exclusion reason and source audit trail.
- [ ] Raw responses/archives are hash or checksum verified and restartable.
- [ ] CSV, Excel, Markdown, and PNG artifacts match the design contract.
- [ ] Targeted tests, full tests, lint, types, live smoke, and historical run are verified before completion is claimed.
