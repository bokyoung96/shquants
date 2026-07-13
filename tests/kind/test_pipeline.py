from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from kind.pipeline import (
    fetch_dates_with_semaphore,
    parse_cached_dates,
    run_pipeline,
)
from kind.validation import PRIMARY_COLUMNS
from kind.workbook import (
    EXPECTED_ITEM_CODE,
    EXPECTED_ITEM_NAME,
    EXPECTED_LABELS,
    EXPECTED_TYPE,
)
from tests.kind.test_client import _page_html


class CachedClient:
    def __init__(self, pages_by_date: dict[str, tuple[Path, ...]]) -> None:
        self.pages_by_date = pages_by_date
        self.calls: list[str] = []

    async def fetch_date(self, date: str, *, refresh: bool = False) -> tuple[Path, ...]:
        self.calls.append(date)
        return self.pages_by_date[date]


class TrackingClient:
    def __init__(self, page: Path) -> None:
        self.page = page
        self.active = 0
        self.max_active = 0

    async def fetch_date(self, date: str, *, refresh: bool = False) -> tuple[Path, ...]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return (self.page,)


def test_run_pipeline_matches_cached_pages_and_writes_outputs(tmp_path: Path) -> None:
    workbook = _write_pipeline_workbook(tmp_path / "announcements.xlsx")
    cache_page = _write_cache_page(
        tmp_path,
        "2024-04-30",
        _page_html([_row(company="SK하이닉스", issuer="00066")], current=1, total=1),
    )
    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"
    client = CachedClient({"2024-04-30": (cache_page,)})

    result = asyncio.run(
        run_pipeline(
            input_path=workbook,
            client=client,
            output_dir=output_dir,
            log_dir=log_dir,
            concurrency=1,
        )
    )

    assert len(result.frame) == 2
    assert list(result.frame.columns) == PRIMARY_COLUMNS
    assert list(result.frame["confidence"]) == ["EXACT_MATCH", "NO_MATCH"]
    assert result.frame.loc[0, "announcement_datetime"] == pd.Timestamp(
        "2024-04-30 08:05"
    )
    assert pd.isna(result.frame.loc[1, "announcement_datetime"])
    assert len(result.audit) == 2
    assert (log_dir / "missing_match.csv").exists()
    assert (log_dir / "schema_error.csv").exists()
    assert (output_dir / "announcements_with_time.xlsx").exists()
    assert client.calls == ["2024-04-30"]


def test_fetch_dates_with_semaphore_limits_concurrency(tmp_path: Path) -> None:
    page = _write_cache_page(
        tmp_path,
        "2024-04-30",
        _page_html([_row()], current=1, total=1),
    )
    client = TrackingClient(page)

    result = asyncio.run(
        fetch_dates_with_semaphore(
            client,
            ["2024-04-30", "2024-05-01", "2024-05-02"],
            concurrency=2,
        )
    )

    assert set(result) == {"2024-04-30", "2024-05-01", "2024-05-02"}
    assert client.max_active == 2


def test_schema_error_keeps_date_available_for_no_match(tmp_path: Path) -> None:
    broken = tmp_path / "broken.html"
    broken.write_text("<html>not KIND</html>", encoding="utf-8")

    disclosures, errors = parse_cached_dates({"2024-04-30": (broken,)})

    assert disclosures["2024-04-30"] == ()
    assert errors == [
        {
            "announcement_date": "2024-04-30",
            "cache_page": str(broken),
            "error_type": "KindSchemaError",
            "message": errors[0]["message"],
        }
    ]


def _write_pipeline_workbook(path: Path) -> Path:
    rows: list[list[object]] = [[None, None, None] for _ in range(15)]
    for index, label in EXPECTED_LABELS.items():
        rows[index][0] = label
    rows[8][1:] = ["A000660", "A005930"]
    rows[9][1:] = ["SK하이닉스", "삼성전자"]
    rows[10][1:] = [EXPECTED_TYPE, EXPECTED_TYPE]
    rows[11][1:] = [EXPECTED_ITEM_CODE, EXPECTED_ITEM_CODE]
    rows[12][1:] = [EXPECTED_ITEM_NAME, EXPECTED_ITEM_NAME]
    rows[13][1:] = [6, 6]
    rows[14] = [pd.Timestamp("2024-03-29"), 20240430, 20240430]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(
            writer,
            sheet_name="timeframe",
            header=False,
            index=False,
        )
        pd.DataFrame([["dummy"]]).to_excel(
            writer,
            sheet_name="value",
            header=False,
            index=False,
        )
    return path


def _write_cache_page(tmp_path: Path, date: str, html: str) -> Path:
    date_dir = tmp_path / "cache" / date
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / "page-0001.html"
    path.write_text(html, encoding="utf-8")
    return path


def _row(
    *,
    time: str = "08:05",
    company: str = "삼성전자",
    issuer: str = "00593",
    title: str = "영업 (잠정) 실적 (공정공시)",
    receipt_id: str = "20240430000001",
) -> str:
    return f"""
        <tr>
          <td>{time}</td>
          <td><a href="#" onclick="companysummary_open('{issuer}');return false;">{company}</a></td>
          <td><a href="#" onclick="openDisclsViewer('{receipt_id}','')">{title}</a></td>
          <td>{company}</td>
          <td>-</td>
        </tr>
    """
