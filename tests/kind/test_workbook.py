from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kind.workbook import (
    WorkbookSchemaError,
    _parse_announcement_date,
    _quarter_from_reference,
    read_timeframe,
)


WorkbookMutator = Callable[[list[list[object]]], None]


def _timeframe_rows() -> list[list[object]]:
    rows: list[list[object]] = [[None, None, None] for _ in range(16)]
    rows[8] = ["코드", "A005930", "A0126Z0"]
    rows[9] = ["코드명", "삼성전자", "삼성에피스홀딩스"]
    rows[10] = ["유형", "FSP-IFRS(M)", "FSP-IFRS(M)"]
    rows[11] = ["아이템코드", "FP56000500", "FP56000500"]
    rows[12] = ["아이템명", "잠정치발표일", "잠정치발표일"]
    rows[13] = ["집계주기", 6, 6]
    rows[14] = [pd.Timestamp("2024-03-29"), 20240430, 20240515]
    rows[15] = [pd.Timestamp("2024-06-28"), 20240731, None]
    return rows


def _write_workbook(
    path: Path,
    *,
    mutate: WorkbookMutator | None = None,
    include_timeframe: bool = True,
) -> Path:
    rows = _timeframe_rows()
    if mutate is not None:
        mutate(rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if include_timeframe:
            pd.DataFrame(rows).to_excel(
                writer, sheet_name="timeframe", header=False, index=False
            )
        pd.DataFrame([["dummy"]]).to_excel(
            writer, sheet_name="value", header=False, index=False
        )
    return path


def _append_inactive_column(
    rows: list[list[object]],
    *,
    ticker: object = "A000001",
    company: object = "미관측회사",
) -> None:
    for row in rows:
        row.append(None)
    rows[8][-1] = ticker
    rows[9][-1] = company
    rows[10][-1] = "FSP-IFRS(M)"
    rows[11][-1] = "FP56000500"
    rows[12][-1] = "잠정치발표일"
    rows[13][-1] = 6


def test_read_timeframe_normalizes_and_sorts_records(tmp_path: Path) -> None:
    path = _write_workbook(tmp_path / "announcements.xlsx")

    actual = read_timeframe(path)

    expected = pd.DataFrame(
        {
            "ticker": ["A005930", "A0126Z0", "A005930"],
            "company": ["삼성전자", "삼성에피스홀딩스", "삼성전자"],
            "quarter": ["2024Q1", "2024Q1", "2024Q2"],
            "announcement_date": pd.to_datetime(
                ["2024-04-30", "2024-05-15", "2024-07-31"]
            ),
        }
    )
    assert list(actual.columns) == [
        "ticker",
        "company",
        "quarter",
        "announcement_date",
    ]
    assert_frame_equal(actual, expected)
    assert pd.api.types.is_datetime64_any_dtype(actual["announcement_date"])


def test_read_timeframe_rejects_changed_item_name(tmp_path: Path) -> None:
    path = _write_workbook(
        tmp_path / "changed-item.xlsx",
        mutate=lambda rows: rows[12].__setitem__(1, "확정치발표일"),
    )

    with pytest.raises(WorkbookSchemaError, match="잠정치발표일"):
        read_timeframe(path)


def test_read_timeframe_rejects_empty_item_name(tmp_path: Path) -> None:
    path = _write_workbook(
        tmp_path / "empty-item.xlsx",
        mutate=lambda rows: rows[12].__setitem__(1, None),
    )

    with pytest.raises(WorkbookSchemaError, match="잠정치발표일"):
        read_timeframe(path)


def test_read_timeframe_requires_timeframe_sheet(tmp_path: Path) -> None:
    path = _write_workbook(
        tmp_path / "missing-sheet.xlsx", include_timeframe=False
    )

    with pytest.raises(WorkbookSchemaError, match="timeframe"):
        read_timeframe(path)


def test_read_timeframe_rejects_too_small_sheet(tmp_path: Path) -> None:
    path = tmp_path / "too-small.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([["only", "one"], ["short", "sheet"]]).to_excel(
            writer, sheet_name="timeframe", header=False, index=False
        )
        pd.DataFrame([["dummy"]]).to_excel(
            writer, sheet_name="value", header=False, index=False
        )

    with pytest.raises(WorkbookSchemaError, match="15 rows.*2 columns"):
        read_timeframe(path)


@pytest.mark.parametrize(
    ("row_index", "expected_label"),
    [
        (8, "코드"),
        (9, "코드명"),
        (10, "유형"),
        (11, "아이템코드"),
        (12, "아이템명"),
        (13, "집계주기"),
    ],
)
def test_read_timeframe_rejects_changed_required_row_label(
    tmp_path: Path, row_index: int, expected_label: str
) -> None:
    path = _write_workbook(
        tmp_path / f"changed-label-{row_index}.xlsx",
        mutate=lambda rows: rows[row_index].__setitem__(0, "unexpected"),
    )

    with pytest.raises(WorkbookSchemaError, match=expected_label):
        read_timeframe(path)


@pytest.mark.parametrize("bad_date", [20240230, "not-a-date"])
def test_read_timeframe_rejects_invalid_announcement_date(
    tmp_path: Path, bad_date: object
) -> None:
    path = _write_workbook(
        tmp_path / "invalid-date.xlsx",
        mutate=lambda rows: rows[14].__setitem__(1, bad_date),
    )

    with pytest.raises(WorkbookSchemaError, match="announcement date"):
        read_timeframe(path)


def test_read_timeframe_rejects_data_without_reference_date(
    tmp_path: Path,
) -> None:
    path = _write_workbook(
        tmp_path / "missing-reference-date.xlsx",
        mutate=lambda rows: rows[14].__setitem__(0, None),
    )

    with pytest.raises(WorkbookSchemaError, match="reference date"):
        read_timeframe(path)


@pytest.mark.parametrize(
    "reference_value",
    [
        datetime(2024, 3, 31),
        date(2024, 3, 31),
        pd.Timestamp("2024-03-31"),
    ],
)
def test_quarter_reference_accepts_only_naive_midnight_calendar_dates(
    reference_value: object,
) -> None:
    assert (
        _quarter_from_reference(reference_value, row_number=15) == "2024Q1"
    )


@pytest.mark.parametrize(
    "reference_value",
    [
        True,
        np.bool_(True),
        20240331,
        np.int64(43890),
        43890.0,
        "2024",
        "2024-03",
        "2024-03-31",
        "not-a-date",
        datetime(2024, 3, 31, 0, 0, tzinfo=timezone.utc),
        pd.Timestamp("2024-03-31", tz="Asia/Seoul"),
        datetime(2024, 3, 31, 12, 30),
        pd.Timestamp("2024-03-31 00:00:01"),
    ],
)
def test_quarter_reference_rejects_non_excel_calendar_date_values(
    reference_value: object,
) -> None:
    with pytest.raises(WorkbookSchemaError, match="row 15"):
        _quarter_from_reference(reference_value, row_number=15)


def test_read_timeframe_validates_nonblank_reference_on_empty_row(
    tmp_path: Path,
) -> None:
    def malformed_empty_row(rows: list[list[object]]) -> None:
        rows[15] = ["2024", None, None]

    path = _write_workbook(
        tmp_path / "malformed-empty-row.xlsx", mutate=malformed_empty_row
    )

    with pytest.raises(WorkbookSchemaError, match="reference date.*row 16"):
        read_timeframe(path)


def test_read_timeframe_skips_fully_blank_data_row(tmp_path: Path) -> None:
    def blank_row(rows: list[list[object]]) -> None:
        rows[15] = [None, None, None]

    path = _write_workbook(tmp_path / "blank-row.xlsx", mutate=blank_row)

    actual = read_timeframe(path)

    assert list(actual["quarter"]) == ["2024Q1", "2024Q1"]


@pytest.mark.parametrize(
    ("metadata_row", "bad_value", "message"),
    [
        (8, None, "ticker"),
        (8, "005930", "ticker"),
        (8, "A", "ticker"),
        (9, None, "company"),
        (9, "   ", "company"),
    ],
)
def test_read_timeframe_rejects_missing_or_invalid_observation_metadata(
    tmp_path: Path,
    metadata_row: int,
    bad_value: object,
    message: str,
) -> None:
    path = _write_workbook(
        tmp_path / f"invalid-metadata-{metadata_row}-{bad_value!s}.xlsx",
        mutate=lambda rows: rows[metadata_row].__setitem__(1, bad_value),
    )

    with pytest.raises(WorkbookSchemaError, match=message):
        read_timeframe(path)


def test_read_timeframe_rejects_duplicate_ticker_quarter_observations(
    tmp_path: Path,
) -> None:
    def duplicate_quarter(rows: list[list[object]]) -> None:
        rows[15][0] = pd.Timestamp("2024-03-31")
        rows[15][1] = 20240501

    path = _write_workbook(
        tmp_path / "duplicate-quarter.xlsx", mutate=duplicate_quarter
    )

    with pytest.raises(WorkbookSchemaError, match="duplicate.*ticker.*quarter"):
        read_timeframe(path)


@pytest.mark.parametrize(
    ("metadata_row", "bad_value", "message"),
    [
        (8, None, "ticker"),
        (9, None, "company"),
        (10, None, "FSP-IFRS"),
        (10, "OTHER", "FSP-IFRS"),
        (11, None, "FP56000500"),
        (11, "OTHER", "FP56000500"),
        (12, None, "잠정치발표일"),
        (12, "확정치발표일", "잠정치발표일"),
        (13, None, "집계주기"),
        (13, True, "집계주기"),
        (13, 3, "집계주기"),
        (13, "6.0", "집계주기"),
    ],
)
def test_read_timeframe_prevalidates_inactive_column_metadata(
    tmp_path: Path,
    metadata_row: int,
    bad_value: object,
    message: str,
) -> None:
    def invalid_inactive_metadata(rows: list[list[object]]) -> None:
        _append_inactive_column(rows)
        rows[metadata_row][-1] = bad_value

    path = _write_workbook(
        tmp_path / f"inactive-metadata-{metadata_row}.xlsx",
        mutate=invalid_inactive_metadata,
    )

    with pytest.raises(WorkbookSchemaError, match=message):
        read_timeframe(path)


@pytest.mark.parametrize(
    "bad_ticker",
    [
        "A12!456",
        "a005930",
        " A005930",
        "A005930 ",
        "A12345",
        "A1234567",
    ],
)
def test_read_timeframe_rejects_ticker_outside_exact_ascii_grammar(
    tmp_path: Path, bad_ticker: str
) -> None:
    def invalid_ticker(rows: list[list[object]]) -> None:
        _append_inactive_column(rows, ticker=bad_ticker)

    path = _write_workbook(
        tmp_path / "invalid-ticker.xlsx", mutate=invalid_ticker
    )

    with pytest.raises(WorkbookSchemaError, match=r"A\[0-9A-Z\]\{6\}"):
        read_timeframe(path)


def test_read_timeframe_rejects_duplicate_ticker_columns(tmp_path: Path) -> None:
    path = _write_workbook(
        tmp_path / "duplicate-ticker-column.xlsx",
        mutate=lambda rows: _append_inactive_column(rows, ticker="A005930"),
    )

    with pytest.raises(WorkbookSchemaError, match="duplicate ticker"):
        read_timeframe(path)


def test_read_timeframe_accepts_string_six_aggregation_period(
    tmp_path: Path,
) -> None:
    def string_period(rows: list[list[object]]) -> None:
        _append_inactive_column(rows)
        rows[13][-1] = "6"

    path = _write_workbook(tmp_path / "string-period.xlsx", mutate=string_period)

    actual = read_timeframe(path)

    assert len(actual) == 3


@pytest.mark.parametrize(
    "announcement_value",
    [
        20240430,
        np.int64(20240430),
        20240430.0,
        np.float64(20240430.0),
        "20240430",
    ],
)
def test_parse_announcement_date_accepts_exact_yyyymmdd_forms(
    announcement_value: object,
) -> None:
    assert _parse_announcement_date(
        announcement_value, row_number=15, column_number=2
    ) == pd.Timestamp("2024-04-30")


@pytest.mark.parametrize(
    "announcement_value",
    [
        True,
        np.bool_(True),
        20240430.5,
        np.float64(20240430.5),
        "2.024043e7",
        "２０２４０４３０",
        " 20240430",
        "20240430 ",
        "not-a-date",
        20240230,
    ],
)
def test_parse_announcement_date_rejects_invalid_forms_with_coordinate(
    announcement_value: object,
) -> None:
    with pytest.raises(
        WorkbookSchemaError, match="announcement date.*row 15, column 2"
    ):
        _parse_announcement_date(
            announcement_value, row_number=15, column_number=2
        )


def test_read_timeframe_reports_announcement_cell_coordinate(
    tmp_path: Path,
) -> None:
    path = _write_workbook(
        tmp_path / "invalid-date-coordinate.xlsx",
        mutate=lambda rows: rows[14].__setitem__(1, 20240230),
    )

    with pytest.raises(
        WorkbookSchemaError, match="announcement date.*row 15, column 2"
    ):
        read_timeframe(path)
