from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kind.workbook import WorkbookSchemaError, read_timeframe


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
