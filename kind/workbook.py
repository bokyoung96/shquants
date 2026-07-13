from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
import re
from typing import Final

import pandas as pd


class WorkbookSchemaError(ValueError):
    """Raised when the KIND workbook does not match the expected schema."""


EXPECTED_LABELS: Final[dict[int, str]] = {
    8: "코드",
    9: "코드명",
    10: "유형",
    11: "아이템코드",
    12: "아이템명",
    13: "집계주기",
}
EXPECTED_ITEM_NAME: Final = "잠정치발표일"
OUTPUT_COLUMNS: Final = [
    "ticker",
    "company",
    "quarter",
    "announcement_date",
]


def _parse_announcement_date(value: object) -> pd.Timestamp:
    if isinstance(value, bool):
        raise WorkbookSchemaError(f"invalid announcement date: {value!r}")

    if isinstance(value, Integral):
        text = str(int(value))
    elif isinstance(value, Real) and float(value).is_integer():
        text = str(int(value))
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise WorkbookSchemaError(f"invalid announcement date: {value!r}")

    if re.fullmatch(r"\d{8}", text) is None:
        raise WorkbookSchemaError(f"invalid announcement date: {value!r}")

    try:
        return pd.to_datetime(text, format="%Y%m%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise WorkbookSchemaError(
            f"invalid announcement date: {value!r}"
        ) from exc


def _quarter_from_reference(value: object, *, row_number: int) -> str:
    try:
        reference_date = pd.to_datetime(value, errors="raise")
    except (TypeError, ValueError) as exc:
        raise WorkbookSchemaError(
            f"invalid quarter reference date at row {row_number}: {value!r}"
        ) from exc

    if pd.isna(reference_date):
        raise WorkbookSchemaError(
            f"missing quarter reference date at row {row_number}"
        )
    return f"{reference_date.year}Q{reference_date.quarter}"


def _observation_metadata(
    frame: pd.DataFrame, *, column: int
) -> tuple[str, str]:
    ticker = frame.iat[8, column]
    if not isinstance(ticker, str) or re.fullmatch(r"A\S+", ticker) is None:
        raise WorkbookSchemaError(
            f"invalid ticker metadata in column {column + 1}: {ticker!r}"
        )

    company = frame.iat[9, column]
    if not isinstance(company, str) or not company.strip():
        raise WorkbookSchemaError(
            f"invalid company metadata in column {column + 1}: {company!r}"
        )
    return ticker, company.strip()


def read_timeframe(path: str | Path) -> pd.DataFrame:
    """Read and normalize the KIND wide timeframe sheet."""

    try:
        frame = pd.read_excel(path, sheet_name="timeframe", header=None)
    except ValueError as exc:
        if "timeframe" in str(exc):
            raise WorkbookSchemaError(
                "workbook is missing required 'timeframe' sheet"
            ) from exc
        raise

    if frame.shape[0] < 15 or frame.shape[1] < 2:
        raise WorkbookSchemaError(
            "timeframe sheet must contain at least 15 rows and 2 columns; "
            f"found {frame.shape[0]} rows and {frame.shape[1]} columns"
        )

    for row_index, expected_label in EXPECTED_LABELS.items():
        actual_label = frame.iat[row_index, 0]
        if actual_label != expected_label:
            raise WorkbookSchemaError(
                f"expected label {expected_label!r} at row {row_index + 1}; "
                f"found {actual_label!r}"
            )

    for column in range(1, frame.shape[1]):
        item_name = frame.iat[12, column]
        if pd.isna(item_name) or item_name != EXPECTED_ITEM_NAME:
            raise WorkbookSchemaError(
                f"expected item name {EXPECTED_ITEM_NAME!r} in column "
                f"{column + 1}; found {item_name!r}"
            )

    records: list[dict[str, object]] = []
    for row_index in range(14, frame.shape[0]):
        reference_value = frame.iat[row_index, 0]
        observation_values = frame.iloc[row_index, 1:]
        has_observation = observation_values.notna().any()

        if pd.isna(reference_value):
            if has_observation:
                raise WorkbookSchemaError(
                    f"missing quarter reference date at row {row_index + 1}"
                )
            continue
        if not has_observation:
            continue

        quarter = _quarter_from_reference(
            reference_value, row_number=row_index + 1
        )
        for column in range(1, frame.shape[1]):
            announcement_value = frame.iat[row_index, column]
            if pd.isna(announcement_value):
                continue
            ticker, company = _observation_metadata(frame, column=column)
            records.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "quarter": quarter,
                    "announcement_date": _parse_announcement_date(
                        announcement_value
                    ),
                }
            )

    result = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    result["announcement_date"] = pd.to_datetime(result["announcement_date"])

    duplicate_mask = result.duplicated(["ticker", "quarter"], keep=False)
    if duplicate_mask.any():
        duplicate = result.loc[duplicate_mask, ["ticker", "quarter"]].iloc[0]
        raise WorkbookSchemaError(
            "duplicate ticker-quarter observation for "
            f"ticker {duplicate['ticker']!r}, quarter {duplicate['quarter']!r}"
        )

    return result.sort_values(
        ["quarter", "ticker", "announcement_date"], kind="stable"
    ).reset_index(drop=True)
