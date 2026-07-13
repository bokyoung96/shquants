from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kind.models import Confidence
from kind.validation import (
    PRIMARY_COLUMNS,
    validate_and_write_reports,
    write_outputs,
    write_schema_errors,
)


def test_validation_reports_missing_duplicates_and_multiples(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A005930", "A005930", "A000660", "A0126Z0"],
            "company": ["삼성전자", "삼성전자", "SK하이닉스", "삼성에피스"],
            "quarter": ["2024Q1", "2024Q1", "2024Q1", "2024Q1"],
            "announcement_date": pd.to_datetime(
                ["2024-04-30", "2024-04-30", "2024-04-25", "2024-05-15"]
            ),
            "announcement_datetime": pd.to_datetime(
                ["2024-04-30 08:31", "2024-04-30 09:00", None, None]
            ),
            "confidence": [
                Confidence.EXACT_MATCH.value,
                Confidence.EXACT_MATCH.value,
                Confidence.NO_MATCH.value,
                Confidence.MULTIPLE_MATCH.value,
            ],
        },
        columns=PRIMARY_COLUMNS,
    )

    summary = validate_and_write_reports(frame, input_row_count=4, log_dir=tmp_path)

    assert summary.input_row_count == 4
    assert summary.output_row_count == 4
    assert summary.missing_match_count == 2
    assert summary.duplicate_match_count == 2
    assert summary.multiple_candidate_count == 1
    assert not summary.is_strictly_complete
    assert len(pd.read_csv(tmp_path / "missing_match.csv")) == 2
    assert len(pd.read_csv(tmp_path / "duplicate_match.csv")) == 2
    assert len(pd.read_csv(tmp_path / "multiple_candidate.csv")) == 1
    assert json.loads((tmp_path / "validation_summary.json").read_text("utf-8"))[
        "multiple_candidate_count"
    ] == 1


def test_validation_rejects_datetime_confidence_mismatches(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A005930"],
            "company": ["삼성전자"],
            "quarter": ["2024Q1"],
            "announcement_date": pd.to_datetime(["2024-04-30"]),
            "announcement_datetime": pd.to_datetime([None]),
            "confidence": [Confidence.EXACT_MATCH.value],
        },
        columns=PRIMARY_COLUMNS,
    )

    with pytest.raises(ValueError, match="matched row"):
        validate_and_write_reports(frame, input_row_count=1, log_dir=tmp_path)


def test_write_outputs_preserves_primary_csv_excel_and_audit(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A005930"],
            "company": ["삼성전자"],
            "quarter": ["2024Q1"],
            "announcement_date": pd.to_datetime(["2024-04-30"]),
            "announcement_datetime": pd.to_datetime(["2024-04-30 08:31"]),
            "confidence": [Confidence.EXACT_MATCH.value],
        },
        columns=PRIMARY_COLUMNS,
    )
    audit = pd.DataFrame(
        [{"ticker": "A005930", "quarter": "2024Q1", "receipt_id": "20240430000001"}]
    )

    write_outputs(frame, audit, output_dir=tmp_path)

    loaded = pd.read_excel(tmp_path / "announcements_with_time.xlsx")
    assert str(loaded["announcement_datetime"].dtype).startswith("datetime64")
    csv_text = (tmp_path / "announcements_with_time.csv").read_text("utf-8-sig")
    assert "2024-04-30 08:31" in csv_text
    assert pd.read_csv(tmp_path / "match_audit.csv").loc[0, "receipt_id"] == 20240430000001


def test_write_schema_errors_uses_stable_columns(tmp_path: Path) -> None:
    write_schema_errors(
        [
            {
                "announcement_date": "2024-04-25",
                "cache_page": "page-0001.html",
                "error_type": "KindSchemaError",
                "message": "bad html",
            }
        ],
        log_dir=tmp_path,
    )

    assert list(pd.read_csv(tmp_path / "schema_error.csv").columns) == [
        "announcement_date",
        "cache_page",
        "error_type",
        "message",
    ]
