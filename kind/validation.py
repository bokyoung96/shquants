from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from kind.models import Confidence


PRIMARY_COLUMNS = [
    "ticker",
    "company",
    "quarter",
    "announcement_date",
    "announcement_datetime",
    "confidence",
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
    _validate_primary_frame(frame, input_row_count=input_row_count)

    missing = frame.loc[frame["announcement_datetime"].isna()].copy()
    duplicate_mask = (
        frame.groupby(["ticker", "quarter"])["announcement_datetime"]
        .transform("nunique")
        .gt(1)
    )
    duplicates = frame.loc[duplicate_mask].copy()
    multiples = frame.loc[frame["confidence"].eq(Confidence.MULTIPLE_MATCH.value)].copy()

    missing.to_csv(log_dir / "missing_match.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(
        log_dir / "duplicate_match.csv",
        index=False,
        encoding="utf-8-sig",
    )
    multiples.to_csv(
        log_dir / "multiple_candidate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = ValidationSummary(
        input_row_count=input_row_count,
        output_row_count=len(frame),
        missing_match_count=len(missing),
        duplicate_match_count=len(duplicates),
        multiple_candidate_count=len(multiples),
    )
    (log_dir / "validation_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def write_schema_errors(
    errors: list[dict[str, object]],
    *,
    log_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    columns = ["announcement_date", "cache_page", "error_type", "message"]
    pd.DataFrame.from_records(errors, columns=columns).to_csv(
        log_dir / "schema_error.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_outputs(frame: pd.DataFrame, audit: pd.DataFrame, *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(
        ["quarter", "ticker", "announcement_date"],
        ignore_index=True,
    )

    csv_frame = ordered.copy()
    csv_frame["announcement_date"] = csv_frame["announcement_date"].dt.strftime(
        "%Y-%m-%d"
    )
    csv_frame["announcement_datetime"] = csv_frame[
        "announcement_datetime"
    ].dt.strftime("%Y-%m-%d %H:%M")
    csv_frame.to_csv(
        output_dir / "announcements_with_time.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(output_dir / "match_audit.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(
        output_dir / "announcements_with_time.xlsx",
        engine="openpyxl",
    ) as writer:
        ordered.to_excel(writer, sheet_name="announcements", index=False)
        audit.to_excel(writer, sheet_name="match_audit", index=False)


def _validate_primary_frame(frame: pd.DataFrame, *, input_row_count: int) -> None:
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
