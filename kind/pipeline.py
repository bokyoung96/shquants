from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from kind.client import KindClient, PlaywrightTransport
from kind.matching import match_disclosure, normalize_company_name
from kind.models import Confidence, Disclosure
from kind.parser import KindSchemaError, parse_disclosure_page
from kind.validation import (
    PRIMARY_COLUMNS,
    ValidationSummary,
    validate_and_write_reports,
    write_outputs,
    write_schema_errors,
)
from kind.workbook import read_timeframe


class DateFetcher(Protocol):
    async def fetch_date(self, date: str, *, refresh: bool = False) -> tuple[Path, ...]:
        ...


@dataclass(frozen=True, slots=True)
class PipelineResult:
    frame: pd.DataFrame
    audit: pd.DataFrame
    validation: ValidationSummary


async def fetch_dates_with_semaphore(
    client: DateFetcher,
    dates: list[str],
    *,
    concurrency: int,
    refresh: bool = False,
) -> dict[str, tuple[Path, ...]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def fetch_one(date: str) -> tuple[str, tuple[Path, ...]]:
        async with semaphore:
            return date, await client.fetch_date(date, refresh=refresh)

    pairs = await asyncio.gather(*(fetch_one(date) for date in dates))
    return dict(pairs)


def parse_cached_dates(
    cache_pages: dict[str, tuple[Path, ...]],
) -> tuple[dict[str, tuple[Disclosure, ...]], list[dict[str, object]]]:
    disclosures_by_date: dict[str, tuple[Disclosure, ...]] = {}
    schema_errors: list[dict[str, object]] = []

    for date, paths in cache_pages.items():
        disclosures: list[Disclosure] = []
        for page_number, path in enumerate(paths, start=1):
            try:
                html = path.read_text(encoding="utf-8")
                parsed = parse_disclosure_page(
                    html,
                    announcement_date=date,
                    page=page_number,
                )
                disclosures.extend(parsed.disclosures)
            except KindSchemaError as error:
                schema_errors.append(
                    {
                        "announcement_date": date,
                        "cache_page": str(path),
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
        disclosures_by_date[date] = tuple(disclosures)

    return disclosures_by_date, schema_errors


def match_all_rows(
    input_frame: pd.DataFrame,
    disclosures_by_date: dict[str, tuple[Disclosure, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []

    for row in input_frame.itertuples(index=False):
        date = pd.Timestamp(row.announcement_date).strftime("%Y-%m-%d")
        disclosures = disclosures_by_date.get(date, ())
        match = match_disclosure(row.ticker, row.company, disclosures)
        matched = match.confidence in {
            Confidence.EXACT_MATCH,
            Confidence.NORMALIZED_MATCH,
        }
        announcement_datetime = pd.NaT
        if matched and match.disclosure is not None:
            announcement_datetime = pd.Timestamp(f"{date} {match.disclosure.time}")

        records.append(
            {
                "ticker": row.ticker,
                "company": row.company,
                "quarter": row.quarter,
                "announcement_date": pd.Timestamp(row.announcement_date),
                "announcement_datetime": announcement_datetime,
                "confidence": match.confidence.value,
            }
        )
        audit_records.append(
            _audit_record(
                row=row,
                date=date,
                disclosures=disclosures,
                confidence=match.confidence,
                disclosure=match.disclosure,
                candidates=match.candidates,
                rejection_reason=match.rejection_reason,
            )
        )

    frame = pd.DataFrame.from_records(records, columns=PRIMARY_COLUMNS)
    frame["announcement_date"] = pd.to_datetime(frame["announcement_date"])
    frame["announcement_datetime"] = pd.to_datetime(frame["announcement_datetime"])
    audit = pd.DataFrame.from_records(audit_records)
    return frame, audit


async def run_pipeline(
    *,
    input_path: Path,
    client: DateFetcher,
    output_dir: Path,
    log_dir: Path,
    concurrency: int,
    refresh: bool = False,
) -> PipelineResult:
    input_frame = read_timeframe(input_path)
    unique_dates = sorted(input_frame["announcement_date"].dt.strftime("%Y-%m-%d").unique())
    cache_pages = await fetch_dates_with_semaphore(
        client,
        list(unique_dates),
        concurrency=concurrency,
        refresh=refresh,
    )
    disclosures_by_date, schema_errors = parse_cached_dates(cache_pages)
    result_frame, audit_frame = match_all_rows(input_frame, disclosures_by_date)
    summary = validate_and_write_reports(
        result_frame,
        input_row_count=len(input_frame),
        log_dir=log_dir,
    )
    write_schema_errors(schema_errors, log_dir=log_dir)
    write_outputs(result_frame, audit_frame, output_dir=output_dir)
    return PipelineResult(frame=result_frame, audit=audit_frame, validation=summary)


async def _run_with_playwright(args: argparse.Namespace) -> PipelineResult:
    async with PlaywrightTransport() as transport:
        client = KindClient(
            transport,
            cache_dir=args.cache_dir,
            min_delay=args.min_delay,
            timeout_seconds=args.timeout,
            max_attempts=args.max_attempts,
        )
        return await run_pipeline(
            input_path=args.input,
            client=client,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            concurrency=args.concurrency,
            refresh=args.refresh,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover KIND disclosure times for timeframe workbook rows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--min-delay", type=float, default=0.75)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)

    result = asyncio.run(_run_with_playwright(args))
    print(result.validation)
    return 0 if result.validation.is_strictly_complete else 2


def _audit_record(
    *,
    row: object,
    date: str,
    disclosures: tuple[Disclosure, ...],
    confidence: Confidence,
    disclosure: Disclosure | None,
    candidates: tuple[Disclosure, ...],
    rejection_reason: str | None,
) -> dict[str, object]:
    candidate_receipts = "|".join(candidate.receipt_id for candidate in candidates)
    candidate_titles = "|".join(candidate.title for candidate in candidates)
    candidate_pages = "|".join(
        f"{candidate.page}:{candidate.position}" for candidate in candidates
    )
    return {
        "ticker": getattr(row, "ticker"),
        "company": getattr(row, "company"),
        "normalized_company": normalize_company_name(getattr(row, "company")),
        "quarter": getattr(row, "quarter"),
        "announcement_date": date,
        "confidence": confidence.value,
        "announcement_time": disclosure.time if disclosure is not None else None,
        "receipt_id": disclosure.receipt_id if disclosure is not None else None,
        "issuer_id": disclosure.issuer_id if disclosure is not None else None,
        "page": disclosure.page if disclosure is not None else None,
        "position": disclosure.position if disclosure is not None else None,
        "candidate_count": len(candidates),
        "candidate_receipts": candidate_receipts,
        "candidate_titles": candidate_titles,
        "candidate_pages": candidate_pages,
        "date_disclosure_count": len(disclosures),
        "rejection_reason": rejection_reason,
    }


if __name__ == "__main__":
    raise SystemExit(main())
