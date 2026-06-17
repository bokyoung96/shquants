from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths
from etfs.refresh.holdings_refresh import (
    DataguideExcelRefreshDriver,
    RefreshTarget,
    extract_sheet_records,
    filter_refresh_targets,
    load_refresh_targets_from_ticker_workbook,
    refresh_targets_to_parquet_files,
    refresh_targets_to_parquet_files_batch,
    ticker_output_path,
    write_records_parquet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh filtered DataGuide ETF holdings into ticker-level parquet files.")
    parser.add_argument("--ticker-workbook", default=paths.REFRESH_TICKER_XLSX.as_posix())
    parser.add_argument("--template", default=paths.REFRESH_TEMPLATE_XLSX.as_posix())
    parser.add_argument("--output-dir", default=paths.REFRESHED_HOLDINGS_FILES_DIR.as_posix())
    parser.add_argument("--work-dir", default=paths.REFRESH_WORK_DIR.as_posix())
    parser.add_argument("--tickers", nargs="*", default=[], help="Optional ETF tickers to refresh, with or without A prefix.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=25, help="Ticker count per batch Excel refresh.")
    parser.add_argument("--all-rows", action="store_true", help="Use every ticker row instead of only visible filtered rows.")
    parser.add_argument("--force", action="store_true", help="Refresh tickers even when their parquet file already exists.")
    parser.add_argument("--hidden", action="store_true", help="Run Excel hidden. DataGuide6 refresh is more reliable visible.")
    parser.add_argument("--manifest", default="", help="Optional manifest path. Empty by default to avoid output clutter.")
    parser.add_argument("--keep-work", action="store_true", help="Keep transient Excel workbooks and refresh scripts for debugging.")
    parser.add_argument(
        "--mode",
        choices=["batch", "single"],
        default="batch",
        help="batch reuses one Excel workbook; single opens one workbook per ticker.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    targets = load_refresh_targets_from_ticker_workbook(
        Path(args.ticker_workbook),
        visible_only=not bool(args.all_rows),
    )
    targets = filter_refresh_targets(targets, tickers=args.tickers, limit=args.limit)
    index_code_by_etf = {target.etf_code: target.index_code for target in targets}

    skipped_existing: list[str] = []
    outputs: dict[str, Path] = {}
    remaining: list[RefreshTarget] = []
    for target in targets:
        output_path = ticker_output_path(output_dir, target.etf_code)
        if output_path.exists() and not args.force:
            skipped_existing.append(target.etf_code)
            continue
        remaining.append(target)

    if not args.force:
        recovered, remaining = _recover_existing_workbooks(
            targets=remaining,
            output_dir=output_dir,
            work_dir=work_dir,
            index_code_by_etf=index_code_by_etf,
        )
        outputs.update(recovered)

    driver = DataguideExcelRefreshDriver(work_dir=work_dir, visible=not bool(args.hidden))
    for chunk in _chunks(remaining, args.chunk_size):
        outputs.update(
            _refresh_chunk(
                chunk,
                template_path=Path(args.template),
                output_dir=output_dir,
                work_dir=work_dir,
                index_code_by_etf=index_code_by_etf,
                driver=driver,
                mode=args.mode,
            )
        )

    if not args.keep_work:
        _remove_default_work_dir(work_dir)
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "refresh_mode": args.mode,
                    "chunk_size": args.chunk_size,
                    "requested_ticker_count": len(targets),
                    "refreshed_or_recovered_count": len(outputs),
                    "skipped_existing_count": len(skipped_existing),
                    "skipped_existing": skipped_existing,
                    "outputs": {ticker: path.as_posix() for ticker, path in outputs.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"wrote {len(outputs)} parquet file(s) under {output_dir}")
    if args.manifest:
        print(f"wrote {Path(args.manifest)}")
    return 0


def _refresh_chunk(
    targets: list[RefreshTarget],
    *,
    template_path: Path,
    output_dir: Path,
    work_dir: Path,
    index_code_by_etf: Mapping[str, str],
    driver: DataguideExcelRefreshDriver,
    mode: str,
) -> dict[str, Path]:
    if not targets:
        return {}
    if mode == "single":
        return refresh_targets_to_parquet_files(
            targets=targets,
            template_path=template_path,
            output_dir=output_dir,
            index_code_by_etf=index_code_by_etf,
            driver=driver,
        )
    try:
        return refresh_targets_to_parquet_files_batch(
            targets=targets,
            template_path=template_path,
            output_dir=output_dir,
            index_code_by_etf=index_code_by_etf,
            driver=driver,
        )
    except RuntimeError:
        recovered, missing = _recover_existing_workbooks(
            targets=targets,
            output_dir=output_dir,
            work_dir=work_dir,
            index_code_by_etf=index_code_by_etf,
        )
        if not missing:
            return recovered
        recovered.update(
            refresh_targets_to_parquet_files(
                targets=missing,
                template_path=template_path,
                output_dir=output_dir,
                index_code_by_etf=index_code_by_etf,
                driver=driver,
            )
        )
        return recovered


def _recover_existing_workbooks(
    *,
    targets: Iterable[RefreshTarget],
    output_dir: Path,
    work_dir: Path,
    index_code_by_etf: Mapping[str, str],
) -> tuple[dict[str, Path], list[RefreshTarget]]:
    outputs: dict[str, Path] = {}
    missing: list[RefreshTarget] = []
    for target in targets:
        workbook_path = _batch_workbook_path(work_dir, target)
        if not workbook_path.exists():
            missing.append(target)
            continue
        try:
            records = extract_sheet_records(workbook_path, index_code_by_etf=index_code_by_etf)
        except Exception:
            missing.append(target)
            continue
        etf_codes = {str(record.get("etf_code", "")).strip() for record in records}
        etf_codes.discard("")
        if etf_codes != {target.etf_code}:
            missing.append(target)
            continue
        output_path = ticker_output_path(output_dir, target.etf_code)
        write_records_parquet(records, output_path)
        outputs[target.etf_code] = output_path
    return outputs, missing


def _batch_workbook_path(work_dir: Path, target: RefreshTarget) -> Path:
    return work_dir / "workbooks" / f"holdings_{_safe_name(target.etf_code)}.xlsx"


def _chunks(targets: list[RefreshTarget], size: int) -> Iterable[list[RefreshTarget]]:
    for offset in range(0, len(targets), size):
        yield targets[offset : offset + size]


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _remove_default_work_dir(work_dir: Path) -> None:
    if not work_dir.exists():
        return
    resolved_work_dir = work_dir.resolve()
    resolved_refresh_dir = paths.REFRESH_DIR.resolve()
    if resolved_work_dir == resolved_refresh_dir or resolved_refresh_dir not in resolved_work_dir.parents:
        return
    for attempt in range(3):
        try:
            shutil.rmtree(resolved_work_dir)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
