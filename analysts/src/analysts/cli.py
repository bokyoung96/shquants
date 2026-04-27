from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Sequence

from .config import build_config
from .graphify import GraphifyCorpusBuilder
from .pipeline import ArasPipeline
from .raw_reports import RawReportCatalog
from .sources.gmail.client import GmailApiClient
from .sources.gmail.pipeline import GmailSourcePipeline
from .sources.gmail.storage import GmailStore
from .sources.gmail.web_capture import PlaywrightWebCapturer
from .sources.telegram.client import auth_login as telegram_auth_login
from .sources.telegram.fetcher import TelegramFetcher
from .sources.telegram.watcher import AsyncWatchResult, WatchUntilRunner
from .storage import SqliteArasStore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysts.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config = subparsers.add_parser("show-config")
    show_config.add_argument("--base-dir", default=".")

    auth = subparsers.add_parser("auth-login")
    auth.add_argument("--base-dir", default=".")

    run_once = subparsers.add_parser("run-once")
    run_once.add_argument("--channel", required=True)
    run_once.add_argument("--base-dir", default=".")
    run_once.add_argument("--fixtures")

    summarize_latest = subparsers.add_parser("summarize-latest")
    summarize_latest.add_argument("--channel", required=True)
    summarize_latest.add_argument("--base-dir", default=".")

    summarize_recent = subparsers.add_parser("summarize-recent")
    summarize_recent.add_argument("--channel", required=True)
    summarize_recent.add_argument("--limit", type=int, default=10)
    summarize_recent.add_argument("--base-dir", default=".")

    watch_until = subparsers.add_parser("watch-until")
    watch_until.add_argument("--channel", action="append", required=True)
    watch_until.add_argument("--until", required=True)
    watch_until.add_argument("--base-dir", default=".")

    graphify_update = subparsers.add_parser("graphify-update")
    graphify_update.add_argument("--base-dir", default=".")

    gmail_auth = subparsers.add_parser("gmail-auth-login")
    gmail_auth.add_argument("--base-dir", default=".")

    gmail_sync_once = subparsers.add_parser("gmail-sync-once")
    gmail_sync_once.add_argument("--base-dir", default=".")
    gmail_sync_once.add_argument("--limit", type=int, default=20)

    gmail_sync_recent = subparsers.add_parser("gmail-sync-recent")
    gmail_sync_recent.add_argument("--base-dir", default=".")
    gmail_sync_recent.add_argument("--limit", type=int, default=20)

    gmail_summarize_latest = subparsers.add_parser("gmail-summarize-latest")
    gmail_summarize_latest.add_argument("--base-dir", default=".")

    gmail_summarize_recent = subparsers.add_parser("gmail-summarize-recent")
    gmail_summarize_recent.add_argument("--base-dir", default=".")
    gmail_summarize_recent.add_argument("--limit", type=int, default=10)

    return parser


def build_default_pipeline(*, base_dir: Path, fixtures_path: str | None = None) -> ArasPipeline:
    config = build_config(base_dir)
    store = SqliteArasStore(config.paths.state_db)
    telethon_module = import_module("analysts.telethon_client")
    if fixtures_path:
        client = telethon_module.FixtureTelegramClient.from_fixture_path(Path(fixtures_path))
        return ArasPipeline(client=client, store=store, config=config)
    client = telethon_module.TelethonChannelClient(base_dir=base_dir, config=config)
    return ArasPipeline(client=client, store=store, config=config)


def build_watch_runner(*, base_dir: Path) -> WatchUntilRunner:
    config = build_config(base_dir)
    store = SqliteArasStore(config.paths.state_db)
    telethon_module = import_module("analysts.telethon_client")
    client = telethon_module.TelethonChannelClient(base_dir=base_dir, config=config)
    pipeline = ArasPipeline(client=client, store=store, config=config)
    fetcher = TelegramFetcher(client=client, store=store, config=config)
    return WatchUntilRunner(client=client, message_ingestor=fetcher, pipeline=pipeline, catch_up=fetcher.catch_up)


def build_gmail_source_pipeline(*, base_dir: Path) -> GmailSourcePipeline:
    config = build_config(base_dir)
    if config.gmail is None:
        raise RuntimeError("Missing Gmail config in analysts/config.local.json")
    gmail_store = GmailStore(config.paths.state_dir / "gmail.sqlite3")
    api = GmailApiClient(
        credentials_path=(
            None
            if config.gmail.client_secret_path is None
            else config.paths.base_dir / config.gmail.client_secret_path
        ),
        credentials_json=config.gmail.client_secret_json,
        token_path=config.paths.base_dir / config.gmail.token_path,
    )
    analysts_pipeline = ArasPipeline(client=object(), store=SqliteArasStore(config.paths.state_db), config=config)
    return GmailSourcePipeline(
        config=config,
        api=api,
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name=config.gmail.account_name,
        query=config.gmail.query,
        body_rules=config.gmail.body_candidate_rules,
        zip_allow_extensions=config.gmail.zip_allow_extensions,
        raw_root=config.paths.gmail_raw_dir,
        web_capturer=PlaywrightWebCapturer(output_root=config.paths.gmail_raw_dir),
    )


def run_gmail_auth_login(*, base_dir: Path) -> int:
    pipeline = build_gmail_source_pipeline(base_dir=base_dir)
    pipeline.api.ensure_authorized()
    return 0


def run_gmail_sync_once(*, base_dir: Path, limit: int) -> int:
    pipeline = build_gmail_source_pipeline(base_dir=base_dir)
    result = pipeline.sync_once(limit=limit)
    print(f"fetched={result.fetched} skipped_existing={result.skipped_existing} last_history_id={result.last_history_id}")
    return 0


def run_gmail_sync_recent(*, base_dir: Path, limit: int) -> int:
    return run_gmail_sync_once(base_dir=base_dir, limit=limit)


def run_gmail_summarize_latest(*, base_dir: Path) -> int:
    pipeline = build_gmail_source_pipeline(base_dir=base_dir)
    execution = pipeline.summarize_latest()
    print(
        " ".join(
            [
                f"processed_files={len(execution.processed_files)}",
                f"summaries={len(execution.summaries)}",
                f"message_id={execution.summary.next_offset}",
            ]
        )
    )
    return 0


def run_gmail_summarize_recent(*, base_dir: Path, limit: int) -> int:
    return run_gmail_summarize_latest(base_dir=base_dir)


def parse_watch_deadline(until: str) -> datetime:
    return datetime.fromisoformat(until)


def print_watch_summary(*, result) -> None:
    print(
        " ".join(
            [
                f"downloaded={result.downloaded}",
                f"duplicates={result.duplicates}",
                f"ignored={result.ignored}",
                f"summarized={result.summarized}",
                f"retries={result.summarize_retries}",
            ]
        )
    )


def normalize_channels(channels: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for channel in channels:
        if channel not in seen:
            ordered.append(channel)
            seen.add(channel)
    return ordered


def configure_watch_logger(*, base_dir: Path) -> logging.Logger:
    logger = logging.getLogger("analysts.watch.cli")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    log_path = build_config(base_dir).paths.state_dir / "telegram.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def run_watch_until(
    *,
    base_dir: Path,
    until: str,
    channel: str | None = None,
    channels: list[str] | None = None,
) -> int:
    resolved_channels = normalize_channels(([channel] if channel is not None else []) + (channels or []))
    logger = configure_watch_logger(base_dir=base_dir)
    logger.info("watch_cli_invoked channels=%s base_dir=%s until=%s", ",".join(resolved_channels), base_dir, until)
    logger.info("watch_catchup_started channels=%s", ",".join(resolved_channels))
    catchup = _run_watch_catchup(base_dir=base_dir, channels=resolved_channels, logger=logger)
    logger.info(
        "watch_catchup_finished channels=%s downloaded=%s duplicates=%s ignored=%s summarized=%s",
        ",".join(resolved_channels),
        catchup.downloaded,
        catchup.duplicates,
        catchup.ignored,
        catchup.summarized,
    )
    runner = build_watch_runner(base_dir=base_dir)
    runner.logger = logger
    deadline = parse_watch_deadline(until)
    if len(resolved_channels) == 1:
        watch_result = asyncio.run(runner.watch_until(channel=resolved_channels[0], until=deadline))
    else:
        watch_result = asyncio.run(runner.watch_until_many(channels=resolved_channels, until=deadline))
    result = _merge_watch_results(catchup=catchup, live=watch_result)
    print_watch_summary(result=result)
    return 0


def _run_watch_catchup(*, base_dir: Path, channels: list[str], logger: logging.Logger) -> AsyncWatchResult:
    aggregate = AsyncWatchResult()
    pipeline = build_default_pipeline(base_dir=base_dir)
    for channel in channels:
        execution = pipeline.run_once(channel=channel)
        aggregate = AsyncWatchResult(
            seen=aggregate.seen,
            downloaded=aggregate.downloaded + execution.summary.downloaded,
            duplicates=aggregate.duplicates + execution.summary.duplicates,
            ignored=aggregate.ignored + execution.summary.ignored,
            message_failures=aggregate.message_failures,
            summarized=aggregate.summarized + len(execution.summaries),
            summarize_failures=aggregate.summarize_failures,
            summarize_retries=aggregate.summarize_retries,
        )
        logger.info(
            "watch_catchup channel=%s downloaded=%s duplicates=%s ignored=%s summarized=%s",
            channel,
            execution.summary.downloaded,
            execution.summary.duplicates,
            execution.summary.ignored,
            len(execution.summaries),
        )
    return aggregate


def _merge_watch_results(*, catchup: AsyncWatchResult, live: AsyncWatchResult) -> AsyncWatchResult:
    return AsyncWatchResult(
        seen=catchup.seen + live.seen,
        downloaded=catchup.downloaded + live.downloaded,
        duplicates=catchup.duplicates + live.duplicates,
        ignored=catchup.ignored + live.ignored,
        message_failures=catchup.message_failures + live.message_failures,
        summarized=catchup.summarized + live.summarized,
        summarize_failures=catchup.summarize_failures + live.summarize_failures,
        summarize_retries=catchup.summarize_retries + live.summarize_retries,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    base_dir = Path(args.base_dir)

    if args.command == "show-config":
        config = build_config(base_dir)
        payload = config.to_display_dict()
        print(json.dumps(payload, indent=2, default=str, sort_keys=True))
        return 0

    if args.command == "auth-login":
        config = build_config(base_dir)
        telegram_auth_login(base_dir=base_dir, config=config)
        return 0

    if args.command == "run-once":
        pipeline = build_default_pipeline(base_dir=base_dir, fixtures_path=args.fixtures)
        execution = pipeline.run_once(channel=args.channel)
        print(
            " ".join(
                [
                    f"downloaded={execution.summary.downloaded}",
                    f"duplicates={execution.summary.duplicates}",
                    f"ignored={execution.summary.ignored}",
                    f"next_offset={execution.summary.next_offset}",
                    f"processed_files={len(execution.processed_files)}",
                    f"summaries={len(execution.summaries)}",
                ]
            )
        )
        return 0

    if args.command == "watch-until":
        normalized = normalize_channels(args.channel)
        if len(normalized) == 1:
            return run_watch_until(base_dir=base_dir, channel=normalized[0], until=args.until)
        return run_watch_until(base_dir=base_dir, channels=normalized, until=args.until)

    if args.command == "summarize-latest":
        pipeline = build_default_pipeline(base_dir=base_dir)
        execution = pipeline.summarize_latest(channel=args.channel)
        print(
            " ".join(
                [
                    f"processed_files={len(execution.processed_files)}",
                    f"summaries={len(execution.summaries)}",
                    f"message_id={execution.summary.next_offset}",
                ]
            )
        )
        return 0


    if args.command == "graphify-update":
        config = build_config(base_dir)
        result = GraphifyCorpusBuilder(config).update()
        print(
            " ".join(
                [
                    f"reports={result.report_count}",
                    f"manifest={result.manifest_path}",
                    f"graphify_invoked={str(result.graphify_invoked).lower()}",
                ]
            )
        )
        return 0

    if args.command == "gmail-auth-login":
        return run_gmail_auth_login(base_dir=base_dir)

    if args.command == "gmail-sync-once":
        return run_gmail_sync_once(base_dir=base_dir, limit=args.limit)

    if args.command == "gmail-sync-recent":
        return run_gmail_sync_recent(base_dir=base_dir, limit=args.limit)

    if args.command == "gmail-summarize-latest":
        return run_gmail_summarize_latest(base_dir=base_dir)

    if args.command == "gmail-summarize-recent":
        return run_gmail_summarize_recent(base_dir=base_dir, limit=args.limit)

    if args.command == "summarize-recent":
        pipeline = build_default_pipeline(base_dir=base_dir)
        reports = [report for report in pipeline.store.list_reports() if report.channel == args.channel]
        if not reports:
            reports = RawReportCatalog(raw_dir=pipeline.config.paths.telegram_raw_dir, channel=args.channel).recent_reports(args.limit)
        else:
            reports = reports[-args.limit:]
        total_processed = 0
        total_summaries = 0
        for report in reports:
            execution = pipeline.summarize_report(report)
            total_processed += len(execution.processed_files)
            total_summaries += len(execution.summaries)
        print(
            " ".join(
                [
                    f"reports={len(reports)}",
                    f"processed_files={total_processed}",
                    f"summaries={total_summaries}",
                ]
            )
        )
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
