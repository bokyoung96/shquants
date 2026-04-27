from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Protocol

from ...domain import ReportRecord

_ACCEPTED_AT_KEY = "_accepted_at"


@dataclass(frozen=True)
class WatchMessageResult:
    status: str
    report: ReportRecord | None = None


@dataclass(frozen=True)
class AsyncWatchResult:
    seen: int = 0
    downloaded: int = 0
    duplicates: int = 0
    ignored: int = 0
    message_failures: int = 0
    summarized: int = 0
    summarize_failures: int = 0
    summarize_retries: int = 0


class AsyncWatchClient(Protocol):
    async def watch_channel(
        self,
        *,
        channel: str,
        until: datetime,
        on_message: Callable[[dict[str, Any]], Awaitable[None] | None],
    ) -> None: ...

    async def watch_channels(
        self,
        *,
        channels: list[str],
        until: datetime,
        on_message: Callable[[dict[str, Any]], Awaitable[None] | None],
    ) -> None: ...


class MessageIngestor(Protocol):
    def ingest_message(self, *, channel: str, message: dict[str, Any]) -> WatchMessageResult: ...


class ReportPipeline(Protocol):
    def summarize_report(self, report: ReportRecord): ...


class WatchUntilRunner:
    def __init__(
        self,
        *,
        client: AsyncWatchClient,
        message_ingestor: MessageIngestor,
        pipeline: ReportPipeline,
        now_fn: Callable[[], datetime] = datetime.now,
        summarize_retry_limit: int = 1,
        heartbeat_interval_seconds: float = 15.0,
        catch_up: Callable[..., Any] | None = None,
        catch_up_interval_seconds: float = 300.0,
        catch_up_limit: int = 300,
        logger: logging.Logger | None = None,
    ) -> None:
        self.client = client
        self.message_ingestor = message_ingestor
        self.pipeline = pipeline
        self.now_fn = now_fn
        self.summarize_retry_limit = summarize_retry_limit
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.catch_up = catch_up
        self.catch_up_interval_seconds = catch_up_interval_seconds
        self.catch_up_limit = catch_up_limit
        self.logger = logger or logging.getLogger("analysts.watch")

    async def watch_until(self, *, channel: str, until: datetime) -> AsyncWatchResult:
        return await self.watch_until_many(channels=[channel], until=until)

    async def watch_until_many(self, *, channels: list[str], until: datetime) -> AsyncWatchResult:
        if until.tzinfo is None:
            raise ValueError("watch-until requires a timezone-aware deadline")
        if self._normalize_now(until=until) >= until:
            return AsyncWatchResult()
        channels = self._normalize_channels(channels)
        if not channels:
            raise ValueError("watch-until requires at least one channel")

        counts = self._new_counts()
        self.logger.info("watch_started %s until=%s", self._channels_log_field(channels), until.isoformat())
        stop_heartbeat = asyncio.Event()

        async def on_message(message: dict[str, Any]) -> None:
            await self._handle_message(channels=channels, message=message, until=until, counts=counts)

        heartbeat_task = asyncio.create_task(
            self._heartbeat(scope=self._channels_log_field(channels), until=until, counts=counts, stop=stop_heartbeat)
        )
        catch_up_task = self._start_catch_up(channels=channels, until=until, counts=counts, stop=stop_heartbeat)
        try:
            await self._await_watch(channels=channels, until=until, on_message=on_message)
        finally:
            stop_heartbeat.set()
            await heartbeat_task
            if catch_up_task is not None:
                await catch_up_task
        self.logger.info(
            "watch_finished %s seen=%s downloaded=%s duplicates=%s ignored=%s message_failures=%s summarized=%s retries=%s failures=%s",
            self._channels_log_field(channels),
            counts["seen"],
            counts["downloaded"],
            counts["duplicates"],
            counts["ignored"],
            counts["message_failures"],
            counts["summarized"],
            counts["summarize_retries"],
            counts["summarize_failures"],
        )
        return AsyncWatchResult(**counts)

    async def _await_watch(
        self,
        *,
        channels: list[str],
        until: datetime,
        on_message: Callable[[dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        if len(channels) == 1:
            maybe_awaitable = self.client.watch_channel(channel=channels[0], until=until, on_message=on_message)
        else:
            maybe_awaitable = self.client.watch_channels(channels=channels, until=until, on_message=on_message)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    def _normalize_now(self, *, until: datetime) -> datetime:
        current = self.now_fn()
        if current.tzinfo is None:
            return current.replace(tzinfo=until.tzinfo)
        return current.astimezone(until.tzinfo)

    @staticmethod
    def _new_counts() -> dict[str, int]:
        return {
            "seen": 0,
            "downloaded": 0,
            "duplicates": 0,
            "ignored": 0,
            "message_failures": 0,
            "summarized": 0,
            "summarize_failures": 0,
            "summarize_retries": 0,
        }

    async def _handle_message(
        self,
        *,
        channels: list[str],
        message: dict[str, Any],
        until: datetime,
        counts: dict[str, int],
    ) -> None:
        channel = self._channel_from_message(message=message, channels=channels)
        if not self._was_accepted_before_deadline(message=message, until=until):
            self.logger.info("watch_message status=ignored_after_deadline channel=%s message_id=%s", channel, message.get("message_id"))
            return
        counts["seen"] += 1
        try:
            result = self.message_ingestor.ingest_message(channel=channel, message=message)
        except Exception:
            counts["message_failures"] += 1
            self.logger.exception("watch_message_failed channel=%s message_id=%s", channel, message.get("message_id"))
            return
        if result.status == "duplicate":
            counts["duplicates"] += 1
            self.logger.info("watch_message status=duplicate channel=%s message_id=%s", channel, message.get("message_id"))
            return
        if result.status == "existing_unsummarized" and result.report is not None:
            counts["duplicates"] += 1
            self.logger.info("watch_message status=existing_unsummarized channel=%s message_id=%s", channel, result.report.message_id)
            self._summarize_report(report=result.report, counts=counts)
            return
        if result.status == "ignored" or result.report is None:
            counts["ignored"] += 1
            self.logger.info("watch_message status=ignored channel=%s message_id=%s", channel, message.get("message_id"))
            return

        counts["downloaded"] += 1
        self.logger.info("watch_message status=downloaded channel=%s message_id=%s", channel, result.report.message_id)
        self._summarize_report(report=result.report, counts=counts)

    def _summarize_report(self, *, report: ReportRecord, counts: dict[str, int]) -> None:
        for attempt in range(self.summarize_retry_limit + 1):
            try:
                self.pipeline.summarize_report(report)
                counts["summarized"] += 1
                self.logger.info("watch_summary_ok message_id=%s", report.message_id)
                return
            except Exception:
                if attempt < self.summarize_retry_limit:
                    counts["summarize_retries"] += 1
                    self.logger.warning("watch_retry message_id=%s attempt=%s", report.message_id, attempt + 1)
                    continue
                counts["summarize_failures"] += 1
                self.logger.exception("watch_summary_failed message_id=%s", report.message_id)

    def _was_accepted_before_deadline(self, *, message: dict[str, Any], until: datetime) -> bool:
        accepted_at = self._accepted_at(message=message, until=until)
        if accepted_at is None:
            return self._normalize_now(until=until) < until
        return accepted_at < until

    @staticmethod
    def _accepted_at(*, message: dict[str, Any], until: datetime) -> datetime | None:
        value = message.get(_ACCEPTED_AT_KEY)
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if value.tzinfo is None:
            return value.replace(tzinfo=until.tzinfo)
        return value.astimezone(until.tzinfo)

    async def _heartbeat(
        self,
        *,
        scope: str,
        until: datetime,
        counts: dict[str, int],
        stop: asyncio.Event,
    ) -> None:
        if self.heartbeat_interval_seconds <= 0:
            return
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.heartbeat_interval_seconds)
                return
            except asyncio.TimeoutError:
                self.logger.info(
                    "watch_heartbeat %s until=%s seen=%s downloaded=%s duplicates=%s ignored=%s message_failures=%s summarized=%s retries=%s failures=%s",
                    scope,
                    until.isoformat(),
                    counts["seen"],
                    counts["downloaded"],
                    counts["duplicates"],
                    counts["ignored"],
                    counts["message_failures"],
                    counts["summarized"],
                    counts["summarize_retries"],
                    counts["summarize_failures"],
                )

    def _start_catch_up(
        self,
        *,
        channels: list[str],
        until: datetime,
        counts: dict[str, int],
        stop: asyncio.Event,
    ) -> asyncio.Task | None:
        if self.catch_up is None or self.catch_up_interval_seconds <= 0:
            return None
        return asyncio.create_task(self._catch_up(channels=channels, until=until, counts=counts, stop=stop))

    async def _catch_up(
        self,
        *,
        channels: list[str],
        until: datetime,
        counts: dict[str, int],
        stop: asyncio.Event,
    ) -> None:
        while self._normalize_now(until=until) < until:
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.catch_up_interval_seconds)
                return
            except asyncio.TimeoutError:
                pass
            for channel in channels:
                try:
                    result = await asyncio.to_thread(self.catch_up, channel=channel, limit=self.catch_up_limit)
                except Exception:
                    counts["message_failures"] += 1
                    self.logger.exception("watch_catch_up_failed channel=%s", channel)
                    continue
                added = self._add_catch_up(counts=counts, result=result)
                self.logger.info(
                    "watch_catch_up channel=%s downloaded=%s duplicates=%s ignored=%s",
                    channel,
                    added["downloaded"],
                    added["duplicates"],
                    added["ignored"],
                )

    @staticmethod
    def _add_catch_up(*, counts: dict[str, int], result: Any) -> dict[str, int]:
        added = {
            "downloaded": WatchUntilRunner._count(getattr(result, "downloaded", 0)),
            "duplicates": WatchUntilRunner._count(getattr(result, "skipped_duplicates", getattr(result, "duplicates", 0))),
            "ignored": WatchUntilRunner._count(getattr(result, "ignored_updates", getattr(result, "ignored", 0))),
        }
        counts["downloaded"] += added["downloaded"]
        return added

    @staticmethod
    def _count(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        return len(value)

    @staticmethod
    def _normalize_channels(channels: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for channel in channels:
            if channel not in seen:
                ordered.append(channel)
                seen.add(channel)
        return ordered

    @staticmethod
    def _channels_log_field(channels: list[str]) -> str:
        if len(channels) == 1:
            return f"channel={channels[0]}"
        return f"channels={','.join(channels)}"

    @staticmethod
    def _channel_from_message(*, message: dict[str, Any], channels: list[str]) -> str:
        chat = message.get("chat") or {}
        title = chat.get("title")
        if isinstance(title, str) and title:
            return title
        return channels[0]
