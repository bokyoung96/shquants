import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from analysts.domain import PipelineExecution, PipelineRunSummary, ReportRecord
from analysts.watcher import AsyncWatchResult, WatchMessageResult, WatchUntilRunner


@dataclass
class FakeAsyncClient:
    messages: list[dict]
    calls: list[tuple[str, datetime]] | None = None
    delay_seconds: float = 0.0

    async def watch_channel(self, *, channel: str, until: datetime, on_message) -> None:
        if self.calls is not None:
            self.calls.append((channel, until))
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        for message in self.messages:
            await on_message(message)

    async def watch_channels(self, *, channels: list[str], until: datetime, on_message) -> None:
        if self.calls is not None:
            for channel in channels:
                self.calls.append((channel, until))
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        for message in self.messages:
            await on_message(message)


class FakePipeline:
    def __init__(self, failures_before_success: int = 0) -> None:
        self.failures_before_success = failures_before_success
        self.calls: list[int] = []

    def summarize_report(self, report: ReportRecord) -> PipelineExecution:
        self.calls.append(report.message_id)
        if len(self.calls) <= self.failures_before_success:
            raise RuntimeError('transient summarize failure')
        return PipelineExecution(
            summary=PipelineRunSummary(downloaded=0, duplicates=0, ignored=0, next_offset=report.message_id),
            processed_files=[],
            summaries=[],
        )


class FakeMessageIngestor:
    def __init__(self, outcomes: list[WatchMessageResult]) -> None:
        self.outcomes = outcomes
        self.calls: list[int] = []

    def ingest_message(self, *, channel: str, message: dict) -> WatchMessageResult:
        self.calls.append(message['message_id'])
        return self.outcomes.pop(0)


class FakeCatcher:
    def __init__(self, results: list[AsyncWatchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def catch_up(self, *, channel: str, limit: int) -> AsyncWatchResult:
        self.calls.append((channel, limit))
        if self.results:
            return self.results.pop(0)
        return AsyncWatchResult()


class RaisingMessageIngestor:
    def ingest_message(self, *, channel: str, message: dict) -> WatchMessageResult:
        raise TypeError("bad payload")


def _report(tmp_path: Path, *, message_id: int, file_unique_id: str) -> ReportRecord:
    pdf_path = tmp_path / 'data' / 'raw' / f'{message_id}.pdf'
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b'%PDF-1.4')
    return ReportRecord(
        id=message_id,
        source='telegram',
        channel='DOC_POOL',
        message_id=message_id,
        published_at='2026-04-15T04:00:00Z',
        title=f'report-{message_id}',
        pdf_path=pdf_path,
        content='',
        metadata={'file_unique_id': file_unique_id},
    )


def test_watch_until_returns_immediately_when_deadline_has_passed(tmp_path: Path) -> None:
    client_calls: list[tuple[str, datetime]] = []
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[], calls=client_calls),
        message_ingestor=FakeMessageIngestor([]),
        pipeline=FakePipeline(),
        now_fn=lambda: datetime.fromisoformat('2026-04-15T17:31:00+09:00'),
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult()
    assert client_calls == []


def test_watch_until_normalizes_naive_now_fn_against_aware_deadline(tmp_path: Path) -> None:
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[]),
        message_ingestor=FakeMessageIngestor([]),
        pipeline=FakePipeline(),
        now_fn=lambda: datetime(2026, 4, 15, 13, 0, 0),
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult()


def test_watch_until_summarizes_each_new_unique_report_once(tmp_path: Path) -> None:
    report = _report(tmp_path, message_id=101, file_unique_id='uniq-101')
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[{'message_id': 101}]),
        message_ingestor=FakeMessageIngestor([WatchMessageResult(status='downloaded', report=report)]),
        pipeline=FakePipeline(),
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult(seen=1, downloaded=1, summarized=1)


def test_watch_until_skips_duplicate_reports_without_resummarizing(tmp_path: Path) -> None:
    report = _report(tmp_path, message_id=101, file_unique_id='uniq-101')
    pipeline = FakePipeline()
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[{'message_id': 101}, {'message_id': 101}]),
        message_ingestor=FakeMessageIngestor(
            [
                WatchMessageResult(status='downloaded', report=report),
                WatchMessageResult(status='duplicate', report=None),
            ]
        ),
        pipeline=pipeline,
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult(seen=2, downloaded=1, duplicates=1, summarized=1)
    assert pipeline.calls == [101]


def test_watch_until_recovers_unsummarized_duplicate_report(tmp_path: Path) -> None:
    report = _report(tmp_path, message_id=111, file_unique_id='uniq-111')
    pipeline = FakePipeline()
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[{'message_id': 111}]),
        message_ingestor=FakeMessageIngestor(
            [WatchMessageResult(status='existing_unsummarized', report=report)]
        ),
        pipeline=pipeline,
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult(seen=1, duplicates=1, summarized=1)
    assert pipeline.calls == [111]


def test_watch_until_retries_summarization_immediately(tmp_path: Path) -> None:
    report = _report(tmp_path, message_id=202, file_unique_id='uniq-202')
    pipeline = FakePipeline(failures_before_success=1)
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[{'message_id': 202}]),
        message_ingestor=FakeMessageIngestor([WatchMessageResult(status='downloaded', report=report)]),
        pipeline=pipeline,
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
        summarize_retry_limit=2,
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult(seen=1, downloaded=1, summarized=1, summarize_retries=1)
    assert pipeline.calls == [202, 202]


def test_watch_until_continues_after_unrecovered_summarize_failure(tmp_path: Path) -> None:
    first_report = _report(tmp_path, message_id=301, file_unique_id='uniq-301')
    second_report = _report(tmp_path, message_id=302, file_unique_id='uniq-302')

    class MixedPipeline:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def summarize_report(self, report: ReportRecord) -> PipelineExecution:
            self.calls.append(report.message_id)
            if report.message_id == 301:
                raise RuntimeError('permanent summarize failure')
            return PipelineExecution(
                summary=PipelineRunSummary(downloaded=0, duplicates=0, ignored=0, next_offset=report.message_id),
                processed_files=[],
                summaries=[],
            )

    pipeline = MixedPipeline()
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[{'message_id': 301}, {'message_id': 302}]),
        message_ingestor=FakeMessageIngestor(
            [
                WatchMessageResult(status='downloaded', report=first_report),
                WatchMessageResult(status='downloaded', report=second_report),
            ]
        ),
        pipeline=pipeline,
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
        summarize_retry_limit=1,
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult(
        seen=2,
        downloaded=2,
        summarized=1,
        summarize_failures=1,
        summarize_retries=1,
    )
    assert pipeline.calls == [301, 301, 302]


def test_watch_until_ignores_messages_arriving_after_deadline(tmp_path: Path) -> None:
    report = _report(tmp_path, message_id=401, file_unique_id='uniq-401')
    current_time = {'value': datetime.fromisoformat('2026-04-15T17:29:59+09:00')}

    async def emit_mixed_messages(*, channel: str, until: datetime, on_message) -> None:
        await on_message({'message_id': 401, '_accepted_at': '2026-04-15T17:29:59+09:00'})
        current_time['value'] = datetime.fromisoformat('2026-04-15T17:30:01+09:00')
        await on_message({'message_id': 402, '_accepted_at': '2026-04-15T17:30:01+09:00'})

    class MixedClient:
        watch_channel = staticmethod(emit_mixed_messages)

    runner = WatchUntilRunner(
        client=MixedClient(),
        message_ingestor=FakeMessageIngestor(
            [
                WatchMessageResult(status='downloaded', report=report),
                WatchMessageResult(status='downloaded', report=_report(tmp_path, message_id=402, file_unique_id='uniq-402')),
            ]
        ),
        pipeline=FakePipeline(),
        now_fn=lambda: current_time['value'],
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult(seen=1, downloaded=1, summarized=1)


def test_watch_until_processes_message_accepted_before_deadline_even_if_it_finishes_late(tmp_path: Path) -> None:
    report = _report(tmp_path, message_id=501, file_unique_id='uniq-501')
    current_time = {'value': datetime.fromisoformat('2026-04-15T17:29:58+09:00')}

    async def emit_preaccepted_message(*, channel: str, until: datetime, on_message) -> None:
        current_time['value'] = datetime.fromisoformat('2026-04-15T17:30:01+09:00')
        await on_message({'message_id': 501, '_accepted_at': '2026-04-15T17:29:59+09:00'})

    class PreAcceptedClient:
        watch_channel = staticmethod(emit_preaccepted_message)

    runner = WatchUntilRunner(
        client=PreAcceptedClient(),
        message_ingestor=FakeMessageIngestor([WatchMessageResult(status='downloaded', report=report)]),
        pipeline=FakePipeline(),
        now_fn=lambda: current_time['value'],
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult(seen=1, downloaded=1, summarized=1)


def test_watch_until_emits_progress_logs_for_retry_and_finish(tmp_path: Path, caplog) -> None:
    report = _report(tmp_path, message_id=601, file_unique_id='uniq-601')
    logger = logging.getLogger('analysts.watch.test')
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[{'message_id': 601}]),
        message_ingestor=FakeMessageIngestor([WatchMessageResult(status='downloaded', report=report)]),
        pipeline=FakePipeline(failures_before_success=1),
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
        summarize_retry_limit=1,
        heartbeat_interval_seconds=60.0,
        logger=logger,
    )

    with caplog.at_level(logging.INFO, logger='analysts.watch.test'):
        asyncio.run(
            runner.watch_until(
                channel='DOC_POOL',
                until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
            )
        )

    text = caplog.text
    assert 'watch_started channel=DOC_POOL' in text
    assert 'watch_message status=downloaded channel=DOC_POOL message_id=601' in text
    assert 'watch_retry message_id=601 attempt=1' in text
    assert 'watch_summary_ok message_id=601' in text
    assert 'watch_finished channel=DOC_POOL' in text


def test_watch_until_emits_heartbeat_while_waiting(tmp_path: Path, caplog) -> None:
    logger = logging.getLogger('analysts.watch.heartbeat')
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[], delay_seconds=0.03),
        message_ingestor=FakeMessageIngestor([]),
        pipeline=FakePipeline(),
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
        heartbeat_interval_seconds=0.01,
        logger=logger,
    )

    with caplog.at_level(logging.INFO, logger='analysts.watch.heartbeat'):
        asyncio.run(
            runner.watch_until(
                channel='DOC_POOL',
                until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
            )
        )

    assert 'watch_heartbeat channel=DOC_POOL' in caplog.text


def test_watch_until_runs_catch_up_without_summarizing(tmp_path: Path) -> None:
    pipeline = FakePipeline()
    catcher = FakeCatcher([AsyncWatchResult(downloaded=2, ignored=1)])
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[], delay_seconds=0.03),
        message_ingestor=FakeMessageIngestor([]),
        pipeline=pipeline,
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
        catch_up=catcher.catch_up,
        catch_up_interval_seconds=0.01,
        catch_up_limit=25,
        heartbeat_interval_seconds=60.0,
    )

    result = asyncio.run(
        runner.watch_until(
            channel='DOC_POOL',
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert catcher.calls
    assert catcher.calls[0] == ('DOC_POOL', 25)
    assert result == AsyncWatchResult(downloaded=2)
    assert pipeline.calls == []


def test_watch_until_many_processes_two_channels_in_one_runner(tmp_path: Path) -> None:
    sector_report = _report(tmp_path, message_id=701, file_unique_id='uniq-701')
    figure_report = _report(tmp_path, message_id=702, file_unique_id='uniq-702')
    pipeline = FakePipeline()
    runner = WatchUntilRunner(
        client=FakeAsyncClient(
            messages=[
                {'message_id': 701, 'chat': {'title': 'DOC_POOL'}},
                {'message_id': 702, 'chat': {'title': 'report_figure_by_offset'}},
            ]
        ),
        message_ingestor=FakeMessageIngestor(
            [
                WatchMessageResult(status='downloaded', report=sector_report),
                WatchMessageResult(status='downloaded', report=figure_report),
            ]
        ),
        pipeline=pipeline,
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
    )

    result = asyncio.run(
        runner.watch_until_many(
            channels=['DOC_POOL', 'report_figure_by_offset'],
            until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
        )
    )

    assert result == AsyncWatchResult(seen=2, downloaded=2, summarized=2)
    assert pipeline.calls == [701, 702]


def test_watch_until_many_logs_channel_specific_events(tmp_path: Path, caplog) -> None:
    report = _report(tmp_path, message_id=703, file_unique_id='uniq-703')
    logger = logging.getLogger('analysts.watch.multi')
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[{'message_id': 703, 'chat': {'title': 'report_figure_by_offset'}}]),
        message_ingestor=FakeMessageIngestor([WatchMessageResult(status='downloaded', report=report)]),
        pipeline=FakePipeline(),
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
        logger=logger,
    )

    with caplog.at_level(logging.INFO, logger='analysts.watch.multi'):
        asyncio.run(
            runner.watch_until_many(
                channels=['DOC_POOL', 'report_figure_by_offset'],
                until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
            )
        )

    text = caplog.text
    assert 'watch_started channels=DOC_POOL,report_figure_by_offset' in text
    assert 'watch_message status=downloaded channel=report_figure_by_offset message_id=703' in text
    assert 'watch_finished channels=DOC_POOL,report_figure_by_offset' in text


def test_watch_until_logs_message_processing_failures(tmp_path: Path, caplog) -> None:
    logger = logging.getLogger('analysts.watch.failures')
    runner = WatchUntilRunner(
        client=FakeAsyncClient(messages=[{'message_id': 801, 'chat': {'title': 'DOC_POOL'}}]),
        message_ingestor=RaisingMessageIngestor(),
        pipeline=FakePipeline(),
        now_fn=lambda: datetime.fromisoformat('2026-04-15T13:00:00+09:00'),
        logger=logger,
    )

    with caplog.at_level(logging.INFO, logger='analysts.watch.failures'):
        result = asyncio.run(
            runner.watch_until(
                channel='DOC_POOL',
                until=datetime.fromisoformat('2026-04-15T17:30:00+09:00'),
            )
        )

    assert result == AsyncWatchResult(seen=1, message_failures=1)
    assert 'watch_message_failed channel=DOC_POOL message_id=801' in caplog.text
