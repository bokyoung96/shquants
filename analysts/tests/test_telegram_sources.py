from pathlib import Path

from analysts.fetcher import TelegramFetcher as LegacyTelegramFetcher
from analysts.runner_entry import DEFAULT_WATCH_CHANNELS as LEGACY_DEFAULT_WATCH_CHANNELS
from analysts.runner_entry import build_arg_parser as legacy_build_arg_parser
from analysts.telethon_client import TelethonChannelClient as LegacyTelethonChannelClient
from analysts.watcher import WatchUntilRunner as LegacyWatchUntilRunner
from analysts.sources.telegram.client import TelethonChannelClient
from analysts.sources.telegram.fetcher import TelegramFetcher
from analysts.sources.telegram.runner import DEFAULT_WATCH_CHANNELS
from analysts.sources.telegram.runner import build_arg_parser
from analysts.sources.telegram.watcher import WatchUntilRunner


def test_legacy_and_new_telegram_client_symbol_match() -> None:
    assert LegacyTelethonChannelClient is TelethonChannelClient


def test_legacy_and_new_telegram_fetcher_symbol_match() -> None:
    assert LegacyTelegramFetcher is TelegramFetcher


def test_legacy_and_new_telegram_watcher_symbol_match() -> None:
    assert LegacyWatchUntilRunner is WatchUntilRunner


def test_legacy_and_new_runner_surface_match() -> None:
    assert LEGACY_DEFAULT_WATCH_CHANNELS == DEFAULT_WATCH_CHANNELS
    assert legacy_build_arg_parser(default_base_dir=Path("analysts")) is not None
    assert build_arg_parser(default_base_dir=Path("analysts")) is not None
