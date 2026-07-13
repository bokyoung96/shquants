from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

from playwright.async_api import APIRequestContext, Playwright, async_playwright

from kind.parser import KindSchemaError, parse_disclosure_page
from kind.selectors import (
    FORM_DEFAULTS,
    KIND_MAIN_URL,
    KIND_SUB_URL,
    PARSER_SCHEMA_VERSION,
)


MANIFEST_VERSION = 1


class Transport(Protocol):
    async def get(self, url: str, *, timeout_ms: int) -> str: ...

    async def post_form(
        self,
        url: str,
        form: dict[str, str],
        *,
        timeout_ms: int,
    ) -> str: ...


class PlaywrightTransport:
    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._context: APIRequestContext | None = None

    async def __aenter__(self) -> PlaywrightTransport:
        self._playwright = await async_playwright().start()
        self._context = await self._playwright.request.new_context()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._context is not None:
            await self._context.dispose()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def get(self, url: str, *, timeout_ms: int) -> str:
        context = self._require_context()
        response = await context.get(url, timeout=timeout_ms)
        if not 200 <= response.status < 300:
            raise RuntimeError(f"KIND GET failed with HTTP {response.status}")
        return await response.text()

    async def post_form(
        self,
        url: str,
        form: dict[str, str],
        *,
        timeout_ms: int,
    ) -> str:
        context = self._require_context()
        response = await context.post(url, form=form, timeout=timeout_ms)
        if not 200 <= response.status < 300:
            raise RuntimeError(f"KIND POST failed with HTTP {response.status}")
        return await response.text()

    def _require_context(self) -> APIRequestContext:
        if self._context is None:
            raise RuntimeError("PlaywrightTransport is not open")
        return self._context


class RateLimiter:
    def __init__(self, min_delay: float) -> None:
        self.min_delay = max(0.0, min_delay)
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            delay = self._last_request + self.min_delay - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = loop.time()
            self._last_request = now


class KindClient:
    def __init__(
        self,
        transport: Transport,
        *,
        cache_dir: Path,
        min_delay: float = 0.75,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        self.transport = transport
        self.cache_dir = Path(cache_dir)
        self.timeout_ms = int(timeout_seconds * 1000)
        self.max_attempts = max(1, max_attempts)
        self.limiter = RateLimiter(min_delay)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._date_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def fetch_date(self, date: str, *, refresh: bool = False) -> tuple[Path, ...]:
        async with self._date_locks[date]:
            cached = self._valid_cached_paths(date)
            if cached is not None and not refresh:
                return cached

            await self._ensure_initialized()
            paths = await self._fetch_uncached_date(date)
            if refresh:
                self._remove_stale_pages(date, paths)
            return paths

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._request_with_retry(
                lambda: self.transport.get(KIND_MAIN_URL, timeout_ms=self.timeout_ms)
            )
            self._initialized = True

    async def _fetch_uncached_date(self, date: str) -> tuple[Path, ...]:
        date_dir = self.cache_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        self._remove_temporary_files(date_dir)

        page_paths: list[Path] = []
        hashes: dict[str, str] = {}
        total_pages = 1
        page_number = 1

        try:
            while page_number <= total_pages:
                form = {**FORM_DEFAULTS, "selDate": date, "pageIndex": str(page_number)}
                html = await self._request_with_retry(
                    lambda form=form: self.transport.post_form(
                        KIND_SUB_URL,
                        form,
                        timeout_ms=self.timeout_ms,
                    )
                )
                try:
                    parsed = parse_disclosure_page(
                        html,
                        announcement_date=date,
                        page=page_number,
                    )
                except KindSchemaError:
                    # Publish malformed pages so the pipeline can retain an
                    # auditable schema error and continue other dates.
                    parsed = None
                if parsed is not None:
                    total_pages = parsed.total_pages
                page_path = date_dir / f"page-{page_number:04d}.html"
                _atomic_write_text(page_path, html)
                page_paths.append(page_path)
                hashes[page_path.name] = _sha256_text(html)
                page_number += 1
        except Exception:
            self._remove_temporary_files(date_dir)
            raise

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "date": date,
            "source_url": KIND_SUB_URL,
            "form": {**FORM_DEFAULTS, "selDate": date},
            "page_count": total_pages,
            "pages": [path.name for path in page_paths],
            "sha256": hashes,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_text(
            date_dir / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        )
        return tuple(page_paths)

    async def _request_with_retry(self, operation: Callable[[], Awaitable[str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                await self.limiter.wait()
                return await operation()
            except Exception as error:
                last_error = error
                if attempt < self.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(
            f"KIND request failed after {self.max_attempts} attempts"
        ) from last_error

    def _valid_cached_paths(self, date: str) -> tuple[Path, ...] | None:
        date_dir = self.cache_dir / date
        manifest_path = date_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        expected_pages = [
            f"page-{index:04d}.html"
            for index in range(1, int(manifest.get("page_count", 0)) + 1)
        ]
        expected_form = {**FORM_DEFAULTS, "selDate": date}
        if (
            manifest.get("manifest_version") != MANIFEST_VERSION
            or manifest.get("parser_schema_version") != PARSER_SCHEMA_VERSION
            or manifest.get("date") != date
            or manifest.get("source_url") != KIND_SUB_URL
            or manifest.get("form") != expected_form
            or manifest.get("pages") != expected_pages
        ):
            return None

        hashes = manifest.get("sha256")
        if not isinstance(hashes, dict):
            return None

        paths = tuple(date_dir / page_name for page_name in expected_pages)
        for path in paths:
            if not path.exists():
                return None
            if _sha256_text(_read_text_exact(path)) != hashes.get(path.name):
                return None
        return paths

    def _remove_stale_pages(self, date: str, valid_paths: tuple[Path, ...]) -> None:
        date_dir = self.cache_dir / date
        valid_names = {path.name for path in valid_paths}
        for path in date_dir.glob("page-*.html"):
            if path.name not in valid_names:
                path.unlink()

    def _remove_temporary_files(self, date_dir: Path) -> None:
        for path in date_dir.glob("*.tmp"):
            path.unlink()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_text_exact(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
