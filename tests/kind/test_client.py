from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from kind.client import KindClient
from kind.selectors import (
    EXPECTED_TABLE_SUMMARY,
    FORM_DEFAULTS,
    KIND_MAIN_URL,
    KIND_SUB_URL,
    PARSER_SCHEMA_VERSION,
)


ANNOUNCEMENT_DATE = "2024-04-25"
PROVISIONAL_TITLE = "영업 (잠정) 실적 (공정공시)"


class FakeTransport:
    def __init__(
        self,
        pages: dict[tuple[str, int], str],
        *,
        failures: dict[tuple[str, int], int] | None = None,
    ) -> None:
        self.pages = pages
        self.failures = failures or {}
        self.get_calls: list[tuple[str, int]] = []
        self.post_calls: list[tuple[str, dict[str, str], int]] = []

    async def get(self, url: str, *, timeout_ms: int) -> str:
        self.get_calls.append((url, timeout_ms))
        return "<html>KIND main</html>"

    async def post_form(
        self,
        url: str,
        form: dict[str, str],
        *,
        timeout_ms: int,
    ) -> str:
        self.post_calls.append((url, dict(form), timeout_ms))
        key = (form["selDate"], int(form["pageIndex"]))
        remaining_failures = self.failures.get(key, 0)
        if remaining_failures:
            self.failures[key] = remaining_failures - 1
            raise RuntimeError("transient KIND failure")
        return self.pages[key]


def test_valid_cache_hit_returns_page_paths_without_network(tmp_path: Path) -> None:
    html = _page_html([_row()], current=1, total=1)
    expected = _write_valid_cache(tmp_path, ANNOUNCEMENT_DATE, [html])
    transport = FakeTransport({(ANNOUNCEMENT_DATE, 1): html})
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0)

    actual = asyncio.run(client.fetch_date(ANNOUNCEMENT_DATE))

    assert actual == expected
    assert transport.get_calls == []
    assert transport.post_calls == []


def test_retry_after_transient_post_failure_uses_expected_form(tmp_path: Path) -> None:
    html = _page_html([_row()], current=1, total=1)
    transport = FakeTransport(
        {(ANNOUNCEMENT_DATE, 1): html},
        failures={(ANNOUNCEMENT_DATE, 1): 1},
    )
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0, max_attempts=2)

    paths = asyncio.run(client.fetch_date(ANNOUNCEMENT_DATE))

    assert len(paths) == 1
    assert transport.get_calls == [(KIND_MAIN_URL, 30_000)]
    assert len(transport.post_calls) == 2
    assert transport.post_calls[-1] == (
        KIND_SUB_URL,
        {**FORM_DEFAULTS, "selDate": ANNOUNCEMENT_DATE, "pageIndex": "1"},
        30_000,
    )


def test_schema_error_page_is_cached_for_pipeline_audit(tmp_path: Path) -> None:
    malformed = "<html><body>temporary empty KIND response</body></html>"
    transport = FakeTransport({(ANNOUNCEMENT_DATE, 1): malformed})
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0)

    (path,) = asyncio.run(client.fetch_date(ANNOUNCEMENT_DATE))

    assert path.read_text(encoding="utf-8") == malformed
    assert (tmp_path / ANNOUNCEMENT_DATE / "manifest.json").exists()


def test_corrupt_cache_hash_refetches_and_rewrites_manifest(tmp_path: Path) -> None:
    old_html = _page_html([_row(time="08:05")], current=1, total=1)
    new_html = _page_html([_row(time="09:15")], current=1, total=1)
    (page_path,) = _write_valid_cache(tmp_path, ANNOUNCEMENT_DATE, [old_html])
    page_path.write_text("corrupted", encoding="utf-8")
    transport = FakeTransport({(ANNOUNCEMENT_DATE, 1): new_html})
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0)

    (actual_path,) = asyncio.run(client.fetch_date(ANNOUNCEMENT_DATE))

    assert actual_path.read_text(encoding="utf-8") == new_html
    assert transport.post_calls
    assert _manifest_hash(tmp_path, ANNOUNCEMENT_DATE, "page-0001.html") == _sha256_text(
        new_html
    )


def test_refresh_replaces_stale_extra_pages_after_success(tmp_path: Path) -> None:
    old_pages = [
        _page_html([_row(time="08:05")], current=1, total=2, links=[2]),
        _page_html([_row(time="09:05")], current=2, total=2, links=[1]),
    ]
    _write_valid_cache(tmp_path, ANNOUNCEMENT_DATE, old_pages)
    new_html = _page_html([_row(time="10:05")], current=1, total=1)
    transport = FakeTransport({(ANNOUNCEMENT_DATE, 1): new_html})
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0)

    paths = asyncio.run(client.fetch_date(ANNOUNCEMENT_DATE, refresh=True))

    assert [path.name for path in paths] == ["page-0001.html"]
    assert not (tmp_path / ANNOUNCEMENT_DATE / "page-0002.html").exists()


def test_pagination_fetches_every_page_count(tmp_path: Path) -> None:
    pages = {
        (ANNOUNCEMENT_DATE, 1): _page_html(
            [_row(time="08:05")], current=1, total=3, links=[2, 3]
        ),
        (ANNOUNCEMENT_DATE, 2): _page_html(
            [_row(time="09:05")], current=2, total=3, links=[1, 3]
        ),
        (ANNOUNCEMENT_DATE, 3): _page_html(
            [_row(time="10:05")], current=3, total=3, links=[1, 2]
        ),
    }
    transport = FakeTransport(pages)
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0)

    paths = asyncio.run(client.fetch_date(ANNOUNCEMENT_DATE))

    assert [path.name for path in paths] == [
        "page-0001.html",
        "page-0002.html",
        "page-0003.html",
    ]
    assert [call[1]["pageIndex"] for call in transport.post_calls] == ["1", "2", "3"]


def test_initial_get_occurs_once_per_client_before_network_fetch(tmp_path: Path) -> None:
    first_date = "2024-04-25"
    second_date = "2024-04-26"
    pages = {
        (first_date, 1): _page_html(
            [_row(receipt_id="20240425000004")], current=1, total=1
        ),
        (second_date, 1): _page_html(
            [_row(receipt_id="20240426000004")], current=1, total=1
        ),
    }
    transport = FakeTransport(pages)
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0)

    asyncio.run(client.fetch_date(first_date))
    asyncio.run(client.fetch_date(second_date))

    assert transport.get_calls == [(KIND_MAIN_URL, 30_000)]


def test_concurrent_same_date_requests_are_coalesced(tmp_path: Path) -> None:
    html = _page_html([_row()], current=1, total=1)
    transport = FakeTransport({(ANNOUNCEMENT_DATE, 1): html})
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0)

    async def fetch_twice() -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        return await asyncio.gather(
            client.fetch_date(ANNOUNCEMENT_DATE),
            client.fetch_date(ANNOUNCEMENT_DATE),
        )

    first, second = asyncio.run(fetch_twice())

    assert first == second
    assert len(transport.post_calls) == 1


def test_partial_failure_does_not_publish_manifest_or_leave_temp_files(
    tmp_path: Path,
) -> None:
    first_page = _page_html([_row(time="08:05")], current=1, total=2, links=[2])
    transport = FakeTransport(
        {(ANNOUNCEMENT_DATE, 1): first_page},
        failures={(ANNOUNCEMENT_DATE, 2): 3},
    )
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0, max_attempts=2)

    with pytest.raises(RuntimeError, match="KIND request failed"):
        asyncio.run(client.fetch_date(ANNOUNCEMENT_DATE))

    date_dir = tmp_path / ANNOUNCEMENT_DATE
    assert not (date_dir / "manifest.json").exists()
    assert list(date_dir.glob("*.tmp")) == []


def test_manifest_metadata_mismatch_refetches(tmp_path: Path) -> None:
    html = _page_html([_row()], current=1, total=1)
    _write_valid_cache(tmp_path, ANNOUNCEMENT_DATE, [html])
    manifest_path = tmp_path / ANNOUNCEMENT_DATE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["form"]["selDate"] = "2024-04-24"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    transport = FakeTransport({(ANNOUNCEMENT_DATE, 1): html})
    client = KindClient(transport, cache_dir=tmp_path, min_delay=0)

    asyncio.run(client.fetch_date(ANNOUNCEMENT_DATE))

    assert len(transport.post_calls) == 1


def _write_valid_cache(cache_dir: Path, date: str, pages: list[str]) -> tuple[Path, ...]:
    date_dir = cache_dir / date
    date_dir.mkdir(parents=True)
    page_paths: list[Path] = []
    hashes: dict[str, str] = {}
    for index, html in enumerate(pages, start=1):
        path = date_dir / f"page-{index:04d}.html"
        path.write_text(html, encoding="utf-8", newline="")
        page_paths.append(path)
        hashes[path.name] = _sha256_text(html)
    manifest = {
        "manifest_version": 1,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "date": date,
        "source_url": KIND_SUB_URL,
        "form": {**FORM_DEFAULTS, "selDate": date},
        "page_count": len(pages),
        "pages": [path.name for path in page_paths],
        "sha256": hashes,
    }
    (date_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return tuple(page_paths)


def _manifest_hash(cache_dir: Path, date: str, page_name: str) -> str:
    manifest = json.loads(
        (cache_dir / date / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest["sha256"][page_name]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(
    *,
    time: str = "08:05",
    company: str = "SK하이닉스",
    title: str = PROVISIONAL_TITLE,
    receipt_id: str = "20240425000004",
) -> str:
    return f"""
        <tr>
          <td>{time}</td>
          <td><a href="#" onclick="companysummary_open('00066');return false;">{company}</a></td>
          <td><a href="#" onclick="openDisclsViewer('{receipt_id}','')">{title}</a></td>
          <td>{company}</td>
          <td>-</td>
        </tr>
    """


def _paging_html(
    *,
    current: int,
    total: int,
    links: Iterable[int] = (),
) -> str:
    clickable = "".join(
        f'<a href="#page_link_{number}" '
        f'onclick="fnPageGo(\'{number}\');return false;">{number}</a>'
        for number in links
    )
    return f"""
        <section class="paging-group">
          <div class="paging type-00">
            {clickable}
            <a class="active" href="#page_link_{current}" onclick="return false;">{current}</a>
          </div>
          <div class="info type-00">[총 <em>1</em> 건] 페이지 <strong>{current}</strong>/{total} </div>
        </section>
    """


def _page_html(
    rows: Iterable[str],
    *,
    current: int,
    total: int,
    links: Iterable[int] = (),
) -> str:
    return f"""
        <html>
          <body>
            <section class="scrarea type-00">
              <table class="list type-00 mt10" summary="{EXPECTED_TABLE_SUMMARY}">
                <tbody>{''.join(rows)}</tbody>
              </table>
            </section>
            {_paging_html(current=current, total=total, links=links)}
          </body>
        </html>
    """
