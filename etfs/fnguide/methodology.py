from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from etfs import paths
from etfs.research import EtfListing, USER_AGENT


DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
FNINDEX_CATALOG_SEED_URL = "https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.SBD"
FNINDEX_DETAIL_URL = "https://www.fnindex.co.kr/overview/detail/{detail_type}/{code}"


@dataclass(frozen=True, slots=True)
class MethodologyCandidate:
    title: str
    url: str
    snippet: str
    query: str


@dataclass(frozen=True, slots=True)
class FnIndexEntry:
    code: str
    name: str
    detail_type: str = "I"


@dataclass(frozen=True, slots=True)
class MethodologyDownload:
    code: str
    name: str
    status: str
    source_url: str
    page_url: str
    file_path: str
    sha256: str
    bytes: int
    query: str
    error: str = ""
    provider: str = ""
    index_name: str = ""
    source_type: str = ""
    confidence: str = ""


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str, str]] = []
        self._in_anchor = False
        self._in_snippet = False
        self._href = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        css_class = attr.get("class", "") or ""
        if tag == "a" and "result__a" in css_class:
            self._in_anchor = True
            self._href = attr.get("href", "") or ""
            self._title_parts = []
        elif "result__snippet" in css_class:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_anchor:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_anchor:
            title = normalize_space(" ".join(self._title_parts))
            url = unwrap_duckduckgo_url(self._href)
            if title and url:
                self.results.append((title, url, ""))
            self._in_anchor = False
        elif self._in_snippet:
            snippet = normalize_space(" ".join(self._snippet_parts))
            if self.results and snippet:
                title, url, _ = self.results[-1]
                self.results[-1] = (title, url, snippet)
            self._in_snippet = False


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def unwrap_duckduckgo_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return unquote(href)


def build_fnguide_queries(listing: EtfListing) -> list[str]:
    return [
        f'"{listing.name}" FnGuide 지수 방법론',
        f'site:fnindex.co.kr "{listing.name}" 방법론',
        f'site:file.fnguide.com/fnindex/files "{listing.name}" pdf',
    ]


def load_fnindex_catalog(client: httpx.Client, *, seed_url: str = FNINDEX_CATALOG_SEED_URL) -> list[FnIndexEntry]:
    response = client.get(seed_url, follow_redirects=True)
    response.raise_for_status()
    return parse_fnindex_catalog(response.text)


def parse_fnindex_catalog(page_html: str) -> list[FnIndexEntry]:
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', page_html)
    if not match:
        return []
    data = json.loads(html.unescape(match.group(1)))
    page_data = data.get("props", {}).get("pageProps", {}).get("pageData", {})
    entries: list[FnIndexEntry] = []
    seen: set[tuple[str, str]] = set()
    for group in page_data.get("menuInfo", []):
        for item in group.get("list", []):
            code = str(item.get("IDX_CD") or "")
            name = str(item.get("IDX_NM") or "")
            target_page = item.get("TARGET_PAGE")
            if not code or not name or target_page not in {"D", None}:
                continue
            detail_type = "I" if code.startswith(("FI", "MK", "KN", "DIV")) else "C"
            key = (code, name)
            if key in seen:
                continue
            seen.add(key)
            entries.append(FnIndexEntry(code=code, name=name, detail_type=detail_type))
    return entries


def build_fnindex_catalog_candidates(
    listing: EtfListing,
    catalog: Iterable[FnIndexEntry],
    *,
    limit: int = 5,
) -> list[MethodologyCandidate]:
    scored: list[tuple[int, FnIndexEntry]] = []
    listing_tokens = _search_tokens(listing.name)
    if not listing_tokens:
        return []
    for entry in catalog:
        entry_tokens = _search_tokens(entry.name)
        overlap = _overlapping_tokens(listing_tokens, entry_tokens)
        if not overlap:
            continue
        score = len(overlap) * 20 + sum(len(token) for token in overlap) * 5
        if len(overlap) >= 2 and (
            _compact_name(listing.name) in _compact_name(entry.name)
            or _compact_name(entry.name) in _compact_name(listing.name)
        ):
            score += 50
        if "fnguide" in entry.name.lower():
            score += 5
        if _is_leverage_or_inverse(entry.name) and not _is_leverage_or_inverse(listing.name):
            score -= 100
        scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1].name, item[1].code))
    return [
        MethodologyCandidate(
            title=entry.name,
            url=FNINDEX_DETAIL_URL.format(detail_type=entry.detail_type, code=entry.code),
            snippet="FnIndex catalog match",
            query="fnindex_catalog",
        )
        for _, entry in scored[:limit]
    ]


def parse_duckduckgo_candidates(search_html: str, *, query: str) -> list[MethodologyCandidate]:
    parser = _DuckDuckGoParser()
    parser.feed(search_html)
    return [
        MethodologyCandidate(title=title, url=url, snippet=snippet, query=query)
        for title, url, snippet in parser.results
    ]


def search_candidates(client: httpx.Client, listing: EtfListing, *, max_results_per_query: int) -> list[MethodologyCandidate]:
    if max_results_per_query <= 0:
        return []
    candidates: list[MethodologyCandidate] = []
    seen: set[str] = set()
    for query in build_fnguide_queries(listing):
        response = client.get(DUCKDUCKGO_HTML_URL, params={"q": query})
        response.raise_for_status()
        for candidate in parse_duckduckgo_candidates(response.text, query=query):
            if candidate.url in seen:
                continue
            seen.add(candidate.url)
            candidates.append(candidate)
            if len(candidates) >= max_results_per_query * len(build_fnguide_queries(listing)):
                return rank_candidates(candidates)
    return rank_candidates(candidates)


def rank_candidates(candidates: Iterable[MethodologyCandidate]) -> list[MethodologyCandidate]:
    return sorted(candidates, key=_candidate_rank)


def _candidate_rank(candidate: MethodologyCandidate) -> tuple[int, str]:
    url = candidate.url.lower()
    title = candidate.title.lower()
    if "fnindex.co.kr/overview/detail" in url:
        return (0, url)
    if "file.fnguide.com/fnindex/files" in url and url.endswith(".pdf"):
        return (1, url)
    if "fnindex.co.kr" in url:
        return (2, url)
    if "fnguide.com" in url:
        return (3, url)
    if "methodology" in title or "방법론" in candidate.title:
        return (4, url)
    if "krx" in url:
        return (8, url)
    return (9, url)


def extract_pdf_links(page_html: str, *, base_url: str) -> list[str]:
    links = []
    for match in re.finditer(r"""(?:href|src)=["']([^"']+\.pdf(?:\?[^"']*)?)["']""", page_html, flags=re.I):
        links.append(urljoin(base_url, html.unescape(match.group(1))))
    for match in re.finditer(r"""https?://[^"'\s<>]+\.pdf(?:\?[^"'\s<>]*)?""", page_html, flags=re.I):
        links.append(html.unescape(match.group(0)))

    seen: set[str] = set()
    unique = []
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        unique.append(link)
    return sorted(unique, key=_pdf_rank)


def _pdf_rank(url: str) -> tuple[int, str]:
    lower = url.lower()
    if "file.fnguide.com/fnindex/files" in lower:
        return (0, lower)
    if "fnguide" in lower or "fnindex" in lower:
        return (1, lower)
    if "krx" in lower:
        return (8, lower)
    return (5, lower)


def download_methodology(
    client: httpx.Client,
    listing: EtfListing,
    *,
    raw_dir: Path,
    fnindex_catalog: Iterable[FnIndexEntry] = (),
    max_results_per_query: int = 5,
) -> MethodologyDownload:
    catalog_candidates = build_fnindex_catalog_candidates(listing, fnindex_catalog)
    try:
        search_results = search_candidates(client, listing, max_results_per_query=max_results_per_query)
    except Exception as exc:  # noqa: BLE001
        search_results = []
        search_error = str(exc)
    else:
        search_error = ""

    candidates = [*catalog_candidates, *rank_candidates(search_results)]
    if not candidates and search_error:
        return _failed_download(listing, status="search_failed", error=search_error)

    for candidate in candidates:
        try:
            pdf_url, page_url = resolve_pdf_url(client, candidate)
            if not pdf_url:
                continue
            return save_pdf(
                client,
                listing,
                pdf_url=pdf_url,
                page_url=page_url,
                query=candidate.query,
                raw_dir=raw_dir,
                index_name=candidate.title,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue

    return _failed_download(
        listing,
        status="not_found",
        error=locals().get("last_error", search_error or "no methodology pdf candidate found"),
    )


def resolve_pdf_url(client: httpx.Client, candidate: MethodologyCandidate) -> tuple[str, str]:
    if candidate.url.lower().split("?")[0].endswith(".pdf"):
        return candidate.url, candidate.url
    response = client.get(candidate.url, follow_redirects=True)
    response.raise_for_status()
    links = extract_pdf_links(response.text, base_url=str(response.url))
    if not links:
        return "", str(response.url)
    return links[0], str(response.url)


def save_pdf(
    client: httpx.Client,
    listing: EtfListing,
    *,
    pdf_url: str,
    page_url: str,
    query: str,
    raw_dir: Path,
    index_name: str = "",
) -> MethodologyDownload:
    response = client.get(pdf_url, follow_redirects=True)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF"):
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            raise ValueError(f"not a pdf response: {content_type}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    path = raw_dir / f"{listing.code}_{digest[:12]}.pdf"
    path.write_bytes(content)
    return MethodologyDownload(
        code=listing.code,
        name=listing.name,
        status="downloaded",
        source_url=pdf_url,
        page_url=page_url,
        file_path=str(path),
        sha256=digest,
        bytes=len(content),
        query=query,
        provider="fnguide",
        index_name=index_name,
        source_type="methodology_pdf",
        confidence="high",
    )


def _failed_download(listing: EtfListing, *, status: str, error: str) -> MethodologyDownload:
    return MethodologyDownload(
        code=listing.code,
        name=listing.name,
        status=status,
        source_url="",
        page_url="",
        file_path="",
        sha256="",
        bytes=0,
        query="",
        error=error,
        provider="fnguide",
        source_type="fnindex_search",
        confidence="low",
    )


def _search_tokens(name: str) -> set[str]:
    compact = _compact_name(name)
    raw_tokens = set(re.findall(r"[A-Za-z0-9]+|[가-힣]+", compact))
    tokens = set(raw_tokens)
    for token in raw_tokens:
        tokens.update(_expand_token(token))
    return {token.lower() for token in tokens if len(token) >= 2 and token.lower() not in _STOPWORDS}


def _expand_token(token: str) -> set[str]:
    expanded: set[str] = set()
    for match in re.finditer(r"[A-Za-z]+|\d+|[가-힣]+", token):
        expanded.add(match.group(0))
    if token.endswith("산업") and len(token) > 2:
        expanded.add(token.removesuffix("산업"))
        expanded.add("산업")
    if token.endswith("테마") and len(token) > 2:
        expanded.add(token.removesuffix("테마"))
        expanded.add("테마")
    if token.endswith("액티브") and len(token) > 3:
        expanded.add(token.removesuffix("액티브"))
    if token == "기계장비":
        expanded.update({"기계", "장비"})
    return expanded


def _overlapping_tokens(left: set[str], right: set[str]) -> set[str]:
    overlaps = left & right
    for left_token in left:
        for right_token in right:
            if len(left_token) >= 3 and len(right_token) >= 3 and (
                left_token in right_token or right_token in left_token
            ):
                overlaps.add(left_token if len(left_token) <= len(right_token) else right_token)
    return overlaps


def _is_leverage_or_inverse(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("레버리지", "인버스", "leverage", "inverse", "2x"))


def _compact_name(name: str) -> str:
    value = name.lower()
    for token in _STOPWORDS:
        value = value.replace(token, " ")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[^0-9a-z가-힣]+", " ", value)
    return normalize_space(value)


_STOPWORDS = {
    "1q",
    "ace",
    "bnk",
    "daishin343",
    "hanaro",
    "kbstar",
    "kodex",
    "koact",
    "plus",
    "rise",
    "sol",
    "tiger",
    "timefolio",
    "woni",
    "fn",
    "fnguide",
    "guide",
    "mkf",
    "index",
    "etf",
    "액티브",
    "지수",
    "토탈리턴",
    "tr",
}


def load_domestic_sector_list(path: Path) -> list[EtfListing]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [EtfListing(code=row["code"], name=row["name"]) for row in csv.DictReader(handle)]


def write_manifest(downloads: list[MethodologyDownload], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pdfs.csv"
    json_path = output_dir / "pdfs.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(downloads[0]).keys()) if downloads else list(MethodologyDownload.__dataclass_fields__))
        writer.writeheader()
        for item in downloads:
            writer.writerow(asdict(item))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(downloads),
        "downloaded_count": sum(item.status == "downloaded" for item in downloads),
        "items": [asdict(item) for item in downloads],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def run(args: argparse.Namespace) -> list[MethodologyDownload]:
    listings = load_domestic_sector_list(Path(args.input))
    if args.max_etfs:
        listings = listings[: args.max_etfs]
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7"}
    downloads: list[MethodologyDownload] = []
    with httpx.Client(headers=headers, timeout=httpx.Timeout(args.timeout), follow_redirects=True) as client:
        fnindex_catalog = load_fnindex_catalog(client)
        for index, listing in enumerate(listings, start=1):
            print(f"[{index}/{len(listings)}] {listing.code} {listing.name}", file=sys.stderr)
            downloads.append(
                download_methodology(
                    client,
                    listing,
                    raw_dir=Path(args.raw_dir),
                    fnindex_catalog=fnindex_catalog,
                    max_results_per_query=args.max_results,
                )
            )
            write_manifest(downloads, Path(args.output_dir))
    return downloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download FnGuide/FnIndex methodology PDFs for ETF listings.")
    parser.add_argument("--input", default=paths.SECTOR_CSV.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_OUTPUT_DIR.as_posix())
    parser.add_argument("--raw-dir", default="etfs/raw/methodologies")
    parser.add_argument("--max-etfs", type=int, default=0)
    parser.add_argument("--max-results", type=int, default=0, help="DuckDuckGo fallback results per query. 0 disables search fallback.")
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    downloads = run(args)
    downloaded = sum(item.status == "downloaded" for item in downloads)
    print(f"processed {len(downloads)} ETFs; downloaded {downloaded} methodology PDFs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
