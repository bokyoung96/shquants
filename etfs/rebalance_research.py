from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse

import fitz
import httpx


NAVER_ETF_LIST_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

OVERSEAS_KEYWORDS = (
    "미국",
    "중국",
    "차이나",
    "대만",
    "일본",
    "인도",
    "베트남",
    "유럽",
    "북미",
    "아시아",
    "한중",
    "한국대만",
    "글로벌",
    "나스닥",
    "S&P",
    "NASDAQ",
    "NYSE",
    "미장",
    "다우",
    "유로",
    "월드",
    "선진국",
    "신흥국",
    "USA",
    "US ",
    " U.S.",
    "China",
    "Taiwan",
    "Japan",
    "India",
    "Vietnam",
    "Europe",
    "Global",
)

NON_EQUITY_KEYWORDS = (
    "채권",
    "금융채",
    "은행채",
    "특수은행채",
    "단기",
    "리츠",
)

DOMESTIC_EXCLUDE_KEYWORDS = OVERSEAS_KEYWORDS + NON_EQUITY_KEYWORDS

SECTOR_INCLUDE_KEYWORDS = (
    "반도체",
    "자동차",
    "모빌리티",
    "은행",
    "증권",
    "보험",
    "금융",
    "헬스케어",
    "바이오",
    "의료",
    "IT",
    "정보기술",
    "소프트웨어",
    "게임",
    "미디어",
    "엔터",
    "커뮤니케이션",
    "화장품",
    "소비재",
    "음식료",
    "필수소비",
    "경기소비",
    "에너지",
    "2차전지",
    "이차전지",
    "배터리",
    "철강",
    "소재",
    "화학",
    "조선",
    "운송",
    "건설",
    "기계",
    "방산",
    "원전",
    "로봇",
    "AI",
    "인터넷",
)

BROAD_MARKET_PATTERNS = (
    re.compile(r"\b200\b"),
    re.compile(r"(?:^|\s)K(?:OSPI)?200(?:\s|$)"),
    re.compile(r"\b100\b"),
    re.compile(r"코스피(?!.*(은행|증권|보험|금융|헬스|바이오|IT|반도체|자동차))"),
    re.compile(r"코스닥150"),
    re.compile(r"KRX\s*300"),
)

REBALANCE_KEYWORDS = (
    "리밸런싱",
    "리벨런싱",
    "정기변경",
    "정기 변경",
    "정기심사",
    "정기 심사",
    "구성종목 변경",
    "구성 종목 변경",
    "수시변경",
    "특별변경",
    "rebalance",
    "rebalancing",
    "review",
    "reconstitution",
)


@dataclass(slots=True)
class Etf:
    code: str
    name: str


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass(slots=True)
class SourceEvidence:
    title: str
    url: str
    query: str
    excerpt: str
    schedule: str
    status: str = "ok"


@dataclass(slots=True)
class EtfResearch:
    code: str
    name: str
    queries: list[str]
    evidence: list[SourceEvidence] = field(default_factory=list)


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._in_result_anchor = False
        self._in_snippet = False
        self._anchor_href = ""
        self._anchor_text: list[str] = []
        self._snippet_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        css_class = attr.get("class", "") or ""
        if tag == "a" and "result__a" in css_class:
            self._in_result_anchor = True
            self._anchor_href = attr.get("href", "") or ""
            self._anchor_text = []
        elif "result__snippet" in css_class:
            self._in_snippet = True
            self._snippet_text = []

    def handle_data(self, data: str) -> None:
        if self._in_result_anchor:
            self._anchor_text.append(data)
        elif self._in_snippet:
            self._snippet_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_result_anchor:
            title = normalize_space(" ".join(self._anchor_text))
            url = unwrap_duckduckgo_url(self._anchor_href)
            if title and url:
                self.results.append(SearchResult(title=title, url=url))
            self._in_result_anchor = False
        elif self._in_snippet:
            snippet = normalize_space(" ".join(self._snippet_text))
            if snippet and self.results and not self.results[-1].snippet:
                self.results[-1].snippet = snippet
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


def parse_duckduckgo_results(search_html: str) -> list[SearchResult]:
    parser = _DuckDuckGoParser()
    parser.feed(search_html)
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for result in parser.results:
        if result.url in seen:
            continue
        seen.add(result.url)
        unique.append(result)
    return unique


def filter_domestic_sector_etfs(items: Iterable[dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        name = str(item.get("itemname", ""))
        code = str(item.get("itemcode", ""))
        if not name or not code:
            continue
        if any(keyword in name for keyword in DOMESTIC_EXCLUDE_KEYWORDS):
            continue
        if any(pattern.search(name) for pattern in BROAD_MARKET_PATTERNS):
            continue
        if not any(keyword in name for keyword in SECTOR_INCLUDE_KEYWORDS):
            continue
        rows.append({"code": code.zfill(6), "name": name})
    return sorted(rows, key=lambda row: (row["name"], row["code"]))


def contains_overseas_keyword(text: str) -> bool:
    normalized = f" {text} "
    normalized_lower = normalized.lower()
    for keyword in OVERSEAS_KEYWORDS:
        if keyword.lower() in normalized_lower:
            return True
    return False


def is_domestic_source_result(result: SearchResult) -> bool:
    haystack = " ".join([result.title, result.url, result.snippet])
    return not contains_overseas_keyword(haystack)


def extract_rebalance_excerpt(text: str, *, max_chars: int = 1400) -> str:
    cleaned = normalize_space(text)
    if not cleaned:
        return ""

    sentence_parts = [
        normalize_space(part)
        for part in re.split(r"\n+|(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=니다\.)\s+", text)
        if normalize_space(part)
    ]
    hits: list[str] = []
    for index, sentence in enumerate(sentence_parts):
        if any(keyword.lower() in sentence.lower() for keyword in REBALANCE_KEYWORDS):
            start = max(0, index - 1)
            end = min(len(sentence_parts), index + 2)
            hits.extend(sentence_parts[start:end])

    if not hits:
        for keyword in ("구성종목", "구성 종목", "산출방법", "methodology"):
            pos = cleaned.lower().find(keyword.lower())
            if pos >= 0:
                return cleaned[max(0, pos - 250) : pos + max_chars].strip()
        return cleaned[:max_chars].strip()

    excerpt = normalize_space(" ".join(dict.fromkeys(hits)))
    return excerpt[:max_chars].strip()


def classify_rebalance_schedule(text: str) -> str:
    lower = text.lower()
    compact = re.sub(r"\s+", "", text)
    if any(token in lower for token in ("monthly", "매월", "월간")):
        return "monthly"
    if any(token in lower for token in ("quarterly", "분기", "매년 4회", "연 4회")):
        return "quarterly"
    if "매년4회" in compact or "연4회" in compact:
        return "quarterly"
    if any(token in lower for token in ("semi-annual", "semiannual", "반기", "매년 2회", "연 2회")):
        return "semiannual"
    if "매년2회" in compact or "연2회" in compact:
        return "semiannual"
    if any(token in lower for token in ("annual", "매년", "연 1회", "연1회")):
        return "annual"
    if any(token in lower for token in ("수시", "특별변경", "extraordinary")):
        return "ad_hoc_or_extraordinary"
    return "unknown"


def build_queries(etf_name: str) -> list[str]:
    return [
        f'"{etf_name}" 기초지수 리밸런싱 정기변경',
        f'"{etf_name}" 기초지수 방법론 구성종목',
        f'"{etf_name}" 투자설명서 PDF 기초지수',
    ]


def fetch_naver_etfs(client: httpx.Client) -> list[dict]:
    response = client.get(NAVER_ETF_LIST_URL)
    response.raise_for_status()
    data = {"result": {"etfItemList": parse_naver_etf_payload(response.content, response.encoding)}}
    return list(data.get("result", {}).get("etfItemList", []))


def parse_naver_etf_payload(content: bytes, encoding: str | None) -> list[dict]:
    text = decode_bytes(content, encoding)
    data = json.loads(text)
    return list(data.get("result", {}).get("etfItemList", []))


def search_duckduckgo(client: httpx.Client, query: str, *, max_results: int) -> list[SearchResult]:
    response = client.get(DUCKDUCKGO_HTML_URL, params={"q": query})
    response.raise_for_status()
    return parse_duckduckgo_results(response.text)[:max_results]


def fetch_url_text(client: httpx.Client, url: str, *, max_bytes: int = 5_000_000) -> str:
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    content = response.content[:max_bytes]
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
        return extract_pdf_text(content)
    return extract_html_text(content, response.encoding)


def extract_pdf_text(content: bytes) -> str:
    with fitz.open(stream=content, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "br", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def extract_html_text(content: bytes, encoding: str | None) -> str:
    decoded = decode_bytes(content, encoding)
    parser = _TextExtractor()
    parser.feed(decoded)
    return normalize_space(" ".join(parser.parts))


def decode_bytes(content: bytes, encoding: str | None) -> str:
    encodings = [encoding, "utf-8", "euc-kr", "cp949"]
    for candidate in encodings:
        if not candidate:
            continue
        try:
            return content.decode(candidate, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode("utf-8", errors="replace")


def research_etf(
    client: httpx.Client,
    etf: dict[str, str],
    *,
    search_results_per_query: int,
    sources_per_etf: int,
) -> EtfResearch:
    queries = build_queries(etf["name"])
    research = EtfResearch(code=etf["code"], name=etf["name"], queries=queries)
    seen_urls: set[str] = set()

    for query in queries:
        try:
            results = search_duckduckgo(client, query, max_results=search_results_per_query)
        except Exception as exc:  # noqa: BLE001
            research.evidence.append(
                SourceEvidence(
                    title="search failed",
                    url=DUCKDUCKGO_HTML_URL,
                    query=query,
                    excerpt=str(exc),
                    schedule="unknown",
                    status="search_failed",
                )
            )
            continue

        for result in results:
            if result.url in seen_urls:
                continue
            if not is_domestic_source_result(result):
                continue
            seen_urls.add(result.url)
            if len(research.evidence) >= sources_per_etf:
                return research
            try:
                text = fetch_url_text(client, result.url)
                excerpt = extract_rebalance_excerpt(text)
                if not excerpt:
                    excerpt = result.snippet
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                excerpt = result.snippet or str(exc)
                status = "fetch_failed"
            if contains_overseas_keyword(excerpt):
                continue
            research.evidence.append(
                SourceEvidence(
                    title=result.title,
                    url=result.url,
                    query=query,
                    excerpt=excerpt,
                    schedule=classify_rebalance_schedule(excerpt),
                    status=status,
                )
            )
    return research


def write_outputs(researches: list[EtfResearch], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "domestic_sector_etf_rebalance_research.json"
    md_path = output_dir / "domestic_sector_etf_rebalance_research.md"
    csv_path = output_dir / "domestic_sector_etf_rebalance_research.csv"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(researches),
        "items": [asdict(item) for item in researches],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["code", "name", "schedule", "status", "title", "url", "query", "excerpt"],
        )
        writer.writeheader()
        for item in researches:
            for evidence in item.evidence:
                writer.writerow(
                    {
                        "code": item.code,
                        "name": item.name,
                        "schedule": evidence.schedule,
                        "status": evidence.status,
                        "title": evidence.title,
                        "url": evidence.url,
                        "query": evidence.query,
                        "excerpt": evidence.excerpt,
                    }
                )

    lines = [
        "# Domestic Sector ETF Underlying Index Rebalance Research",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- ETF count: {len(researches)}",
        "",
    ]
    for item in researches:
        lines.extend([f"## {item.name} ({item.code})", ""])
        for evidence in item.evidence:
            lines.extend(
                [
                    f"- Schedule: `{evidence.schedule}`; status: `{evidence.status}`",
                    f"- Source: [{evidence.title}]({evidence.url})",
                    f"- Query: `{evidence.query}`",
                    f"- Excerpt: {evidence.excerpt}",
                    "",
                ]
            )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path, csv_path


def run(args: argparse.Namespace) -> list[EtfResearch]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7"}
    timeout = httpx.Timeout(args.timeout)
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        items = fetch_naver_etfs(client)
        etfs = filter_domestic_sector_etfs(items)
        if args.max_etfs:
            etfs = etfs[: args.max_etfs]
    researches: list[EtfResearch] = []
    if args.workers <= 1:
        for index, etf in enumerate(etfs, start=1):
            print(f"[{index}/{len(etfs)}] researching {etf['code']} {etf['name']}", file=sys.stderr)
            researches.append(research_etf_with_new_client(etf, args, headers))
            write_outputs(researches, Path(args.output_dir))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_etf = {
                executor.submit(research_etf_with_new_client, etf, args, headers): etf for etf in etfs
            }
            for index, future in enumerate(as_completed(future_to_etf), start=1):
                etf = future_to_etf[future]
                try:
                    researches.append(future.result())
                    print(f"[{index}/{len(etfs)}] done {etf['code']} {etf['name']}", file=sys.stderr)
                except Exception as exc:  # noqa: BLE001
                    researches.append(
                        EtfResearch(
                            code=etf["code"],
                            name=etf["name"],
                            queries=build_queries(etf["name"]),
                            evidence=[
                                SourceEvidence(
                                    title="research failed",
                                    url="",
                                    query="",
                                    excerpt=str(exc),
                                    schedule="unknown",
                                    status="research_failed",
                                )
                            ],
                        )
                    )
                    print(f"[{index}/{len(etfs)}] failed {etf['code']} {etf['name']}: {exc}", file=sys.stderr)
                write_outputs(sorted(researches, key=lambda item: (item.name, item.code)), Path(args.output_dir))
    researches = sorted(researches, key=lambda item: (item.name, item.code))
    write_outputs(researches, Path(args.output_dir))
    return researches


def research_etf_with_new_client(
    etf: dict[str, str], args: argparse.Namespace, headers: dict[str, str]
) -> EtfResearch:
    timeout = httpx.Timeout(args.timeout)
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        return research_etf(
            client,
            etf,
            search_results_per_query=args.search_results,
            sources_per_etf=args.sources_per_etf,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search the web for domestic sector ETF underlying-index rebalance mechanisms."
    )
    parser.add_argument("--output-dir", default="etfs/output")
    parser.add_argument("--max-etfs", type=int, default=0, help="Limit ETFs for smoke runs. 0 means all.")
    parser.add_argument("--search-results", type=int, default=5)
    parser.add_argument("--sources-per-etf", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    researches = run(args)
    print(f"researched {len(researches)} ETFs; outputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
