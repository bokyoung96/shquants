from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx

from etfs import paths


NAVER_ETF_LIST_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

FOREIGN_KEYWORDS = (
    "미국",
    "중국",
    "차이나",
    "일본",
    "인도",
    "베트남",
    "유럽",
    "글로벌",
    "선진국",
    "신흥국",
    "해외",
    "S&P",
    "NASDAQ",
    "NYSE",
    "Dow",
    "USA",
    "US ",
    "U.S.",
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
    "은행채",
    "특수은행채",
    "국고채",
    "회사채",
    "금융채",
    "통안채",
    "CD금리",
    "KOFR",
    "머니마켓",
    "단기금융",
    "금현물",
    "금선물",
    "은선물",
    "원유",
    "구리",
    "농산물",
    "달러",
    "환율",
    "인버스",
    "레버리지",
)

SECTOR_KEYWORDS = (
    "반도체",
    "자동차",
    "모빌리티",
    "은행",
    "증권",
    "보험",
    "금융",
    "지주",
    "바이오",
    "헬스케어",
    "의료",
    "IT",
    "정보기술",
    "소프트웨어",
    "게임",
    "미디어",
    "엔터",
    "커뮤니케이션",
    "화장품",
    "소비",
    "소비재",
    "필수소비",
    "경기소비",
    "음식료",
    "에너지",
    "2차전지",
    "이차전지",
    "배터리",
    "철강",
    "소재",
    "화학",
    "조선",
    "해운",
    "건설",
    "기계",
    "방산",
    "우주",
    "원전",
    "로봇",
    "AI",
    "인터넷",
)

BROAD_MARKET_PATTERNS = (
    re.compile(r"(^|\s)KODEX\s*200($|\s)"),
    re.compile(r"(^|\s)TIGER\s*200($|\s)"),
    re.compile(r"(^|\s)RISE\s*200($|\s)"),
    re.compile(r"(^|\s)ACE\s*200($|\s)"),
    re.compile(r"(^|\s)1Q\s*200($|\s)"),
    re.compile(r"(^|\s|[A-Z])K\s*200($|\s)", re.I),
    re.compile(r"KOSPI\s*200"),
    re.compile(r"코스피\s*200"),
    re.compile(r"코스닥\s*150"),
    re.compile(r"KRX\s*300"),
)


@dataclass(frozen=True, slots=True)
class EtfListing:
    code: str
    name: str


@dataclass(frozen=True, slots=True)
class EtfClassification:
    listing: EtfListing
    is_domestic_sector: bool
    reason: str


def fetch_naver_etf_list(client: httpx.Client) -> list[EtfListing]:
    response = client.get(NAVER_ETF_LIST_URL)
    response.raise_for_status()
    return normalize_naver_etf_rows(parse_naver_etf_payload(response.content, response.encoding))


def parse_naver_etf_payload(content: bytes, encoding: str | None) -> list[dict]:
    text = decode_bytes(content, encoding)
    data = json.loads(text)
    return list(data.get("result", {}).get("etfItemList", []))


def normalize_naver_etf_rows(items: Iterable[dict]) -> list[EtfListing]:
    rows: list[EtfListing] = []
    for item in items:
        name = str(item.get("itemname", "")).strip()
        code = str(item.get("itemcode", "")).strip()
        if not name or not code:
            continue
        rows.append(EtfListing(code=code.zfill(6), name=name))
    return sorted(rows, key=lambda row: (row.name, row.code))


def classify_etf_listing(listing: EtfListing) -> EtfClassification:
    name = listing.name
    if _contains_any(name, FOREIGN_KEYWORDS):
        return EtfClassification(listing=listing, is_domestic_sector=False, reason="foreign_exposure")
    if _contains_any(name, NON_EQUITY_KEYWORDS):
        return EtfClassification(listing=listing, is_domestic_sector=False, reason="non_equity_or_derivative")
    if any(pattern.search(name) for pattern in BROAD_MARKET_PATTERNS):
        return EtfClassification(listing=listing, is_domestic_sector=False, reason="broad_market")
    if not _contains_any(name, SECTOR_KEYWORDS):
        return EtfClassification(listing=listing, is_domestic_sector=False, reason="no_sector_keyword")
    return EtfClassification(listing=listing, is_domestic_sector=True, reason="domestic_sector_keyword")


def filter_domestic_sector_etfs(items: Iterable[dict]) -> list[dict[str, str]]:
    rows = []
    for listing in normalize_naver_etf_rows(items):
        classification = classify_etf_listing(listing)
        if classification.is_domestic_sector:
            rows.append({"code": listing.code, "name": listing.name})
    return rows


def fetch_etf_research_universe(*, timeout: float) -> tuple[list[EtfListing], list[EtfClassification]]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7"}
    with httpx.Client(headers=headers, timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
        listings = fetch_naver_etf_list(client)
    return listings, [classify_etf_listing(listing) for listing in listings]


def write_outputs(
    listings: list[EtfListing],
    classifications: list[EtfClassification],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "all.csv"
    domestic_path = output_dir / "sector.csv"
    json_path = output_dir / "universe.json"

    with all_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["code", "name"])
        writer.writeheader()
        for listing in listings:
            writer.writerow(asdict(listing))

    with domestic_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["code", "name", "reason"])
        writer.writeheader()
        for classification in classifications:
            if classification.is_domestic_sector:
                writer.writerow(
                    {
                        "code": classification.listing.code,
                        "name": classification.listing.name,
                        "reason": classification.reason,
                    }
                )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(listings),
        "domestic_sector_count": sum(item.is_domestic_sector for item in classifications),
        "items": [
            {
                "code": item.listing.code,
                "name": item.listing.name,
                "is_domestic_sector": item.is_domestic_sector,
                "reason": item.reason,
            }
            for item in classifications
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_path, domestic_path, json_path


def decode_bytes(content: bytes, encoding: str | None) -> str:
    for candidate in (encoding, "utf-8", "euc-kr", "cp949"):
        if not candidate:
            continue
        try:
            return content.decode(candidate, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode("utf-8", errors="replace")


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def run(args: argparse.Namespace) -> tuple[list[EtfListing], list[EtfClassification]]:
    listings, classifications = fetch_etf_research_universe(timeout=args.timeout)
    write_outputs(listings, classifications, Path(args.output_dir))
    return listings, classifications


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch ETF listings and classify domestic sector ETFs.")
    parser.add_argument("--output-dir", default=paths.UNIVERSE_OUTPUT_DIR.as_posix())
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    listings, classifications = run(args)
    domestic_count = sum(item.is_domestic_sector for item in classifications)
    print(f"fetched {len(listings)} ETFs; domestic sector ETFs: {domestic_count}; outputs written to {args.output_dir}")
    return 0


__all__ = [
    "EtfClassification",
    "EtfListing",
    "NAVER_ETF_LIST_URL",
    "classify_etf_listing",
    "decode_bytes",
    "fetch_etf_research_universe",
    "fetch_naver_etf_list",
    "filter_domestic_sector_etfs",
    "normalize_naver_etf_rows",
    "parse_naver_etf_payload",
    "write_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
