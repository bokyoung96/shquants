from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths


FOREIGN_OR_GLOBAL_TERMS = (
    "MSCI",
    "DAX",
    "독일",
    "유로스탁스",
    "라틴",
    "토탈월드",
    "선진",
    "글로벌",
    "미국",
    "중국",
    "차이나",
    "일본",
    "인도",
    "베트남",
    "멕시코",
    "필리핀",
    "러시아",
    "싱가포르",
    "아시아",
    "대만",
    "BYD",
    "구글",
    "마이크로소프트",
    "애플",
    "엔비디아",
    "일라이릴리",
)
ASSET_ALLOCATION_TERMS = ("TDF", "TRF", "자산배분", "멀티에셋", "주식혼합")
FACTOR_DIVIDEND_VALUE_TERMS = (
    "고배당",
    "배당",
    "밸류업",
    "밸류",
    "가치",
    "주주환원",
    "ESG",
    "팩터",
    "모멘텀",
    "성장주",
    "우량주",
    "최소변동성",
    "퀄리티",
    "하이인컴",
    "로우볼",
    "우선주",
    "경기방어",
    "퀀트",
)
REAL_ESTATE_INFRA_TERMS = ("리츠", "부동산", "인프라")
DOMESTIC_GROUP_THEME_TERMS = (
    "그룹",
    "KPOP",
    "K-POP",
    "K수출",
    "K-",
    "Fn",
    "뉴딜",
    "메타버스",
    "5G",
    "푸드",
    "뷰티",
    "콘텐츠",
    "웹툰",
    "드라마",
    "e커머스",
    "기술",
    "설비투자",
    "기후변화",
    "자율주행",
    "혁신성장",
    "BBIG",
    "운송",
    "수소경제",
    "수출주",
    "업종대표",
    "전략산업",
    "여행레저",
    "이노베이션",
    "컬처",
    "R&D",
    "IPO",
    "베스트일레븐",
    "Top5",
    "포커스",
    "내수주",
    "동학개미",
    "플랫폼",
    "대장장이",
    "전기",
    "수소차",
    "농업",
    "전력",
    "탄소",
    "테크",
    "블루칩",
    "TOP10",
    "원자력",
    "밸류체인",
)
BROAD_MARKET_NAME_TERMS = (
    " 200",
    "200TR",
    "200액티브",
    "200타겟",
    "200동일가중",
    "200exTOP",
    "코스피",
    "코스닥",
    "중소형",
    "KRX100",
    "KTOP30",
)
CASH_COMMODITY_NAME_TERMS = (
    "CD",
    "금리",
    "국채",
    "전단채",
    "특수채",
    "하이일드",
    "국공채",
    "단기자금",
    "물가채",
    "선물",
    "은액티브",
    "금액티브",
    "국제금",
    "골드",
    "콩선물",
)


@dataclass(frozen=True, slots=True)
class FamilyRecord:
    code: str
    name: str
    is_domestic_sector: bool
    classification_reason: str
    product_family: str
    coverage_provider: str
    provider_status: str
    expansion_lane: str
    next_action: str


def load_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported payload: {path}")
    return list(payload["items"])


def build_family_records(
    universe_items: Iterable[Mapping[str, object]],
    fnguide_items: Iterable[Mapping[str, object]],
) -> list[FamilyRecord]:
    fnguide_by_code = {str(item.get("code", "")): item for item in fnguide_items}
    records: list[FamilyRecord] = []
    for item in universe_items:
        code = str(item.get("code", ""))
        is_domestic_sector = bool(item.get("is_domestic_sector", False))
        classification_reason = str(item.get("reason", ""))
        fnguide_item = fnguide_by_code.get(code)
        provider = _provider_for(fnguide_item)
        provider_status = str(fnguide_item.get("status", "")) if fnguide_item else ""
        expansion_lane, next_action = _classify_lane(is_domestic_sector, provider, provider_status)
        records.append(
            FamilyRecord(
                code=code,
                name=str(item.get("name", "")),
                is_domestic_sector=is_domestic_sector,
                classification_reason=classification_reason,
                product_family=_product_family(str(item.get("name", "")), is_domestic_sector, classification_reason),
                coverage_provider=provider,
                provider_status=provider_status,
                expansion_lane=expansion_lane,
                next_action=next_action,
            )
        )
    return records


def write_family_inventory(universe_path: Path, fnguide_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    records = build_family_records(load_items(universe_path), load_items(fnguide_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "families.csv"
    json_path = output_dir / "families.json"
    md_path = output_dir / "families.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else list(FamilyRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    lane_counts = dict(Counter(record.expansion_lane for record in records))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "domestic_sector_count": sum(record.is_domestic_sector for record in records),
        "lane_counts": lane_counts,
        "product_family_counts": dict(Counter(record.product_family for record in records)),
        "provider_counts": dict(Counter(record.coverage_provider or "unassigned" for record in records)),
        "items": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(records), encoding="utf-8")
    return csv_path, json_path, md_path


def _provider_for(item: Mapping[str, object] | None) -> str:
    if not item:
        return ""
    provider = str(item.get("provider", ""))
    return provider or "fnguide"


def _classify_lane(is_domestic_sector: bool, provider: str, provider_status: str) -> tuple[str, str]:
    if provider == "fnguide":
        if provider_status == "downloaded":
            return "fnguide_reference", "continue_fnguide_data_pipeline"
        return "fnguide_reference", "resolve_fnguide_methodology_gap"
    if is_domestic_sector:
        return "provider_discovery", "identify_index_provider_and_methodology_source"
    return "future_product_family", "classify_index_provider_after_family_selection"


def _product_family(name: str, is_domestic_sector: bool, reason: str) -> str:
    if is_domestic_sector:
        return "domestic_sector"
    if reason == "broad_market":
        return "domestic_broad_market"
    if reason == "foreign_exposure":
        return "foreign_or_global"
    if reason == "non_equity_or_derivative":
        return "fixed_income_cash_commodity_or_derivative"
    if _contains_any(name, BROAD_MARKET_NAME_TERMS):
        return "domestic_broad_market"
    if _contains_any(name, CASH_COMMODITY_NAME_TERMS):
        return "fixed_income_cash_commodity_or_derivative"
    if _contains_any(name, FOREIGN_OR_GLOBAL_TERMS):
        return "foreign_or_global"
    if _contains_any(name, ASSET_ALLOCATION_TERMS):
        return "asset_allocation_or_tdf"
    if _contains_any(name, FACTOR_DIVIDEND_VALUE_TERMS):
        return "domestic_factor_dividend_or_value"
    if _contains_any(name, REAL_ESTATE_INFRA_TERMS):
        return "real_estate_or_infrastructure"
    if _contains_any(name, DOMESTIC_GROUP_THEME_TERMS):
        return "domestic_group_or_theme"
    return "other_or_unclassified"


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _markdown_summary(records: list[FamilyRecord]) -> str:
    lane_counts = Counter(record.expansion_lane for record in records)
    product_family_counts = Counter(record.product_family for record in records)
    provider_counts = Counter(record.coverage_provider or "unassigned" for record in records)
    discovery_records = [record for record in records if record.expansion_lane == "provider_discovery"]
    lines = [
        "# ETF index family inventory",
        "",
        f"- ETFs covered: {len(records)}",
        f"- Domestic sector ETFs: {sum(record.is_domestic_sector for record in records)}",
        "",
        "## Expansion Lanes",
        "",
    ]
    for name, count in sorted(lane_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Product Families", ""])
    for name, count in sorted(product_family_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Coverage Providers", ""])
    for name, count in sorted(provider_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Provider Discovery", ""])
    if discovery_records:
        lines.extend(["| code | name | reason | next action |", "| --- | --- | --- | --- |"])
        for record in discovery_records:
            lines.append(f"| {record.code} | {record.name} | {record.classification_reason} | {record.next_action} |")
    else:
        lines.append("- No domestic-sector ETFs are currently outside the FnGuide manifest.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an ETF index-family expansion inventory.")
    parser.add_argument("--universe", default=paths.UNIVERSE_JSON.as_posix())
    parser.add_argument("--fnguide", default=paths.FNGUIDE_PDFS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.CLASSIFICATION_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, json_path, md_path = write_family_inventory(Path(args.universe), Path(args.fnguide), Path(args.output_dir))
    print(f"wrote {csv_path}, {json_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
