from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths


@dataclass(frozen=True, slots=True)
class RequirementRecord:
    code: str
    name: str
    index_name: str
    methodology_status: str
    methodology_file: str
    source_url: str
    family: str
    rebalance_frequency: str
    rebalance_months: str
    weighting_scheme: str
    weight_cap: str
    required_data: list[str]
    available_data: list[str]
    missing_data: list[str]


def load_rule_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported rules payload: {path}")
    return list(payload["items"])


def infer_required_data(item: Mapping[str, object]) -> list[str]:
    rule_set = _mapping(item.get("rule_set"))
    rebalance = _mapping(rule_set.get("rebalance"))
    weighting = _mapping(rule_set.get("weighting"))
    screening = _mapping(rule_set.get("screening"))
    family = str(rule_set.get("family", ""))
    status = str(item.get("status", ""))

    required = {
        "rule_profile",
        "stock_prices",
        "listed_shares",
        "corporate_actions",
        "constituent_universe",
        "krx_trading_calendar",
    }
    if status == "downloaded" and item.get("file_path"):
        required.add("methodology_pdf")
    else:
        required.add("methodology_pdf_missing")

    timing = str(rebalance.get("timing", ""))
    if "선물" in timing or "옵션" in timing or "futures" in timing.lower() or "option" in timing.lower():
        required.add("futures_options_expiry_calendar")
    if "마지막 영업일" in timing or "month" in timing.lower():
        required.add("month_end_business_day_calendar")

    if bool(weighting.get("uses_free_float")):
        required.add("free_float_ratio")
    scheme = str(weighting.get("scheme", ""))
    if scheme in {"float_market_cap_weighted", "market_cap_weighted"}:
        required.add("market_cap")
    if scheme == "score_weighted":
        required.add("score_inputs")
    if scheme == "fundamental_float_adjusted":
        required.update({"fundamental_factors", "free_float_ratio"})
    if scheme == "custom_weighted":
        required.add("custom_index_formula_inputs")
    if scheme == "iif_adjusted":
        required.add("index_inclusion_factor")

    if bool(screening.get("market_cap")):
        required.add("market_cap")
    if bool(screening.get("liquidity")):
        required.add("trading_value_liquidity")
    if bool(screening.get("keyword")) or family == "keyword_theme":
        required.add("keyword_source_documents")
    if bool(screening.get("fics")) or family == "sector_theme":
        required.add("fics_industry_classification")
    if screening.get("selection_count"):
        required.add("ranking_inputs")
    if family == "dividend":
        required.add("dividend_data")

    return sorted(required)


def build_requirement_records(items: Iterable[Mapping[str, object]]) -> list[RequirementRecord]:
    records: list[RequirementRecord] = []
    for item in items:
        required = infer_required_data(item)
        available = [name for name in required if _is_available(name, item)]
        missing = [name for name in required if name not in available]
        rule_set = _mapping(item.get("rule_set"))
        rebalance = _mapping(rule_set.get("rebalance"))
        weighting = _mapping(rule_set.get("weighting"))
        months = rebalance.get("months") or []
        records.append(
            RequirementRecord(
                code=str(item.get("code", "")),
                name=str(item.get("name", "")),
                index_name=str(rule_set.get("index_name", "")),
                methodology_status="available" if item.get("status") == "downloaded" and item.get("file_path") else "missing",
                methodology_file=str(item.get("file_path", "")),
                source_url=str(item.get("source_url", "")),
                family=str(rule_set.get("family", "")),
                rebalance_frequency=str(rebalance.get("frequency", "")),
                rebalance_months=",".join(str(month) for month in months),
                weighting_scheme=str(weighting.get("scheme", "")),
                weight_cap=str(weighting.get("cap", "")),
                required_data=required,
                available_data=available,
                missing_data=missing,
            )
        )
    return records


def write_data_requirements(rules_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    records = build_requirement_records(load_rule_items(rules_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "requirements.csv"
    json_path = output_dir / "requirements.json"
    md_path = output_dir / "requirements.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_csv_row(records[0]).keys()) if records else list(RequirementRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "methodology_available_count": sum(record.methodology_status == "available" for record in records),
        "items": [asdict(record) for record in records],
        "data_dictionary": _data_dictionary(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(records), encoding="utf-8")
    return csv_path, json_path, md_path


def _csv_row(record: RequirementRecord) -> dict[str, object]:
    row = asdict(record)
    row["required_data"] = "|".join(record.required_data)
    row["available_data"] = "|".join(record.available_data)
    row["missing_data"] = "|".join(record.missing_data)
    return row


def _is_available(name: str, item: Mapping[str, object]) -> bool:
    if name == "methodology_pdf":
        path = Path(str(item.get("file_path", "")))
        return bool(path) and path.exists()
    if name == "rule_profile":
        return True
    return False


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _data_dictionary() -> dict[str, str]:
    return {
        "methodology_pdf": "Downloaded methodology PDF used as the source document.",
        "rule_profile": f"Extracted rule set in {paths.FNGUIDE_RULES_JSON.as_posix()}.",
        "stock_prices": "Daily stock prices required for index value and weights.",
        "listed_shares": "Listed/common shares for market-cap and index-share calculations.",
        "free_float_ratio": "Free-float ratio used by most FnGuide/FnIndex formulas.",
        "corporate_actions": "Share changes, delistings, splits, rights issues, and other index divisor events.",
        "constituent_universe": "Eligible KOSPI/KOSDAQ common-stock universe and exclusion flags.",
        "krx_trading_calendar": "Korean exchange business-day calendar.",
        "futures_options_expiry_calendar": "KOSPI/KOSDAQ futures/options expiry schedule for rebalance dates.",
        "month_end_business_day_calendar": "Month-end business-day calendar for selection or special rebalance checks.",
        "market_cap": "Market capitalization data for screening, ranking, and market-cap weighting.",
        "trading_value_liquidity": "Average trading value or liquidity data for screens.",
        "keyword_source_documents": "Reports, disclosures, or keyword scores used by keyword/theme indices.",
        "fics_industry_classification": "FnGuide Industry Classification Standard sector/industry mapping.",
        "score_inputs": "Score model inputs for score-weighted methodologies.",
        "fundamental_factors": "Financial statement/fundamental inputs for RAFI or fundamental-weighted rules.",
        "custom_index_formula_inputs": "Methodology-specific custom weights or third-party score inputs.",
        "index_inclusion_factor": "IIF/index inclusion factor needed for index-share calculations.",
        "dividend_data": "Dividend, yield, and high-dividend ranking inputs.",
        "ranking_inputs": "Inputs needed to select top-N constituents.",
        "methodology_pdf_missing": "Methodology PDF was not downloaded for this ETF.",
    }


def _markdown_summary(records: list[RequirementRecord]) -> str:
    available = sum(record.methodology_status == "available" for record in records)
    missing = len(records) - available
    data_counts: dict[str, int] = {}
    for record in records:
        for name in record.missing_data:
            data_counts[name] = data_counts.get(name, 0) + 1

    lines = [
        "# ETF Index Methodology Data Requirements",
        "",
        f"- ETFs covered: {len(records)}",
        f"- Methodology PDFs available: {available}",
        f"- Methodology PDFs missing: {missing}",
        "",
        "## Missing Or Unverified Data",
        "",
    ]
    for name, count in sorted(data_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "## ETF Methodology Files",
            "",
            "| code | name | index | methodology | missing data |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        methodology = record.methodology_file if record.methodology_file else "missing"
        missing_data = ", ".join(record.missing_data)
        lines.append(f"| {record.code} | {record.name} | {record.index_name} | {methodology} | {missing_data} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize methodology files and data requirements by ETF.")
    parser.add_argument("--rules", default=paths.FNGUIDE_RULES_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, json_path, md_path = write_data_requirements(Path(args.rules), Path(args.output_dir))
    print(f"wrote {csv_path}, {json_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
