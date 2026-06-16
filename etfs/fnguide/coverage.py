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


CORE_MARKET_DATA = {
    "stock_prices",
    "listed_shares",
    "corporate_actions",
    "constituent_universe",
    "krx_trading_calendar",
    "market_cap",
    "trading_value_liquidity",
    "free_float_ratio",
    "fics_industry_classification",
    "ranking_inputs",
    "futures_options_expiry_calendar",
    "month_end_business_day_calendar",
}
EXTERNAL_MODEL_DATA = {"keyword_source_documents", "score_inputs"}
DIVIDEND_OR_CUSTOM_DATA = {
    "dividend_data",
    "fundamental_factors",
    "custom_index_formula_inputs",
    "index_inclusion_factor",
}


@dataclass(frozen=True, slots=True)
class FnGuideCoverageRecord:
    code: str
    name: str
    index_name: str
    methodology_status: str
    methodology_file: str
    family: str
    rebalance_frequency: str
    rebalance_months: str
    weighting_scheme: str
    readiness: str
    next_action: str
    missing_data: list[str]


def load_requirement_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported requirements payload: {path}")
    return list(payload["items"])


def build_coverage_records(items: Iterable[Mapping[str, object]]) -> list[FnGuideCoverageRecord]:
    return [_build_record(item) for item in items]


def write_fnguide_coverage(requirements_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    records = build_coverage_records(load_requirement_items(requirements_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fnguide.csv"
    json_path = output_dir / "fnguide.json"
    md_path = output_dir / "fnguide.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_csv_row(records[0]).keys()) if records else list(FnGuideCoverageRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))

    payload = {
        "provider": "fnguide",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "methodology_available_count": sum(record.methodology_status == "available" for record in records),
        "readiness_counts": dict(Counter(record.readiness for record in records)),
        "family_counts": dict(Counter(record.family for record in records)),
        "weighting_counts": dict(Counter(record.weighting_scheme for record in records)),
        "missing_data_counts": _missing_data_counts(records),
        "items": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(records), encoding="utf-8")
    return csv_path, json_path, md_path


def _build_record(item: Mapping[str, object]) -> FnGuideCoverageRecord:
    missing_data = _list_field(item.get("missing_data"))
    readiness, next_action = _classify_readiness(str(item.get("methodology_status", "")), missing_data)
    return FnGuideCoverageRecord(
        code=str(item.get("code", "")),
        name=str(item.get("name", "")),
        index_name=str(item.get("index_name", "")),
        methodology_status=str(item.get("methodology_status", "")),
        methodology_file=str(item.get("methodology_file", "")),
        family=str(item.get("family", "")),
        rebalance_frequency=str(item.get("rebalance_frequency", "")),
        rebalance_months=str(item.get("rebalance_months", "")),
        weighting_scheme=str(item.get("weighting_scheme", "")),
        readiness=readiness,
        next_action=next_action,
        missing_data=missing_data,
    )


def _classify_readiness(methodology_status: str, missing_data: list[str]) -> tuple[str, str]:
    missing = set(missing_data)
    if methodology_status != "available" or "methodology_pdf_missing" in missing:
        return "blocked_missing_pdf", "find_fnguide_methodology_pdf"
    if missing & EXTERNAL_MODEL_DATA:
        return "needs_external_model_data", "collect_theme_keyword_and_score_inputs"
    if missing & DIVIDEND_OR_CUSTOM_DATA:
        return "needs_dividend_or_custom_data", "collect_dividend_and_custom_formula_inputs"
    if missing & CORE_MARKET_DATA:
        return "needs_core_market_data", "load_core_market_and_calendar_data"
    if missing:
        return "needs_data_mapping", "map_remaining_required_data"
    return "ready_for_calculation", "implement_rebalance_calculation"


def _csv_row(record: FnGuideCoverageRecord) -> dict[str, object]:
    row = asdict(record)
    row["missing_data"] = "|".join(record.missing_data)
    return row


def _list_field(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item for item in value.split("|") if item]
    return []


def _missing_data_counts(records: list[FnGuideCoverageRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.missing_data)
    return dict(counter)


def _markdown_summary(records: list[FnGuideCoverageRecord]) -> str:
    readiness_counts = Counter(record.readiness for record in records)
    family_counts = Counter(record.family for record in records)
    missing_counts = _missing_data_counts(records)
    available = sum(record.methodology_status == "available" for record in records)

    lines = [
        "# FnGuide first-pass coverage",
        "",
        f"- ETFs covered: {len(records)}",
        f"- Methodology PDFs available: {available}",
        f"- Methodology PDFs missing: {len(records) - available}",
        "",
        "## Readiness",
        "",
    ]
    for name, count in sorted(readiness_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Families", ""])
    for name, count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Missing Data", ""])
    for name, count in sorted(missing_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(
        [
            "",
            "## ETF Actions",
            "",
            "| code | name | index | readiness | next action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(f"| {record.code} | {record.name} | {record.index_name} | {record.readiness} | {record.next_action} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize FnGuide ETF methodology coverage and next actions.")
    parser.add_argument("--requirements", default=paths.FNGUIDE_REQUIREMENTS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, json_path, md_path = write_fnguide_coverage(Path(args.requirements), Path(args.output_dir))
    print(f"wrote {csv_path}, {json_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
