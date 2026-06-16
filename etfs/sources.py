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


SOURCE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sp_global", ("S&P", "SP500", "SNP")),
    ("msci", ("MSCI",)),
    ("nasdaq", ("NASDAQ", "나스닥")),
    ("stoxx", ("STOXX", "유로스탁스", "DAX")),
    ("fnguide", ("FnGuide", "Fn ")),
    ("iselect", ("iSelect",)),
    ("kis", ("KIS",)),
    ("krx", ("KRX", "코스피", "코스닥", "KOSPI", "KOSDAQ")),
)


@dataclass(frozen=True, slots=True)
class SourceRecord:
    code: str
    name: str
    product_family: str
    coverage_provider: str
    provider_status: str
    source_candidate: str
    source_kind: str
    confidence: str
    next_action: str


def load_family_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported families payload: {path}")
    return list(payload["items"])


def build_source_records(items: Iterable[Mapping[str, object]]) -> list[SourceRecord]:
    return [_build_record(item) for item in items]


def write_source_inventory(families_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    records = build_source_records(load_family_items(families_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sources.csv"
    json_path = output_dir / "sources.json"
    md_path = output_dir / "sources.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else list(SourceRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "source_counts": dict(Counter(record.source_candidate or "unknown" for record in records)),
        "source_family_counts": _source_family_counts(records),
        "items": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(records), encoding="utf-8")
    return csv_path, json_path, md_path


def _build_record(item: Mapping[str, object]) -> SourceRecord:
    coverage_provider = str(item.get("coverage_provider", ""))
    provider_status = str(item.get("provider_status", ""))
    name = str(item.get("name", ""))
    product_family = str(item.get("product_family", ""))
    source_candidate, confidence = _infer_source_candidate(name, coverage_provider, product_family)
    return SourceRecord(
        code=str(item.get("code", "")),
        name=name,
        product_family=product_family,
        coverage_provider=coverage_provider,
        provider_status=provider_status,
        source_candidate=source_candidate,
        source_kind="methodology_provider" if source_candidate else "",
        confidence=confidence,
        next_action=_next_action(source_candidate, confidence),
    )


def _infer_source_candidate(name: str, coverage_provider: str, product_family: str) -> tuple[str, str]:
    if coverage_provider:
        return coverage_provider, "high"
    lowered = name.lower()
    for source, terms in SOURCE_TERMS:
        if any(term.lower() in lowered for term in terms):
            return source, "medium"
    if _looks_like_domestic_200_index(name):
        return "krx", "medium"
    fallback = _family_fallback_source(product_family)
    if fallback:
        return fallback, "low"
    return "", "low"


def _looks_like_domestic_200_index(name: str) -> bool:
    return any(token in name for token in (" 200", "200TR", "K200"))


def _next_action(source_candidate: str, confidence: str) -> str:
    if not source_candidate:
        return "manual_source_research"
    if confidence == "high":
        return f"continue_{source_candidate}_pipeline"
    if confidence == "low":
        return f"research_{source_candidate}_sources"
    return f"build_{source_candidate}_methodology_probe"


def _family_fallback_source(product_family: str) -> str:
    if product_family == "domestic_broad_market":
        return "krx"
    if product_family == "fixed_income_cash_commodity_or_derivative":
        return "fixed_income_or_commodity_provider"
    if product_family == "foreign_or_global":
        return "global_index_provider"
    if product_family in {
        "domestic_group_or_theme",
        "domestic_factor_dividend_or_value",
        "asset_allocation_or_tdf",
        "real_estate_or_infrastructure",
    }:
        return "issuer_or_domestic_index_provider"
    return ""


def _source_family_counts(records: list[SourceRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        source = record.source_candidate or "unknown"
        counter[f"{record.product_family}:{source}"] += 1
    return dict(counter)


def _markdown_summary(records: list[SourceRecord]) -> str:
    source_counts = Counter(record.source_candidate or "unknown" for record in records)
    family_source_counts = Counter(f"{record.product_family}:{record.source_candidate or 'unknown'}" for record in records)
    unknown_records = [record for record in records if not record.source_candidate]
    lines = [
        "# ETF methodology source candidates",
        "",
        f"- ETFs covered: {len(records)}",
        f"- Source candidates assigned: {sum(bool(record.source_candidate) for record in records)}",
        f"- Manual source research: {len(unknown_records)}",
        "",
        "## Sources",
        "",
    ]
    for name, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Family By Source", ""])
    for name, count in sorted(family_source_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Manual Source Research", ""])
    if unknown_records:
        lines.extend(["| code | name | family |", "| --- | --- | --- |"])
        for record in unknown_records:
            lines.append(f"| {record.code} | {record.name} | {record.product_family} |")
    else:
        lines.append("- No records require manual source research.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ETF methodology source-candidate inventory.")
    parser.add_argument("--families", default=paths.FAMILIES_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.SOURCES_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, json_path, md_path = write_source_inventory(Path(args.families), Path(args.output_dir))
    print(f"wrote {csv_path}, {json_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
