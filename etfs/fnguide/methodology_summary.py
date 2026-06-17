from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

import pandas as pd

from etfs import paths


def build_etf_methodology_summary(
    *,
    holdings_dir: Path = paths.REFRESHED_HOLDINGS_FILES_DIR,
    rules_path: Path = paths.FNGUIDE_RULES_JSON,
    requirements_path: Path = paths.FNGUIDE_REQUIREMENTS_JSON,
    audit_path: Path = paths.FNGUIDE_METHODOLOGY_AUDIT_JSON,
) -> dict[str, object]:
    rules_by_code = _items_by_code(_load_items(rules_path))
    requirements_by_code = _items_by_code(_load_items(requirements_path))
    blockers_by_index = _audit_blockers_by_index(audit_path)

    items = []
    for parquet_path in sorted(holdings_dir.glob("holdings_*.parquet")):
        etf_code = parquet_path.stem.removeprefix("holdings_")
        rule_item = rules_by_code.get(etf_code, {})
        requirement_item = requirements_by_code.get(etf_code, {})
        rules = _mapping(rule_item.get("rules"))
        index_code = _index_code_from_page_url(str(rule_item.get("page_url", "")))
        review_flags = _review_flags(rule_item, rules, requirement_item, blockers_by_index.get(index_code, []))
        items.append(
            {
                "etf_code": etf_code,
                "etf_name": str(rule_item.get("name") or requirement_item.get("name") or ""),
                "holdings_file": parquet_path.as_posix(),
                **_holdings_stats(parquet_path),
                "methodology_pdf_status": _methodology_pdf_status(rule_item, requirement_item),
                "methodology_file": str(rule_item.get("file_path") or requirement_item.get("methodology_file") or ""),
                "index_code": index_code,
                "index_name": str(rules.get("index_name") or requirement_item.get("index_name") or ""),
                "methodology_structured_status": _structured_status(rule_item, review_flags),
                "rebalance_frequency": str(rules.get("review_frequency") or requirement_item.get("rebalance_frequency") or ""),
                "rebalance_months": [int(month) for month in rules.get("review_months", [])],
                "rebalance_timing": str(rules.get("rebalance_timing", "")),
                "rebalance_date_status": _rebalance_date_status(rules, requirement_item),
                "weighting_scheme": str(rules.get("weighting_scheme") or requirement_item.get("weighting_scheme") or ""),
                "weight_cap": str(rules.get("weight_cap") or requirement_item.get("weight_cap") or ""),
                "readiness": str(requirement_item.get("readiness", "")),
                "next_action": str(requirement_item.get("next_action", "")),
                "required_data": [str(value) for value in requirement_item.get("required_data", [])],
                "available_data": [str(value) for value in requirement_item.get("available_data", [])],
                "missing_data": [str(value) for value in requirement_item.get("missing_data", [])],
                "review_flags": review_flags,
            }
        )

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": items,
    }


def write_etf_methodology_summary(
    *,
    holdings_dir: Path = paths.REFRESHED_HOLDINGS_FILES_DIR,
    rules_path: Path = paths.FNGUIDE_RULES_JSON,
    requirements_path: Path = paths.FNGUIDE_REQUIREMENTS_JSON,
    audit_path: Path = paths.FNGUIDE_METHODOLOGY_AUDIT_JSON,
    output_dir: Path = paths.FNGUIDE_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    summary = build_etf_methodology_summary(
        holdings_dir=holdings_dir,
        rules_path=rules_path,
        requirements_path=requirements_path,
        audit_path=audit_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / paths.FNGUIDE_ETF_METHODOLOGY_SUMMARY_JSON.name
    csv_path = output_dir / paths.FNGUIDE_ETF_METHODOLOGY_SUMMARY_CSV.name
    md_path = output_dir / paths.FNGUIDE_ETF_METHODOLOGY_SUMMARY_MD.name
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary["items"], csv_path)
    md_path.write_text(_markdown(summary["items"]), encoding="utf-8")
    return json_path, csv_path, md_path


def _load_items(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [dict(_mapping(item)) for item in payload["items"]]
    return []


def _items_by_code(items: list[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    return {str(item.get("code", "")).removeprefix("A"): item for item in items if item.get("code")}


def _audit_blockers_by_index(path: Path) -> dict[str, list[str]]:
    blockers: dict[str, list[str]] = {}
    for item in _load_items(path):
        index_code = str(item.get("index_code", ""))
        blockers[index_code] = [str(value) for value in item.get("blockers", [])]
    return blockers


def _holdings_stats(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path, columns=["as_of"])
    dates = sorted(str(value) for value in frame["as_of"].dropna().unique())
    return {
        "holdings_row_count": int(len(frame)),
        "holdings_start_as_of": dates[0] if dates else "",
        "holdings_latest_as_of": dates[-1] if dates else "",
        "holdings_as_of_count": len(dates),
    }


def _methodology_pdf_status(
    rule_item: Mapping[str, object],
    requirement_item: Mapping[str, object],
) -> str:
    status = str(rule_item.get("status", ""))
    if status:
        return status
    requirement_status = str(requirement_item.get("methodology_status", ""))
    if requirement_status == "missing":
        return "missing"
    return "missing_rules"


def _structured_status(rule_item: Mapping[str, object], review_flags: list[str]) -> str:
    status = str(rule_item.get("status", ""))
    if status != "downloaded":
        return "missing_pdf"
    return "draft_review_required" if review_flags else "rules_extracted"


def _rebalance_date_status(
    rules: Mapping[str, object],
    requirement_item: Mapping[str, object],
) -> str:
    frequency = str(rules.get("review_frequency", ""))
    months = rules.get("review_months", [])
    if frequency in {"", "unknown"} or not months:
        return "missing_methodology_rule"
    missing_data = {str(value) for value in requirement_item.get("missing_data", [])}
    if {"krx_trading_calendar", "futures_options_expiry_calendar", "month_end_business_day_calendar"} & missing_data:
        return "requires_calendar"
    return "rule_available"


def _review_flags(
    rule_item: Mapping[str, object],
    rules: Mapping[str, object],
    requirement_item: Mapping[str, object],
    audit_blockers: list[str],
) -> list[str]:
    flags = list(audit_blockers)
    status = str(rule_item.get("status", ""))
    frequency = str(rules.get("review_frequency", ""))
    if status != "downloaded":
        flags.append("methodology_pdf_missing")
    if status == "downloaded" and (frequency in {"", "unknown"} or not rules.get("review_months")):
        flags.append("rebalance_rule_missing_or_ambiguous")
    if str(rules.get("weight_cap", "")) and "weight cap exists in rules but residual/fixed bucket scope is unresolved" not in flags:
        flags.append("weight_cap_scope_requires_review")
    for value in requirement_item.get("missing_data", []):
        if value == "methodology_pdf_missing":
            continue
        if value in {"krx_trading_calendar", "futures_options_expiry_calendar", "month_end_business_day_calendar"}:
            flags.append(f"date_resolution_missing:{value}")
    return sorted(set(flags))


def _index_code_from_page_url(url: str) -> str:
    path = urlparse(url).path
    return path.rstrip("/").split("/")[-1] if path else ""


def _write_csv(items: object, path: Path) -> None:
    rows = [_csv_row(item) for item in items if isinstance(item, Mapping)]
    fieldnames = list(rows[0]) if rows else [
        "etf_code",
        "etf_name",
        "holdings_file",
        "holdings_latest_as_of",
        "methodology_pdf_status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_row(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "etf_code": item.get("etf_code", ""),
        "etf_name": item.get("etf_name", ""),
        "holdings_file": item.get("holdings_file", ""),
        "holdings_latest_as_of": item.get("holdings_latest_as_of", ""),
        "holdings_row_count": item.get("holdings_row_count", ""),
        "methodology_pdf_status": item.get("methodology_pdf_status", ""),
        "methodology_structured_status": item.get("methodology_structured_status", ""),
        "index_code": item.get("index_code", ""),
        "index_name": item.get("index_name", ""),
        "rebalance_frequency": item.get("rebalance_frequency", ""),
        "rebalance_months": ",".join(str(value) for value in item.get("rebalance_months", [])),
        "rebalance_timing": item.get("rebalance_timing", ""),
        "rebalance_date_status": item.get("rebalance_date_status", ""),
        "weighting_scheme": item.get("weighting_scheme", ""),
        "weight_cap": item.get("weight_cap", ""),
        "readiness": item.get("readiness", ""),
        "next_action": item.get("next_action", ""),
        "missing_data": "|".join(str(value) for value in item.get("missing_data", [])),
        "review_flags": "|".join(str(value) for value in item.get("review_flags", [])),
    }


def _markdown(items: object) -> str:
    rows = [item for item in items if isinstance(item, Mapping)]
    status_counts: dict[str, int] = {}
    date_counts: dict[str, int] = {}
    for item in rows:
        status = str(item.get("methodology_pdf_status", ""))
        date_status = str(item.get("rebalance_date_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        date_counts[date_status] = date_counts.get(date_status, 0) + 1

    lines = [
        "# ETF Methodology Summary",
        "",
        f"- ETFs: {len(rows)}",
        f"- PDF status: {_counts_text(status_counts)}",
        f"- Rebalance date status: {_counts_text(date_counts)}",
        "",
        "| ETF | Index | PDF | Rebalance | Cap | Date status | Review flags |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        months = ",".join(str(value) for value in item.get("rebalance_months", []))
        rebalance = f"{item.get('rebalance_frequency', '')} {months} {item.get('rebalance_timing', '')}".strip()
        flags = "<br>".join(str(value) for value in item.get("review_flags", [])[:5])
        lines.append(
            f"| {item.get('etf_code')} {item.get('etf_name')} | {item.get('index_code')} {item.get('index_name')} | "
            f"{item.get('methodology_pdf_status')} | {rebalance} | {item.get('weight_cap')} | "
            f"{item.get('rebalance_date_status')} | {flags} |"
        )
    return "\n".join(lines) + "\n"


def _counts_text(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ETF-level FnGuide methodology summary from holdings and rules.")
    parser.add_argument("--holdings-dir", default=paths.REFRESHED_HOLDINGS_FILES_DIR.as_posix())
    parser.add_argument("--rules", default=paths.FNGUIDE_RULES_JSON.as_posix())
    parser.add_argument("--requirements", default=paths.FNGUIDE_REQUIREMENTS_JSON.as_posix())
    parser.add_argument("--audit", default=paths.FNGUIDE_METHODOLOGY_AUDIT_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path, csv_path, md_path = write_etf_methodology_summary(
        holdings_dir=Path(args.holdings_dir),
        rules_path=Path(args.rules),
        requirements_path=Path(args.requirements),
        audit_path=Path(args.audit),
        output_dir=Path(args.output_dir),
    )
    print(f"wrote {json_path}, {csv_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
