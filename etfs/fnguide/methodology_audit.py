from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths


ENGINE_READY_STATUSES = {"methodology_verified", "validated_constituents_against_etf_holdings", "validated_against_official"}


def build_methodology_audit(specs: Iterable[Mapping[str, object]]) -> dict[str, object]:
    items = [_audit_item(spec) for spec in specs]
    status_counts = Counter(str(item["status"]) for item in items)
    blocker_counts: Counter[str] = Counter()
    for item in items:
        blocker_counts.update(str(blocker) for blocker in item["blockers"])
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "total": len(items),
            "engine_ready": sum(bool(item["engine_ready"]) for item in items),
            "blocked": sum(not bool(item["engine_ready"]) for item in items),
        },
        "status_counts": dict(status_counts),
        "blocker_counts": dict(blocker_counts),
        "items": items,
    }


def build_methodology_review_queue(specs: Iterable[Mapping[str, object]]) -> dict[str, object]:
    items = []
    for spec in specs:
        item = _review_queue_item(spec)
        if item is not None:
            items.append(item)
    category_counts = Counter(str(item["category"]) for item in items)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {"total": len(items), "by_category": dict(category_counts)},
        "items": items,
    }


def write_methodology_audit(specs_path: Path, output_dir: Path) -> tuple[Path, Path]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    specs = payload.get("indices", [])
    audit = build_methodology_audit(specs)
    review_queue = build_methodology_review_queue(specs)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "methodology_audit.json"
    md_path = output_dir / "methodology_audit.md"
    queue_json_path = output_dir / "methodology_review_queue.json"
    queue_md_path = output_dir / "methodology_review_queue.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(audit), encoding="utf-8")
    queue_json_path.write_text(json.dumps(review_queue, ensure_ascii=False, indent=2), encoding="utf-8")
    queue_md_path.write_text(_review_queue_markdown(review_queue), encoding="utf-8")
    return json_path, md_path


def _audit_item(spec: Mapping[str, object]) -> dict[str, object]:
    status = str(spec.get("status", ""))
    selection = _mapping(spec.get("selection"))
    open_questions = [str(item) for item in spec.get("open_questions", [])]
    blockers: list[str] = []
    if status not in ENGINE_READY_STATUSES:
        blockers.append(f"status={status}")
    if (
        selection.get("total_constituents") is None
        and not _mapping(selection.get("variable_count"))
        and not any("selection.total_constituents" in item for item in open_questions)
    ):
        blockers.append("selection.total_constituents missing")
    blockers.extend(open_questions)
    return {
        "index_code": str(spec.get("index_code", "")),
        "index_name": str(spec.get("index_name", "")),
        "status": status,
        "engine_ready": not blockers,
        "total_constituents": selection.get("total_constituents"),
        "variable_count": dict(_mapping(selection.get("variable_count"))),
        "blockers": blockers,
    }


def _review_queue_item(spec: Mapping[str, object]) -> dict[str, object] | None:
    audit_item = _audit_item(spec)
    blockers = [str(item) for item in audit_item["blockers"]]
    if not blockers:
        return None
    selection = _mapping(spec.get("selection"))
    weighting = _mapping(spec.get("weighting"))
    category, next_action = _review_category_and_action(selection, weighting, blockers)
    return {
        "index_code": str(spec.get("index_code", "")),
        "index_name": str(spec.get("index_name", "")),
        "status": str(spec.get("status", "")),
        "category": category,
        "next_action": next_action,
        "products": [dict(_mapping(product)) for product in spec.get("products", [])],
        "selection": dict(selection),
        "weighting": dict(weighting),
        "blockers": blockers,
    }


def _review_category_and_action(
    selection: Mapping[str, object],
    weighting: Mapping[str, object],
    blockers: list[str],
) -> tuple[str, str]:
    if any("weight cap exists" in blocker for blocker in blockers):
        return "tiered_or_unresolved_weight_cap", "parse tiered/fixed bucket cap scope before engine enablement"
    if selection.get("min_constituents") is not None or selection.get("max_constituents") is not None:
        return "range_or_max_count", "verify whether range/max count is executable or needs a variable-count rule"
    if _mapping(selection.get("variable_count")):
        return "variable_count_review", "verify variable-count implementation against official constituents"
    if str(weighting.get("base", "")) in {"unknown", ""}:
        return "missing_core_methodology", "recover index code or methodology PDF before extraction"
    if any("selection.total_constituents" in blocker for blocker in blockers):
        return "missing_selection_count", "inspect selection section and encode exact/range/variable-count rule"
    return "status_review", "review draft status and promote only after methodology evidence is complete"


def _markdown(audit: Mapping[str, object]) -> str:
    counts = _mapping(audit.get("counts"))
    lines = [
        "# FnGuide Methodology Audit",
        "",
        f"- Total specs: {counts.get('total', 0)}",
        f"- Engine ready: {counts.get('engine_ready', 0)}",
        f"- Blocked: {counts.get('blocked', 0)}",
        "",
        "## Blocked Specs",
        "",
        "| index_code | index_name | status | blockers |",
        "| --- | --- | --- | --- |",
    ]
    for item in audit.get("items", []):
        if item.get("engine_ready"):
            continue
        blockers = "<br>".join(str(value) for value in item.get("blockers", []))
        lines.append(f"| {item.get('index_code')} | {item.get('index_name')} | {item.get('status')} | {blockers} |")
    return "\n".join(lines) + "\n"


def _review_queue_markdown(review_queue: Mapping[str, object]) -> str:
    counts = _mapping(review_queue.get("counts"))
    lines = [
        "# FnGuide Methodology Review Queue",
        "",
        f"- Total review items: {counts.get('total', 0)}",
        "",
        "| index_code | index_name | category | next_action | blockers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in review_queue.get("items", []):
        blockers = "<br>".join(str(value) for value in item.get("blockers", []))
        lines.append(
            f"| {item.get('index_code')} | {item.get('index_name')} | {item.get('category')} | "
            f"{item.get('next_action')} | {blockers} |"
        )
    return "\n".join(lines) + "\n"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit canonical FnGuide methodology specs for engine readiness.")
    parser.add_argument("--specs", default=paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path, md_path = write_methodology_audit(Path(args.specs), Path(args.output_dir))
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
