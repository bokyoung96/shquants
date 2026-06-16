from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths


KRX_DATA_INDEX_URL = "https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd?locale=en"


@dataclass(frozen=True, slots=True)
class KrxMethodologyProbe:
    code: str
    name: str
    product_family: str
    source_confidence: str
    index_name_candidates: list[str]
    methodology_probe: str
    source_url: str
    next_action: str


def load_source_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported sources payload: {path}")
    return list(payload["items"])


def build_krx_records(items: Iterable[Mapping[str, object]]) -> list[KrxMethodologyProbe]:
    records: list[KrxMethodologyProbe] = []
    for item in items:
        if str(item.get("source_candidate", "")) != "krx":
            continue
        candidates = infer_krx_index_candidates(str(item.get("name", "")))
        records.append(
            KrxMethodologyProbe(
                code=str(item.get("code", "")),
                name=str(item.get("name", "")),
                product_family=str(item.get("product_family", "")),
                source_confidence=str(item.get("confidence", "")),
                index_name_candidates=candidates,
                methodology_probe="krx_index_data_system",
                source_url=KRX_DATA_INDEX_URL,
                next_action="probe_krx_index_data_system" if candidates else "manual_krx_index_mapping",
            )
        )
    return records


def infer_krx_index_candidates(name: str) -> list[str]:
    if "KRX금현물" in name or "금현물" in name:
        return ["KRX Gold Spot"]
    if "KRX기후변화솔루션" in name:
        return ["KRX Climate Change Solutions"]
    if "KRX300" in name or "KRX 300" in name:
        return ["KRX 300"]
    if "KRX100" in name or "KRX 100" in name:
        return ["KRX 100"]
    if "KTOP30" in name or "KTOP 30" in name:
        return ["KTOP 30"]
    if "코스닥글로벌" in name:
        return ["KOSDAQ Global"]
    if "코스닥TOP10" in name or "코스닥 TOP10" in name:
        return ["KOSDAQ Top 10"]
    if "코스닥150" in name or "KOSDAQ150" in name or "KOSDAQ 150" in name:
        return ["KOSDAQ 150"]
    if "코스닥" in name or "KOSDAQ" in name:
        return ["KOSDAQ"]
    if "중소형" in name:
        return ["KOSPI Small Cap"]
    if "코스피100" in name or "KOSPI100" in name or "KOSPI 100" in name:
        return ["KOSPI 100"]
    if "코스피" in name or "KOSPI" in name:
        return ["KOSPI"]
    if _looks_like_kospi_200(name):
        return ["KOSPI 200"]
    return []


def write_krx_manifest(sources_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    records = build_krx_records(load_source_items(sources_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "krx.csv"
    json_path = output_dir / "krx.json"
    md_path = output_dir / "krx.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_csv_row(records[0]).keys()) if records else list(KrxMethodologyProbe.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))

    payload = {
        "provider": "krx",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "mapped_count": sum(bool(record.index_name_candidates) for record in records),
        "index_candidate_counts": _index_candidate_counts(records),
        "items": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(records), encoding="utf-8")
    return csv_path, json_path, md_path


def _looks_like_kospi_200(name: str) -> bool:
    if "200" not in name:
        return False
    if "200exTOP" in name:
        return True
    return bool(re.search(r"(^|\s|[A-Z])200($|\s|[A-Z가-힣])", name)) or "K200" in name


def _csv_row(record: KrxMethodologyProbe) -> dict[str, object]:
    row = asdict(record)
    row["index_name_candidates"] = "|".join(record.index_name_candidates)
    return row


def _index_candidate_counts(records: list[KrxMethodologyProbe]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for candidate in record.index_name_candidates:
            counter[candidate] += 1
    return dict(counter)


def _markdown_summary(records: list[KrxMethodologyProbe]) -> str:
    candidate_counts = _index_candidate_counts(records)
    unmapped = [record for record in records if not record.index_name_candidates]
    lines = [
        "# KRX methodology probe manifest",
        "",
        f"- ETFs covered: {len(records)}",
        f"- Index candidates mapped: {len(records) - len(unmapped)}",
        f"- Manual KRX mappings: {len(unmapped)}",
        f"- Probe source: {KRX_DATA_INDEX_URL}",
        "",
        "## Index Candidates",
        "",
    ]
    for name, count in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Manual KRX Mapping", ""])
    if unmapped:
        lines.extend(["| code | name | family |", "| --- | --- | --- |"])
        for record in unmapped:
            lines.append(f"| {record.code} | {record.name} | {record.product_family} |")
    else:
        lines.append("- No KRX records require manual index mapping.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build KRX methodology probe manifest from source candidates.")
    parser.add_argument("--sources", default=paths.SOURCES_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.KRX_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, json_path, md_path = write_krx_manifest(Path(args.sources), Path(args.output_dir))
    print(f"wrote {csv_path}, {json_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
