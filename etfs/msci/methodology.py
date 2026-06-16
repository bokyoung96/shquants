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


MSCI_METHODOLOGY_LIBRARY_URL = "https://www.msci.com/indexes/index-resources/index-methodology"
MSCI_GIMI_METHODOLOGY_URL = "https://www.msci.com/documents/10199/ab796822-b8bf-9122-5e67-cfb93af723c9"
MSCI_UNIVERSAL_METHODOLOGY_URL = "https://www.msci.com/indexes/documents/methodology/4_MSCI_Universal_Indexes_Methodology_20241113.pdf"
MSCI_ESG_LEADERS_METHODOLOGY_URL = "https://www.msci.com/documents/10199/ead0ff4d-e776-b418-b6fa-581a2ff82d30"
MSCI_US_REIT_METHODOLOGY_URL = "https://www.msci.com/eqb/methodology/meth_docs/MSCI_US_REIT_Methodology_Jan2023_QCIR.pdf"


@dataclass(frozen=True, slots=True)
class MsciMethodologyProbe:
    code: str
    name: str
    product_family: str
    source_confidence: str
    index_name_candidates: list[str]
    methodology_probe: str
    methodology_url_candidates: list[str]
    next_action: str


def load_source_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported sources payload: {path}")
    return list(payload["items"])


def build_msci_records(items: Iterable[Mapping[str, object]]) -> list[MsciMethodologyProbe]:
    records: list[MsciMethodologyProbe] = []
    for item in items:
        if str(item.get("source_candidate", "")) != "msci":
            continue
        name = str(item.get("name", ""))
        candidates = infer_msci_index_candidates(name)
        records.append(
            MsciMethodologyProbe(
                code=str(item.get("code", "")),
                name=name,
                product_family=str(item.get("product_family", "")),
                source_confidence=str(item.get("confidence", "")),
                index_name_candidates=candidates,
                methodology_probe="msci_methodology_library",
                methodology_url_candidates=infer_methodology_urls(candidates),
                next_action="probe_msci_methodology_library" if candidates else "manual_msci_index_mapping",
            )
        )
    return records


def infer_msci_index_candidates(name: str) -> list[str]:
    upper_name = name.upper()
    if "ESG유니버설" in name or "ESG UNIVERSAL" in upper_name:
        return ["MSCI Korea ESG Universal"]
    if "ESG리더스" in name or "ESG LEADERS" in upper_name:
        return ["MSCI Korea ESG Leaders"]
    if "리츠" in name or "REIT" in upper_name:
        return ["MSCI US REIT"]
    if "글로벌워터" in name or "GLOBAL WATER" in upper_name:
        return ["MSCI ACWI IMI Water ESG Filtered"]
    if "차이나2차전지" in name or "중국" in name or "CHINA" in upper_name:
        return ["MSCI China"]
    if "신흥국" in name or "이머징" in name or " MSCI EM" in upper_name:
        return ["MSCI Emerging Markets"]
    if "선진국" in name or "DEVELOPED" in upper_name:
        return ["MSCI World"]
    if "글로벌MSCI" in name or "GLOBAL MSCI" in upper_name:
        return ["MSCI ACWI"]
    if "멕시코" in name or "MEXICO" in upper_name:
        return ["MSCI Mexico"]
    if "인도네시아" in name or "INDONESIA" in upper_name:
        return ["MSCI Indonesia"]
    if "필리핀" in name or "PHILIPPINES" in upper_name:
        return ["MSCI Philippines"]
    if "러시아" in name or "RUSSIA" in upper_name:
        return ["MSCI Russia"]
    if "KOREA" in upper_name:
        return ["MSCI Korea"]
    return []


def infer_methodology_urls(index_name_candidates: list[str]) -> list[str]:
    urls: list[str] = []
    for candidate in index_name_candidates:
        if candidate in {
            "MSCI ACWI",
            "MSCI China",
            "MSCI Emerging Markets",
            "MSCI Indonesia",
            "MSCI Korea",
            "MSCI Mexico",
            "MSCI Philippines",
            "MSCI Russia",
            "MSCI World",
        }:
            urls.append(MSCI_GIMI_METHODOLOGY_URL)
        elif candidate == "MSCI Korea ESG Universal":
            urls.extend([MSCI_GIMI_METHODOLOGY_URL, MSCI_UNIVERSAL_METHODOLOGY_URL])
        elif candidate == "MSCI Korea ESG Leaders":
            urls.extend([MSCI_GIMI_METHODOLOGY_URL, MSCI_ESG_LEADERS_METHODOLOGY_URL])
        elif candidate == "MSCI US REIT":
            urls.extend([MSCI_GIMI_METHODOLOGY_URL, MSCI_US_REIT_METHODOLOGY_URL])
        elif candidate == "MSCI ACWI IMI Water ESG Filtered":
            urls.append(MSCI_GIMI_METHODOLOGY_URL)
    return _dedupe(urls)


def write_msci_manifest(sources_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    records = build_msci_records(load_source_items(sources_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "msci.csv"
    json_path = output_dir / "msci.json"
    md_path = output_dir / "msci.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_csv_row(records[0]).keys()) if records else list(MsciMethodologyProbe.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))

    payload = {
        "provider": "msci",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "mapped_count": sum(bool(record.index_name_candidates) for record in records),
        "index_candidate_counts": _index_candidate_counts(records),
        "methodology_library_url": MSCI_METHODOLOGY_LIBRARY_URL,
        "items": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(records), encoding="utf-8")
    return csv_path, json_path, md_path


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _csv_row(record: MsciMethodologyProbe) -> dict[str, object]:
    row = asdict(record)
    row["index_name_candidates"] = "|".join(record.index_name_candidates)
    row["methodology_url_candidates"] = "|".join(record.methodology_url_candidates)
    return row


def _index_candidate_counts(records: list[MsciMethodologyProbe]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for candidate in record.index_name_candidates:
            counter[candidate] += 1
    return dict(counter)


def _markdown_summary(records: list[MsciMethodologyProbe]) -> str:
    candidate_counts = _index_candidate_counts(records)
    unmapped = [record for record in records if not record.index_name_candidates]
    lines = [
        "# MSCI methodology probe manifest",
        "",
        f"- ETFs covered: {len(records)}",
        f"- Index candidates mapped: {len(records) - len(unmapped)}",
        f"- Manual MSCI mappings: {len(unmapped)}",
        f"- Methodology library: {MSCI_METHODOLOGY_LIBRARY_URL}",
        "",
        "## Index Candidates",
        "",
    ]
    for name, count in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Manual MSCI Mapping", ""])
    if unmapped:
        lines.extend(["| code | name | family |", "| --- | --- | --- |"])
        for record in unmapped:
            lines.append(f"| {record.code} | {record.name} | {record.product_family} |")
    else:
        lines.append("- No MSCI records require manual index mapping.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MSCI methodology probe manifest from source candidates.")
    parser.add_argument("--sources", default=paths.SOURCES_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.MSCI_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, json_path, md_path = write_msci_manifest(Path(args.sources), Path(args.output_dir))
    print(f"wrote {csv_path}, {json_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
