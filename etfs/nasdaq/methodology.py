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


NASDAQ_METHODOLOGY_LIBRARY_URL = "https://indexes.nasdaq.com/Resource/Index/Methodology"
NASDAQ_100_METHODOLOGY_URL = "https://indexes.nasdaq.com/docs/Methodology_NDX.pdf"
NASDAQ_NEXT_GEN_100_METHODOLOGY_URL = "https://indexes.nasdaq.com/docs/Methodology_NGX.pdf"
NASDAQ_BIOTECH_METHODOLOGY_URL = "https://indexes.nasdaqomx.com/docs/methodology_nbi.pdf"
NASDAQ_CLEAN_EDGE_METHODOLOGY_URL = "https://indexes.nasdaqomx.com/docs/methodology_cels.pdf"
NASDAQ_DIVIDEND_ACHIEVERS_METHODOLOGY_URL = "https://indexes.nasdaqomx.com/docs/methodology_dvg.pdf"
PHLX_SEMICONDUCTOR_METHODOLOGY_URL = "https://indexes.nasdaqomx.com/docs/methodology_sox.pdf"


@dataclass(frozen=True, slots=True)
class NasdaqMethodologyProbe:
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


def build_nasdaq_records(items: Iterable[Mapping[str, object]]) -> list[NasdaqMethodologyProbe]:
    records: list[NasdaqMethodologyProbe] = []
    for item in items:
        if str(item.get("source_candidate", "")) != "nasdaq":
            continue
        name = str(item.get("name", ""))
        candidates = infer_nasdaq_index_candidates(name)
        records.append(
            NasdaqMethodologyProbe(
                code=str(item.get("code", "")),
                name=name,
                product_family=str(item.get("product_family", "")),
                source_confidence=str(item.get("confidence", "")),
                index_name_candidates=candidates,
                methodology_probe="nasdaq_index_methodology_library",
                methodology_url_candidates=infer_methodology_urls(candidates),
                next_action="probe_nasdaq_methodology_library" if candidates else "manual_nasdaq_index_mapping",
            )
        )
    return records


def infer_nasdaq_index_candidates(name: str) -> list[str]:
    if "필라델피아" in name or "반도체나스닥" in name:
        return ["PHLX Semiconductor Sector"]
    if "넥스트100" in name:
        return ["Nasdaq Next Generation 100"]
    if "바이오" in name:
        return ["Nasdaq Biotechnology"]
    if "클린에너지" in name:
        return ["Nasdaq Clean Edge Green Energy"]
    if "배당" in name:
        return ["Nasdaq US Dividend Achievers"]
    if "나스닥100" in name or "NASDAQ100" in name or "NASDAQ 100" in name:
        return ["Nasdaq-100"]
    if "나스닥" in name or "NASDAQ" in name:
        return ["Nasdaq-100"]
    return []


def infer_methodology_urls(index_name_candidates: list[str]) -> list[str]:
    urls: list[str] = []
    for candidate in index_name_candidates:
        if candidate == "Nasdaq-100":
            urls.append(NASDAQ_100_METHODOLOGY_URL)
        elif candidate == "Nasdaq Next Generation 100":
            urls.append(NASDAQ_NEXT_GEN_100_METHODOLOGY_URL)
        elif candidate == "Nasdaq Biotechnology":
            urls.append(NASDAQ_BIOTECH_METHODOLOGY_URL)
        elif candidate == "Nasdaq Clean Edge Green Energy":
            urls.append(NASDAQ_CLEAN_EDGE_METHODOLOGY_URL)
        elif candidate == "Nasdaq US Dividend Achievers":
            urls.append(NASDAQ_DIVIDEND_ACHIEVERS_METHODOLOGY_URL)
        elif candidate == "PHLX Semiconductor Sector":
            urls.append(PHLX_SEMICONDUCTOR_METHODOLOGY_URL)
    return urls


def write_nasdaq_manifest(sources_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    records = build_nasdaq_records(load_source_items(sources_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nasdaq.csv"
    json_path = output_dir / "nasdaq.json"
    md_path = output_dir / "nasdaq.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_csv_row(records[0]).keys()) if records else list(NasdaqMethodologyProbe.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))

    payload = {
        "provider": "nasdaq",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "mapped_count": sum(bool(record.index_name_candidates) for record in records),
        "index_candidate_counts": _index_candidate_counts(records),
        "methodology_library_url": NASDAQ_METHODOLOGY_LIBRARY_URL,
        "items": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(records), encoding="utf-8")
    return csv_path, json_path, md_path


def _csv_row(record: NasdaqMethodologyProbe) -> dict[str, object]:
    row = asdict(record)
    row["index_name_candidates"] = "|".join(record.index_name_candidates)
    row["methodology_url_candidates"] = "|".join(record.methodology_url_candidates)
    return row


def _index_candidate_counts(records: list[NasdaqMethodologyProbe]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for candidate in record.index_name_candidates:
            counter[candidate] += 1
    return dict(counter)


def _markdown_summary(records: list[NasdaqMethodologyProbe]) -> str:
    candidate_counts = _index_candidate_counts(records)
    unmapped = [record for record in records if not record.index_name_candidates]
    lines = [
        "# Nasdaq methodology probe manifest",
        "",
        f"- ETFs covered: {len(records)}",
        f"- Index candidates mapped: {len(records) - len(unmapped)}",
        f"- Manual Nasdaq mappings: {len(unmapped)}",
        f"- Methodology library: {NASDAQ_METHODOLOGY_LIBRARY_URL}",
        "",
        "## Index Candidates",
        "",
    ]
    for name, count in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Manual Nasdaq Mapping", ""])
    if unmapped:
        lines.extend(["| code | name | family |", "| --- | --- | --- |"])
        for record in unmapped:
            lines.append(f"| {record.code} | {record.name} | {record.product_family} |")
    else:
        lines.append("- No Nasdaq records require manual index mapping.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Nasdaq methodology probe manifest from source candidates.")
    parser.add_argument("--sources", default=paths.SOURCES_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.NASDAQ_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, json_path, md_path = write_nasdaq_manifest(Path(args.sources), Path(args.output_dir))
    print(f"wrote {csv_path}, {json_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
