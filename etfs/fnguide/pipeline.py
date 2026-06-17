from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths
from etfs.common.cap import write_cap_candidate_report
from etfs.fnguide.data_requirements import write_data_requirements
from etfs.fnguide.methodology_audit import write_methodology_audit
from etfs.fnguide.methodology_extraction import write_methodology_extractions
from etfs.fnguide.methodology_summary import write_etf_methodology_summary
from etfs.fnguide.methodology_specs import write_draft_specs, write_methodology_specs
from etfs.fnguide.validation import (
    write_validation_fixtures,
    write_validation_results,
)


def run_offline_pipeline(
    *,
    rules_path: Path = paths.FNGUIDE_RULES_JSON,
    extraction_output_dir: Path = paths.FNGUIDE_EXTRACTION_OUTPUT_DIR,
    overrides_path: Path = paths.FNGUIDE_SPEC_OVERRIDES_JSON,
    validation_inputs: Iterable[Path] = (),
    validation_output_dir: Path = paths.VALIDATION_OUTPUT_DIR,
    inventory_output_dir: Path = paths.FNGUIDE_OUTPUT_DIR,
    holdings_dir: Path = paths.REFRESHED_HOLDINGS_FILES_DIR,
) -> dict[str, object]:
    extraction_output_dir.mkdir(parents=True, exist_ok=True)
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    inventory_output_dir.mkdir(parents=True, exist_ok=True)

    extractions_json, extractions_md = write_methodology_extractions(rules_path, extraction_output_dir)
    draft_specs = write_draft_specs(extractions_json, extraction_output_dir)
    methodology_specs = write_methodology_specs(
        draft_specs,
        extraction_output_dir,
        overrides_path=overrides_path,
    )
    audit_json, audit_md = write_methodology_audit(methodology_specs, extraction_output_dir)
    requirements_csv, requirements_json, requirements_md = write_data_requirements(rules_path, inventory_output_dir)
    summary_json, summary_csv, summary_md = write_etf_methodology_summary(
        holdings_dir=holdings_dir,
        rules_path=rules_path,
        requirements_path=requirements_json,
        audit_path=audit_json,
        output_dir=inventory_output_dir,
    )

    outputs: dict[str, str] = {
        "methodology_extractions": extractions_json.as_posix(),
        "methodology_extractions_md": extractions_md.as_posix(),
        "draft_specs": draft_specs.as_posix(),
        "methodology_specs": methodology_specs.as_posix(),
        "methodology_audit": audit_json.as_posix(),
        "methodology_audit_md": audit_md.as_posix(),
        "requirements": requirements_json.as_posix(),
        "requirements_csv": requirements_csv.as_posix(),
        "requirements_md": requirements_md.as_posix(),
        "etf_methodology_summary": summary_json.as_posix(),
        "etf_methodology_summary_csv": summary_csv.as_posix(),
        "etf_methodology_summary_md": summary_md.as_posix(),
    }
    skipped: list[str] = []

    validation_paths = [path for path in validation_inputs if path.exists()]
    fixtures: Path | None = None
    if validation_paths:
        fixtures = write_validation_fixtures(
            validation_paths,
            validation_output_dir,
            index_code_by_etf=index_code_by_etf_from_specs(methodology_specs),
        )
        validation_results = write_validation_results(fixtures, methodology_specs, validation_output_dir)
        cap_candidates_json, cap_candidates_md = write_cap_candidate_report(fixtures, methodology_specs, validation_output_dir)
        outputs["validation_fixtures"] = fixtures.as_posix()
        outputs["validation_results"] = validation_results.as_posix()
        outputs["cap_candidates"] = cap_candidates_json.as_posix()
        outputs["cap_candidates_md"] = cap_candidates_md.as_posix()
    else:
        skipped.append("validation: input workbooks not provided")

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
        "skipped": skipped,
    }


def index_code_by_etf_from_specs(specs_path: Path) -> dict[str, str]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for spec in payload.get("indices", []):
        if not isinstance(spec, Mapping):
            continue
        index_code = str(spec.get("index_code", "")).strip()
        if not index_code:
            continue
        products = spec.get("products", [])
        if not isinstance(products, list):
            continue
        for product in products:
            if not isinstance(product, Mapping):
                continue
            etf_code = str(product.get("etf_code", "")).strip()
            if etf_code:
                result[etf_code] = index_code
    return result


def write_offline_pipeline_manifest(
    output_path: Path,
    **kwargs: object,
) -> Path:
    manifest = run_offline_pipeline(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline FnGuide methodology-to-engine artifact pipeline.")
    parser.add_argument("--rules", default=paths.FNGUIDE_RULES_JSON.as_posix())
    parser.add_argument("--extraction-output-dir", default=paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix())
    parser.add_argument("--overrides", default=paths.FNGUIDE_SPEC_OVERRIDES_JSON.as_posix())
    parser.add_argument("--validation-input", nargs="*", default=[])
    parser.add_argument("--validation-output-dir", default=paths.VALIDATION_OUTPUT_DIR.as_posix())
    parser.add_argument("--inventory-output-dir", default=paths.FNGUIDE_OUTPUT_DIR.as_posix())
    parser.add_argument("--holdings-dir", default=paths.REFRESHED_HOLDINGS_FILES_DIR.as_posix())
    parser.add_argument("--manifest", default=(paths.FNGUIDE_OUTPUT_DIR / "offline_pipeline_manifest.json").as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = write_offline_pipeline_manifest(
        Path(args.manifest),
        rules_path=Path(args.rules),
        extraction_output_dir=Path(args.extraction_output_dir),
        overrides_path=Path(args.overrides),
        validation_inputs=[Path(value) for value in args.validation_input],
        validation_output_dir=Path(args.validation_output_dir),
        inventory_output_dir=Path(args.inventory_output_dir),
        holdings_dir=Path(args.holdings_dir),
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
