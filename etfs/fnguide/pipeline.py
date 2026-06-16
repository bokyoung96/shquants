from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from etfs import paths
from etfs.fnguide.methodology_audit import write_methodology_audit
from etfs.fnguide.methodology_engine import (
    write_engine_input_requirements,
    write_engine_input_template,
    write_engine_promotion_candidates,
    write_engine_support_matrix,
    write_methodology_replication_report,
    write_target_weights,
)
from etfs.fnguide.methodology_extraction import write_methodology_extractions
from etfs.fnguide.methodology_specs import write_draft_specs, write_methodology_specs
from etfs.fnguide.replication import build_kss_replication, write_kss_replication_artifacts
from etfs.fnguide.replication_data import write_kss_data_requirements
from etfs.fnguide.validation import (
    write_target_weight_validation_results,
    write_validation_fixtures,
    write_validation_results,
)


DEFAULT_INDEX_CODE_BY_ETF = {"0167A0": "FI00.WLT.KSS"}


def run_offline_pipeline(
    *,
    rules_path: Path = paths.FNGUIDE_RULES_JSON,
    extraction_output_dir: Path = paths.FNGUIDE_EXTRACTION_OUTPUT_DIR,
    overrides_path: Path = paths.FNGUIDE_SPEC_OVERRIDES_JSON,
    validation_inputs: Iterable[Path] = (Path("etfs/validation_A0167A0.xlsx"),),
    validation_output_dir: Path = paths.VALIDATION_OUTPUT_DIR,
    engine_output_dir: Path = paths.FNGUIDE_ENGINE_OUTPUT_DIR,
    engine_inputs_path: Path = paths.FNGUIDE_ENGINE_INPUTS_JSON,
    replication_output_dir: Path = paths.FNGUIDE_REPLICATION_OUTPUT_DIR,
    kss_snapshot_path: Path = paths.FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_snapshot.json",
) -> dict[str, object]:
    extraction_output_dir.mkdir(parents=True, exist_ok=True)
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    engine_output_dir.mkdir(parents=True, exist_ok=True)
    replication_output_dir.mkdir(parents=True, exist_ok=True)

    extractions_json, extractions_md = write_methodology_extractions(rules_path, extraction_output_dir)
    draft_specs = write_draft_specs(extractions_json, extraction_output_dir)
    methodology_specs = write_methodology_specs(
        draft_specs,
        extraction_output_dir,
        overrides_path=overrides_path,
    )
    audit_json, audit_md = write_methodology_audit(methodology_specs, extraction_output_dir)

    outputs: dict[str, str] = {
        "methodology_extractions": extractions_json.as_posix(),
        "methodology_extractions_md": extractions_md.as_posix(),
        "draft_specs": draft_specs.as_posix(),
        "methodology_specs": methodology_specs.as_posix(),
        "methodology_audit": audit_json.as_posix(),
        "methodology_audit_md": audit_md.as_posix(),
    }
    skipped: list[str] = []

    validation_paths = [path for path in validation_inputs if path.exists()]
    fixtures: Path | None = None
    if validation_paths:
        fixtures = write_validation_fixtures(
            validation_paths,
            validation_output_dir,
            index_code_by_etf=DEFAULT_INDEX_CODE_BY_ETF,
        )
        validation_results = write_validation_results(fixtures, methodology_specs, validation_output_dir)
        outputs["validation_fixtures"] = fixtures.as_posix()
        outputs["validation_results"] = validation_results.as_posix()
    else:
        skipped.append("validation: input workbooks not found")

    engine_requirements = write_engine_input_requirements(methodology_specs, engine_output_dir)
    outputs["engine_input_requirements"] = engine_requirements.as_posix()
    engine_input_template = write_engine_input_template(methodology_specs, engine_output_dir)
    outputs["engine_input_template"] = engine_input_template.as_posix()
    engine_support_json, engine_support_md = write_engine_support_matrix(methodology_specs, engine_output_dir)
    outputs["engine_support_matrix"] = engine_support_json.as_posix()
    outputs["engine_support_matrix_md"] = engine_support_md.as_posix()
    promotion_json, promotion_md = write_engine_promotion_candidates(methodology_specs, engine_output_dir)
    outputs["engine_promotion_candidates"] = promotion_json.as_posix()
    outputs["engine_promotion_candidates_md"] = promotion_md.as_posix()
    replication_json, replication_md = write_methodology_replication_report(methodology_specs, engine_output_dir)
    outputs["methodology_replication_report"] = replication_json.as_posix()
    outputs["methodology_replication_report_md"] = replication_md.as_posix()
    kss_requirements = write_kss_data_requirements(
        replication_output_dir,
        available_datasets={"methodology_spec"} | ({"etf_holdings_snapshot"} if fixtures is not None else set()),
    )
    outputs["kss_data_requirements"] = kss_requirements.as_posix()

    if kss_snapshot_path.exists():
        snapshot_payload = json.loads(kss_snapshot_path.read_text(encoding="utf-8"))
        kss_result = build_kss_replication(
            as_of=str(snapshot_payload.get("as_of", "")),
            effective_date=str(snapshot_payload.get("effective_date", "")),
            snapshot_rows=snapshot_payload.get("rows", []),
            validation_weights=snapshot_payload.get("validation_weights", []),
            validation_source_type=str(snapshot_payload.get("validation_source_type", "missing")),
            specs_path=methodology_specs,
        )
        outputs.update(write_kss_replication_artifacts(kss_result, replication_output_dir))
    else:
        skipped.append("kss_replication: kss_snapshot not found")

    if engine_inputs_path.exists():
        target_weights = write_target_weights(engine_inputs_path, methodology_specs, engine_output_dir)
        outputs["target_weights"] = target_weights.as_posix()
        if fixtures is not None:
            target_validation = write_target_weight_validation_results(
                fixtures,
                target_weights,
                validation_output_dir,
                weight_tolerance=0.0,
            )
            outputs["target_weight_validation"] = target_validation.as_posix()
        else:
            skipped.append("target_weight_validation: validation fixtures not found")
    else:
        skipped.append("target_weights: engine_inputs not found; fill engine_inputs.template.json")

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
        "skipped": skipped,
    }


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
    parser.add_argument("--validation-input", nargs="*", default=["etfs/validation_A0167A0.xlsx"])
    parser.add_argument("--validation-output-dir", default=paths.VALIDATION_OUTPUT_DIR.as_posix())
    parser.add_argument("--engine-output-dir", default=paths.FNGUIDE_ENGINE_OUTPUT_DIR.as_posix())
    parser.add_argument("--engine-inputs", default=paths.FNGUIDE_ENGINE_INPUTS_JSON.as_posix())
    parser.add_argument("--replication-output-dir", default=paths.FNGUIDE_REPLICATION_OUTPUT_DIR.as_posix())
    parser.add_argument(
        "--kss-snapshot",
        default=(paths.FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_snapshot.json").as_posix(),
    )
    parser.add_argument("--manifest", default=(paths.FNGUIDE_ENGINE_OUTPUT_DIR / "offline_pipeline_manifest.json").as_posix())
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
        engine_output_dir=Path(args.engine_output_dir),
        engine_inputs_path=Path(args.engine_inputs),
        replication_output_dir=Path(args.replication_output_dir),
        kss_snapshot_path=Path(args.kss_snapshot),
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
