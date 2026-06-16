import json
from pathlib import Path

from etfs.fnguide import pipeline


def test_run_offline_pipeline_sequences_verified_artifacts_and_skips_missing_engine_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_write_methodology_extractions(rules_path: Path, output_dir: Path) -> tuple[Path, Path]:
        calls.append(f"extractions:{rules_path.name}")
        json_path = output_dir / "methodology_extractions.json"
        md_path = output_dir / "methodology_extractions.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("", encoding="utf-8")
        return json_path, md_path

    def fake_write_draft_specs(extractions_path: Path, output_dir: Path) -> Path:
        calls.append(f"draft:{extractions_path.name}")
        path = output_dir / "draft_specs.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def fake_write_methodology_specs(draft_path: Path, output_dir: Path, *, overrides_path: Path) -> Path:
        calls.append(f"specs:{draft_path.name}:{overrides_path.name}")
        path = output_dir / "methodology_specs.json"
        path.write_text('{"indices": []}', encoding="utf-8")
        return path

    def fake_write_methodology_audit(specs_path: Path, output_dir: Path) -> tuple[Path, Path]:
        calls.append(f"audit:{specs_path.name}")
        json_path = output_dir / "methodology_audit.json"
        md_path = output_dir / "methodology_audit.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("", encoding="utf-8")
        return json_path, md_path

    def fake_write_validation_fixtures(workbook_paths, output_dir: Path, *, index_code_by_etf):
        calls.append(f"fixtures:{len(list(workbook_paths))}:{index_code_by_etf['0167A0']}")
        path = output_dir / "validation_fixtures.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def fake_write_validation_results(fixtures_path: Path, specs_path: Path, output_dir: Path) -> Path:
        calls.append(f"validation:{fixtures_path.name}:{specs_path.name}")
        path = output_dir / "validation_results.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def fake_write_engine_input_requirements(specs_path: Path, output_dir: Path) -> Path:
        calls.append(f"requirements:{specs_path.name}")
        path = output_dir / "engine_input_requirements.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def fake_write_engine_input_template(specs_path: Path, output_dir: Path) -> Path:
        calls.append(f"input_template:{specs_path.name}")
        path = output_dir / "engine_inputs.template.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def fake_write_engine_support_matrix(specs_path: Path, output_dir: Path) -> tuple[Path, Path]:
        calls.append(f"support:{specs_path.name}")
        json_path = output_dir / "engine_support_matrix.json"
        md_path = output_dir / "engine_support_matrix.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("", encoding="utf-8")
        return json_path, md_path

    def fake_write_engine_promotion_candidates(specs_path: Path, output_dir: Path) -> tuple[Path, Path]:
        calls.append(f"promotion:{specs_path.name}")
        json_path = output_dir / "engine_promotion_candidates.json"
        md_path = output_dir / "engine_promotion_candidates.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("", encoding="utf-8")
        return json_path, md_path

    def fake_write_methodology_replication_report(
        specs_path: Path,
        output_dir: Path,
        *,
        kss_replication_validation_path: Path,
    ) -> tuple[Path, Path]:
        calls.append(f"replication:{specs_path.name}")
        json_path = output_dir / "methodology_replication_report.json"
        md_path = output_dir / "methodology_replication_report.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("", encoding="utf-8")
        return json_path, md_path

    def fake_write_target_weights(inputs_path: Path, specs_path: Path, output_dir: Path) -> Path:
        calls.append(f"targets:{inputs_path.name}:{specs_path.name}")
        path = output_dir / "target_weights.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def fake_write_target_weight_validation_results(
        fixtures_path: Path,
        target_weights_path: Path,
        output_dir: Path,
        *,
        weight_tolerance: float,
    ) -> Path:
        calls.append(f"target_validation:{fixtures_path.name}:{target_weights_path.name}:{weight_tolerance}")
        path = output_dir / "target_weight_validation.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(pipeline, "write_methodology_extractions", fake_write_methodology_extractions)
    monkeypatch.setattr(pipeline, "write_draft_specs", fake_write_draft_specs)
    monkeypatch.setattr(pipeline, "write_methodology_specs", fake_write_methodology_specs)
    monkeypatch.setattr(pipeline, "write_methodology_audit", fake_write_methodology_audit)
    monkeypatch.setattr(pipeline, "write_validation_fixtures", fake_write_validation_fixtures)
    monkeypatch.setattr(pipeline, "write_validation_results", fake_write_validation_results)
    monkeypatch.setattr(pipeline, "write_engine_input_requirements", fake_write_engine_input_requirements)
    monkeypatch.setattr(pipeline, "write_engine_input_template", fake_write_engine_input_template)
    monkeypatch.setattr(pipeline, "write_engine_support_matrix", fake_write_engine_support_matrix)
    monkeypatch.setattr(pipeline, "write_engine_promotion_candidates", fake_write_engine_promotion_candidates)
    monkeypatch.setattr(pipeline, "write_methodology_replication_report", fake_write_methodology_replication_report)
    monkeypatch.setattr(pipeline, "write_target_weights", fake_write_target_weights)
    monkeypatch.setattr(pipeline, "write_target_weight_validation_results", fake_write_target_weight_validation_results)

    rules_path = tmp_path / "rules.json"
    overrides_path = tmp_path / "spec_overrides.json"
    validation_path = tmp_path / "validation_A0167A0.xlsx"
    engine_inputs_path = tmp_path / "engine" / "engine_inputs.json"
    rules_path.write_text("{}", encoding="utf-8")
    overrides_path.write_text("{}", encoding="utf-8")
    validation_path.write_text("", encoding="utf-8")
    engine_inputs_path.parent.mkdir(parents=True)
    engine_inputs_path.write_text("{}", encoding="utf-8")

    manifest = pipeline.run_offline_pipeline(
        rules_path=rules_path,
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=overrides_path,
        validation_inputs=[validation_path],
        validation_output_dir=tmp_path / "validation",
        engine_output_dir=tmp_path / "engine",
        engine_inputs_path=engine_inputs_path,
    )

    assert calls == [
        "extractions:rules.json",
        "draft:methodology_extractions.json",
        "specs:draft_specs.json:spec_overrides.json",
        "audit:methodology_specs.json",
        "fixtures:1:FI00.WLT.KSS",
        "validation:validation_fixtures.json:methodology_specs.json",
        "requirements:methodology_specs.json",
        "input_template:methodology_specs.json",
        "support:methodology_specs.json",
        "promotion:methodology_specs.json",
        "replication:methodology_specs.json",
        "targets:engine_inputs.json:methodology_specs.json",
        "target_validation:validation_fixtures.json:target_weights.json:0.0",
    ]
    assert manifest["outputs"]["methodology_specs"].endswith("methodology_specs.json")
    assert manifest["outputs"]["engine_input_requirements"].endswith("engine_input_requirements.json")
    assert manifest["outputs"]["engine_input_template"].endswith("engine_inputs.template.json")
    assert manifest["outputs"]["engine_support_matrix"].endswith("engine_support_matrix.json")
    assert manifest["outputs"]["engine_promotion_candidates"].endswith("engine_promotion_candidates.json")
    assert manifest["outputs"]["methodology_replication_report"].endswith("methodology_replication_report.json")
    assert manifest["outputs"]["kss_data_requirements"].endswith("kss_data_requirements.json")
    assert manifest["outputs"]["target_weights"].endswith("target_weights.json")
    assert manifest["outputs"]["target_weight_validation"].endswith("target_weight_validation.json")
    assert manifest["skipped"] == ["kss_replication: kss_snapshot not found"]


def test_run_offline_pipeline_skips_target_validation_when_engine_inputs_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(pipeline, "write_methodology_extractions", lambda rules_path, output_dir: _touch_pair(output_dir))
    monkeypatch.setattr(pipeline, "write_draft_specs", lambda extractions_path, output_dir: _touch(output_dir / "draft_specs.json"))
    monkeypatch.setattr(
        pipeline,
        "write_methodology_specs",
        lambda draft_path, output_dir, *, overrides_path: _touch(output_dir / "methodology_specs.json"),
    )
    monkeypatch.setattr(pipeline, "write_methodology_audit", lambda specs_path, output_dir: _touch_pair(output_dir))
    monkeypatch.setattr(pipeline, "write_engine_input_requirements", lambda specs_path, output_dir: _touch(output_dir / "engine_input_requirements.json"))
    monkeypatch.setattr(pipeline, "write_engine_input_template", lambda specs_path, output_dir: _touch(output_dir / "engine_inputs.template.json"))
    monkeypatch.setattr(
        pipeline,
        "write_engine_support_matrix",
        lambda specs_path, output_dir: (_touch(output_dir / "engine_support_matrix.json"), _touch(output_dir / "engine_support_matrix.md")),
    )
    monkeypatch.setattr(
        pipeline,
        "write_engine_promotion_candidates",
        lambda specs_path, output_dir: (
            _touch(output_dir / "engine_promotion_candidates.json"),
            _touch(output_dir / "engine_promotion_candidates.md"),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_methodology_replication_report",
        lambda specs_path, output_dir, *, kss_replication_validation_path: (
            _touch(output_dir / "methodology_replication_report.json"),
            _touch(output_dir / "methodology_replication_report.md"),
        ),
    )

    rules_path = _touch(tmp_path / "rules.json")
    overrides_path = _touch(tmp_path / "spec_overrides.json")

    manifest = pipeline.run_offline_pipeline(
        rules_path=rules_path,
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=overrides_path,
        validation_inputs=[],
        validation_output_dir=tmp_path / "validation",
        engine_output_dir=tmp_path / "engine",
        engine_inputs_path=tmp_path / "engine" / "engine_inputs.json",
    )

    assert manifest["outputs"]["engine_input_template"].endswith("engine_inputs.template.json")
    assert manifest["outputs"]["methodology_replication_report"].endswith("methodology_replication_report.json")
    assert manifest["outputs"]["kss_data_requirements"].endswith("kss_data_requirements.json")
    assert manifest["skipped"] == [
        "validation: input workbooks not found",
        "kss_replication: kss_snapshot not found",
        "target_weights: engine_inputs not found; fill engine_inputs.template.json",
    ]


def test_offline_pipeline_writes_kss_data_requirements_and_skips_missing_snapshot(tmp_path: Path) -> None:
    manifest = pipeline.run_offline_pipeline(
        rules_path=Path("etfs/output/providers/fnguide/rules.json"),
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=Path("etfs/output/extractions/fnguide/spec_overrides.json"),
        validation_inputs=[],
        validation_output_dir=tmp_path / "validation",
        engine_output_dir=tmp_path / "engine",
        engine_inputs_path=tmp_path / "engine" / "engine_inputs.json",
        replication_output_dir=tmp_path / "replication",
        kss_snapshot_path=tmp_path / "replication" / "kss_snapshot.json",
    )

    assert manifest["outputs"]["kss_data_requirements"] == (
        tmp_path / "replication" / "kss_data_requirements.json"
    ).as_posix()
    assert "kss_replication: kss_snapshot not found" in manifest["skipped"]


def test_offline_pipeline_keeps_kss_and_engine_target_weight_outputs_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(pipeline, "write_methodology_extractions", lambda rules_path, output_dir: _touch_pair(output_dir))
    monkeypatch.setattr(pipeline, "write_draft_specs", lambda extractions_path, output_dir: _touch(output_dir / "draft_specs.json"))
    monkeypatch.setattr(
        pipeline,
        "write_methodology_specs",
        lambda draft_path, output_dir, *, overrides_path: _touch(output_dir / "methodology_specs.json"),
    )
    monkeypatch.setattr(pipeline, "write_methodology_audit", lambda specs_path, output_dir: _touch_pair(output_dir))
    monkeypatch.setattr(pipeline, "write_engine_input_requirements", lambda specs_path, output_dir: _touch(output_dir / "engine_input_requirements.json"))
    monkeypatch.setattr(pipeline, "write_engine_input_template", lambda specs_path, output_dir: _touch(output_dir / "engine_inputs.template.json"))
    monkeypatch.setattr(
        pipeline,
        "write_engine_support_matrix",
        lambda specs_path, output_dir: (_touch(output_dir / "engine_support_matrix.json"), _touch(output_dir / "engine_support_matrix.md")),
    )
    monkeypatch.setattr(
        pipeline,
        "write_engine_promotion_candidates",
        lambda specs_path, output_dir: (
            _touch(output_dir / "engine_promotion_candidates.json"),
            _touch(output_dir / "engine_promotion_candidates.md"),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_methodology_replication_report",
        lambda specs_path, output_dir, *, kss_replication_validation_path: (
            _touch(output_dir / "methodology_replication_report.json"),
            _touch(output_dir / "methodology_replication_report.md"),
        ),
    )
    monkeypatch.setattr(pipeline, "write_target_weights", lambda inputs_path, specs_path, output_dir: _touch(output_dir / "target_weights.json"))
    monkeypatch.setattr(
        pipeline,
        "build_kss_replication",
        lambda **kwargs: {
            "kwargs": kwargs,
            "target_weight_result": {"target_weights": []},
            "validation": {},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "write_kss_replication_artifacts",
        lambda result, output_dir: {
            "selected_buckets": (output_dir / "kss_selected_buckets.json").as_posix(),
            "target_weights": (output_dir / "kss_target_weights.json").as_posix(),
            "replication_validation": (output_dir / "kss_replication_validation.json").as_posix(),
            "replication_validation_md": (output_dir / "kss_replication_validation.md").as_posix(),
        },
    )

    rules_path = _touch(tmp_path / "rules.json")
    overrides_path = _touch(tmp_path / "spec_overrides.json")
    engine_inputs_path = _touch(tmp_path / "engine" / "engine_inputs.json")
    snapshot_path = tmp_path / "replication" / "kss_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "as_of": "2026-05-29",
                "effective_date": "2026-06-14",
                "rows": [{"security_code": "A000001"}],
                "validation_weights": [{"security_code": "A000001", "official_weight": 1.0}],
                "validation_source_type": "official_target_weights",
            }
        ),
        encoding="utf-8",
    )

    manifest = pipeline.run_offline_pipeline(
        rules_path=rules_path,
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=overrides_path,
        validation_inputs=[],
        validation_output_dir=tmp_path / "validation",
        engine_output_dir=tmp_path / "engine",
        engine_inputs_path=engine_inputs_path,
        replication_output_dir=tmp_path / "replication",
        kss_snapshot_path=snapshot_path,
    )

    assert manifest["outputs"]["kss_selected_buckets"] == (tmp_path / "replication" / "kss_selected_buckets.json").as_posix()
    assert manifest["outputs"]["kss_target_weights"] == (tmp_path / "replication" / "kss_target_weights.json").as_posix()
    assert manifest["outputs"]["kss_replication_validation"] == (tmp_path / "replication" / "kss_replication_validation.json").as_posix()
    assert manifest["outputs"]["kss_replication_validation_md"] == (tmp_path / "replication" / "kss_replication_validation.md").as_posix()
    assert manifest["outputs"]["target_weights"] == (tmp_path / "engine" / "target_weights.json").as_posix()
    assert "kss_replication: kss_snapshot not found" not in manifest["skipped"]


def test_offline_pipeline_replication_report_uses_current_run_kss_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed_validation_paths: list[Path] = []

    monkeypatch.setattr(pipeline, "write_methodology_extractions", lambda rules_path, output_dir: _touch_pair(output_dir))
    monkeypatch.setattr(pipeline, "write_draft_specs", lambda extractions_path, output_dir: _touch(output_dir / "draft_specs.json"))
    monkeypatch.setattr(
        pipeline,
        "write_methodology_specs",
        lambda draft_path, output_dir, *, overrides_path: _touch(output_dir / "methodology_specs.json"),
    )
    monkeypatch.setattr(pipeline, "write_methodology_audit", lambda specs_path, output_dir: _touch_pair(output_dir))
    monkeypatch.setattr(pipeline, "write_engine_input_requirements", lambda specs_path, output_dir: _touch(output_dir / "engine_input_requirements.json"))
    monkeypatch.setattr(pipeline, "write_engine_input_template", lambda specs_path, output_dir: _touch(output_dir / "engine_inputs.template.json"))
    monkeypatch.setattr(
        pipeline,
        "write_engine_support_matrix",
        lambda specs_path, output_dir: (_touch(output_dir / "engine_support_matrix.json"), _touch(output_dir / "engine_support_matrix.md")),
    )
    monkeypatch.setattr(
        pipeline,
        "write_engine_promotion_candidates",
        lambda specs_path, output_dir: (
            _touch(output_dir / "engine_promotion_candidates.json"),
            _touch(output_dir / "engine_promotion_candidates.md"),
        ),
    )

    def fake_write_methodology_replication_report(
        specs_path: Path,
        output_dir: Path,
        *,
        kss_replication_validation_path: Path,
    ) -> tuple[Path, Path]:
        observed_validation_paths.append(kss_replication_validation_path)
        assert kss_replication_validation_path == tmp_path / "replication" / "kss_replication_validation.json"
        assert kss_replication_validation_path.exists()
        return _touch(output_dir / "methodology_replication_report.json"), _touch(output_dir / "methodology_replication_report.md")

    def fake_write_kss_replication_artifacts(result, output_dir: Path) -> dict[str, str]:
        return {
            "selected_buckets": _touch(output_dir / "kss_selected_buckets.json").as_posix(),
            "target_weights": _touch(output_dir / "kss_target_weights.json").as_posix(),
            "replication_validation": _touch(output_dir / "kss_replication_validation.json").as_posix(),
            "replication_validation_md": _touch(output_dir / "kss_replication_validation.md").as_posix(),
        }

    monkeypatch.setattr(pipeline, "write_methodology_replication_report", fake_write_methodology_replication_report)
    monkeypatch.setattr(pipeline, "build_kss_replication", lambda **kwargs: {"validation": {}})
    monkeypatch.setattr(pipeline, "write_kss_replication_artifacts", fake_write_kss_replication_artifacts)

    rules_path = _touch(tmp_path / "rules.json")
    overrides_path = _touch(tmp_path / "spec_overrides.json")
    snapshot_path = tmp_path / "replication" / "kss_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "as_of": "2026-05-29",
                "effective_date": "2026-06-14",
                "rows": [],
                "validation_weights": [],
                "validation_source_type": "official_target_weights",
            }
        ),
        encoding="utf-8",
    )

    pipeline.run_offline_pipeline(
        rules_path=rules_path,
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=overrides_path,
        validation_inputs=[],
        validation_output_dir=tmp_path / "validation",
        engine_output_dir=tmp_path / "engine",
        engine_inputs_path=tmp_path / "engine" / "engine_inputs.json",
        replication_output_dir=tmp_path / "replication",
        kss_snapshot_path=snapshot_path,
    )

    assert observed_validation_paths == [tmp_path / "replication" / "kss_replication_validation.json"]


def test_pipeline_parser_defaults_to_offline_artifacts() -> None:
    args = pipeline.build_parser().parse_args([])

    assert args.rules == "etfs/output/providers/fnguide/rules.json"
    assert args.extraction_output_dir == "etfs/output/extractions/fnguide"
    assert args.validation_input == ["etfs/validation_A0167A0.xlsx"]
    assert args.engine_output_dir == "etfs/output/engine/fnguide"
    assert args.replication_output_dir == "etfs/output/replication/fnguide"
    assert args.kss_snapshot == "etfs/output/replication/fnguide/kss_snapshot.json"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _touch_pair(output_dir: Path) -> tuple[Path, Path]:
    return _touch(output_dir / "methodology_extractions.json"), _touch(output_dir / "methodology_extractions.md")
