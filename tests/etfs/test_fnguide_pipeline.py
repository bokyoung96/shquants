import json
from pathlib import Path

from etfs.fnguide import pipeline


def test_index_code_by_etf_from_specs_uses_all_products_without_provider_specific_defaults(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps(
            {
                "indices": [
                    {
                        "index_code": "FI00.EXAMPLE.A",
                        "products": [
                            {"etf_code": "111111", "etf_name": "Example ETF A"},
                            {"etf_code": "222222", "etf_name": "Example ETF B"},
                        ],
                    },
                    {"index_code": "FI00.EXAMPLE.B", "products": [{"etf_code": "333333"}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert pipeline.index_code_by_etf_from_specs(specs_path) == {
        "111111": "FI00.EXAMPLE.A",
        "222222": "FI00.EXAMPLE.A",
        "333333": "FI00.EXAMPLE.B",
    }


def test_run_offline_pipeline_sequences_generic_artifacts_and_explicit_validation_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(pipeline, "write_methodology_extractions", lambda rules_path, output_dir: _touch_pair(output_dir))
    monkeypatch.setattr(pipeline, "write_draft_specs", lambda extractions_path, output_dir: _touch(output_dir / "draft_specs.json"))

    def fake_write_methodology_specs(draft_path: Path, output_dir: Path, *, overrides_path: Path) -> Path:
        calls.append(f"specs:{draft_path.name}:{overrides_path.name}")
        path = output_dir / "methodology_specs.json"
        path.write_text(
            json.dumps({"indices": [{"index_code": "FI00.EXAMPLE", "products": [{"etf_code": "123456"}]}]}),
            encoding="utf-8",
        )
        return path

    def fake_write_validation_fixtures(workbook_paths, output_dir: Path, *, index_code_by_etf):
        calls.append(f"fixtures:{len(list(workbook_paths))}:{index_code_by_etf['123456']}")
        return _touch(output_dir / "validation_fixtures.json")

    monkeypatch.setattr(pipeline, "write_methodology_specs", fake_write_methodology_specs)
    monkeypatch.setattr(pipeline, "write_methodology_audit", lambda specs_path, output_dir: _touch_pair(output_dir, "methodology_audit"))
    monkeypatch.setattr(pipeline, "write_validation_fixtures", fake_write_validation_fixtures)
    monkeypatch.setattr(pipeline, "write_validation_results", lambda fixtures_path, specs_path, output_dir: _touch(output_dir / "validation_results.json"))
    monkeypatch.setattr(
        pipeline,
        "write_cap_candidate_report",
        lambda fixtures_path, specs_path, output_dir: (
            _touch(output_dir / "cap_candidates.json"),
            _touch(output_dir / "cap_candidates.md"),
        ),
    )
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
        "write_fnguide_data_inventory",
        lambda output_dir, *, specs_path: (_touch(output_dir / "data_inventory.json"), _touch(output_dir / "data_inventory.md")),
    )
    monkeypatch.setattr(
        pipeline,
        "write_methodology_replication_report",
        lambda specs_path, output_dir: (
            _touch(output_dir / "methodology_replication_report.json"),
            _touch(output_dir / "methodology_replication_report.md"),
        ),
    )
    monkeypatch.setattr(pipeline, "write_target_weights", lambda inputs_path, specs_path, output_dir: _touch(output_dir / "target_weights.json"))
    monkeypatch.setattr(
        pipeline,
        "write_target_weight_validation_results",
        lambda fixtures_path, target_weights_path, output_dir, *, weight_tolerance: _touch(output_dir / "target_weight_validation.json"),
    )

    rules_path = _touch(tmp_path / "rules.json")
    overrides_path = _touch(tmp_path / "spec_overrides.json")
    validation_path = _touch(tmp_path / "holdings.xlsx")
    engine_inputs_path = _touch(tmp_path / "engine" / "engine_inputs.json")

    manifest = pipeline.run_offline_pipeline(
        rules_path=rules_path,
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=overrides_path,
        validation_inputs=[validation_path],
        validation_output_dir=tmp_path / "validation",
        engine_output_dir=tmp_path / "engine",
        engine_inputs_path=engine_inputs_path,
        inventory_output_dir=tmp_path / "methodology",
    )

    assert calls == ["specs:draft_specs.json:spec_overrides.json", "fixtures:1:FI00.EXAMPLE"]
    assert manifest["outputs"]["data_inventory"].endswith("data_inventory.json")
    assert manifest["outputs"]["cap_candidates"].endswith("cap_candidates.json")
    assert manifest["outputs"]["cap_candidates_md"].endswith("cap_candidates.md")
    assert manifest["outputs"]["methodology_replication_report"].endswith("methodology_replication_report.json")
    assert manifest["outputs"]["target_weights"].endswith("target_weights.json")
    assert manifest["outputs"]["target_weight_validation"].endswith("target_weight_validation.json")
    assert all(not key.startswith("kss_") for key in manifest["outputs"])
    assert manifest["skipped"] == []


def test_run_offline_pipeline_skips_optional_inputs_without_named_etf_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "write_methodology_extractions", lambda rules_path, output_dir: _touch_pair(output_dir))
    monkeypatch.setattr(pipeline, "write_draft_specs", lambda extractions_path, output_dir: _touch(output_dir / "draft_specs.json"))
    monkeypatch.setattr(
        pipeline,
        "write_methodology_specs",
        lambda draft_path, output_dir, *, overrides_path: _touch_specs(output_dir / "methodology_specs.json"),
    )
    monkeypatch.setattr(pipeline, "write_methodology_audit", lambda specs_path, output_dir: _touch_pair(output_dir, "methodology_audit"))
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
        "write_fnguide_data_inventory",
        lambda output_dir, *, specs_path: (_touch(output_dir / "data_inventory.json"), _touch(output_dir / "data_inventory.md")),
    )
    monkeypatch.setattr(
        pipeline,
        "write_methodology_replication_report",
        lambda specs_path, output_dir: (
            _touch(output_dir / "methodology_replication_report.json"),
            _touch(output_dir / "methodology_replication_report.md"),
        ),
    )

    manifest = pipeline.run_offline_pipeline(
        rules_path=_touch(tmp_path / "rules.json"),
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=_touch(tmp_path / "spec_overrides.json"),
        validation_inputs=[],
        validation_output_dir=tmp_path / "validation",
        engine_output_dir=tmp_path / "engine",
        engine_inputs_path=tmp_path / "engine" / "engine_inputs.json",
        inventory_output_dir=tmp_path / "methodology",
    )

    assert "validation_fixtures" not in manifest["outputs"]
    assert manifest["skipped"] == [
        "validation: input workbooks not provided",
        "target_weights: engine_inputs not found; fill engine_inputs.template.json",
    ]


def test_pipeline_parser_does_not_default_to_a_validation_fixture_or_kss_snapshot() -> None:
    args = pipeline.build_parser().parse_args([])

    assert args.rules == "etfs/output/methodology/fnguide/rules.json"
    assert args.validation_input == []
    assert args.engine_output_dir == "etfs/output/methodology/fnguide"
    assert args.inventory_output_dir == "etfs/output/methodology/fnguide"
    assert not hasattr(args, "kss_snapshot")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _touch_specs(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"indices": []}', encoding="utf-8")
    return path


def _touch_pair(output_dir: Path, stem: str = "methodology_extractions") -> tuple[Path, Path]:
    return _touch(output_dir / f"{stem}.json"), _touch(output_dir / f"{stem}.md")
