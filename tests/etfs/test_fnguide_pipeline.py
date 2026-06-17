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
    monkeypatch.setattr(
        pipeline,
        "write_data_requirements",
        lambda rules_path, output_dir: (
            _touch(output_dir / "requirements.csv"),
            _touch(output_dir / "requirements.json"),
            _touch(output_dir / "requirements.md"),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_etf_methodology_summary",
        lambda *, holdings_dir, rules_path, requirements_path, audit_path, output_dir: (
            _touch(output_dir / "etf_methodology_summary.json"),
            _touch(output_dir / "etf_methodology_summary.csv"),
            _touch(output_dir / "etf_methodology_summary.md"),
        ),
    )
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

    rules_path = _touch(tmp_path / "rules.json")
    overrides_path = _touch(tmp_path / "spec_overrides.json")
    validation_path = _touch(tmp_path / "holdings.xlsx")

    manifest = pipeline.run_offline_pipeline(
        rules_path=rules_path,
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=overrides_path,
        validation_inputs=[validation_path],
        validation_output_dir=tmp_path / "validation",
        inventory_output_dir=tmp_path / "methodology",
        holdings_dir=tmp_path / "files",
    )

    assert calls == ["specs:draft_specs.json:spec_overrides.json", "fixtures:1:FI00.EXAMPLE"]
    assert manifest["outputs"]["requirements"].endswith("requirements.json")
    assert manifest["outputs"]["etf_methodology_summary"].endswith("etf_methodology_summary.json")
    assert manifest["outputs"]["cap_candidates"].endswith("cap_candidates.json")
    assert manifest["outputs"]["cap_candidates_md"].endswith("cap_candidates.md")
    assert "methodology_replication_report" not in manifest["outputs"]
    assert "target_weights" not in manifest["outputs"]
    assert "target_weight_validation" not in manifest["outputs"]
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
    monkeypatch.setattr(
        pipeline,
        "write_data_requirements",
        lambda rules_path, output_dir: (
            _touch(output_dir / "requirements.csv"),
            _touch(output_dir / "requirements.json"),
            _touch(output_dir / "requirements.md"),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_etf_methodology_summary",
        lambda *, holdings_dir, rules_path, requirements_path, audit_path, output_dir: (
            _touch(output_dir / "etf_methodology_summary.json"),
            _touch(output_dir / "etf_methodology_summary.csv"),
            _touch(output_dir / "etf_methodology_summary.md"),
        ),
    )

    manifest = pipeline.run_offline_pipeline(
        rules_path=_touch(tmp_path / "rules.json"),
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=_touch(tmp_path / "spec_overrides.json"),
        validation_inputs=[],
        validation_output_dir=tmp_path / "validation",
        inventory_output_dir=tmp_path / "methodology",
        holdings_dir=tmp_path / "files",
    )

    assert "validation_fixtures" not in manifest["outputs"]
    assert "target_weights" not in manifest["outputs"]
    assert manifest["skipped"] == ["validation: input workbooks not provided"]


def test_pipeline_parser_does_not_default_to_a_validation_fixture_or_kss_snapshot() -> None:
    args = pipeline.build_parser().parse_args([])

    assert args.rules == "etfs/output/methodology/fnguide/rules.json"
    assert args.validation_input == []
    assert args.inventory_output_dir == "etfs/output/methodology/fnguide"
    assert args.holdings_dir == "etfs/output/files"
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
