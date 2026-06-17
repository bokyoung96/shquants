import json
from pathlib import Path

from etfs.fnguide.coverage import build_coverage_records, write_fnguide_coverage


def _requirement_item(**overrides):
    item = {
        "code": "111111",
        "name": "SAMPLE ETF",
        "index_name": "FnGuide Sample Index",
        "methodology_status": "available",
        "methodology_file": "etfs/output/methodologies/111111.pdf",
        "source_url": "https://example.com/sample.pdf",
        "family": "keyword_theme",
        "rebalance_frequency": "semiannual",
        "rebalance_months": "6,12",
        "weighting_scheme": "score_weighted",
        "weight_cap": "25%",
        "required_data": [
            "methodology_pdf",
            "rule_profile",
            "stock_prices",
            "keyword_source_documents",
            "score_inputs",
        ],
        "available_data": ["methodology_pdf", "rule_profile"],
        "missing_data": ["stock_prices", "keyword_source_documents", "score_inputs"],
    }
    item.update(overrides)
    return item


def test_build_coverage_records_groups_fnguide_next_actions() -> None:
    records = build_coverage_records(
        [
            _requirement_item(),
            _requirement_item(
                code="222222",
                methodology_status="missing",
                methodology_file="",
                missing_data=["methodology_pdf_missing"],
            ),
            _requirement_item(
                code="333333",
                family="dividend",
                weighting_scheme="custom_weighted",
                missing_data=["stock_prices", "dividend_data", "custom_index_formula_inputs"],
            ),
        ]
    )

    assert records[0].readiness == "needs_external_model_data"
    assert records[0].next_action == "collect_theme_keyword_and_score_inputs"
    assert records[1].readiness == "blocked_missing_pdf"
    assert records[1].next_action == "find_fnguide_methodology_pdf"
    assert records[2].readiness == "needs_dividend_or_custom_data"
    assert records[2].next_action == "collect_dividend_and_custom_formula_inputs"


def test_write_fnguide_coverage_uses_simple_fnguide_file_names(tmp_path: Path) -> None:
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_text(
        json.dumps({"items": [_requirement_item()]}, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path, json_path, md_path = write_fnguide_coverage(requirements_path, tmp_path)

    assert [csv_path.name, json_path.name, md_path.name] == ["fnguide.csv", "fnguide.json", "fnguide.md"]
    assert "needs_external_model_data" in csv_path.read_text(encoding="utf-8-sig")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "fnguide"
    assert payload["count"] == 1
    assert "FnGuide first-pass coverage" in md_path.read_text(encoding="utf-8")

