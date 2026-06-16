import json
from pathlib import Path

from etfs.fnguide.data_requirements import (
    build_requirement_records,
    infer_required_data,
    write_data_requirements,
)


def _rule_item(**overrides):
    item = {
        "code": "111111",
        "name": "SAMPLE ETF",
        "status": "downloaded",
        "source_url": "https://example.com/method.pdf",
        "page_url": "https://example.com",
        "file_path": "etfs/raw/methodologies/111111.pdf",
        "rule_set": {
            "index_name": "FnGuide Sample Index",
            "updated": "April 2026",
            "family": "keyword_theme",
            "rebalance": {"frequency": "semiannual", "months": [6, 12], "timing": "선물옵션 만기일 D+2"},
            "weighting": {"scheme": "score_weighted", "cap": "25%", "uses_free_float": True},
            "screening": {
                "selection_count": 20,
                "market_cap": True,
                "liquidity": True,
                "keyword": True,
                "fics": True,
            },
        },
    }
    item.update(overrides)
    return item


def test_infer_required_data_marks_rule_driven_inputs() -> None:
    required = infer_required_data(_rule_item())

    assert "methodology_pdf" in required
    assert "stock_prices" in required
    assert "free_float_ratio" in required
    assert "market_cap" in required
    assert "trading_value_liquidity" in required
    assert "keyword_source_documents" in required
    assert "fics_industry_classification" in required
    assert "score_inputs" in required
    assert "futures_options_expiry_calendar" in required


def test_build_requirement_records_preserves_methodology_file_and_missing_status() -> None:
    records = build_requirement_records([_rule_item(), _rule_item(code="222222", status="not_found", file_path="")])

    assert records[0].methodology_file == "etfs/raw/methodologies/111111.pdf"
    assert records[0].methodology_status == "available"
    assert "stock_prices" in records[0].missing_data
    assert records[1].methodology_status == "missing"
    assert "methodology_pdf_missing" in records[1].missing_data


def test_write_data_requirements_uses_simple_file_names(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"items": [_rule_item()]}, ensure_ascii=False), encoding="utf-8")

    csv_path, json_path, md_path = write_data_requirements(rules_path, tmp_path)

    assert [csv_path.name, json_path.name, md_path.name] == ["requirements.csv", "requirements.json", "requirements.md"]
    assert "stock_prices" in csv_path.read_text(encoding="utf-8-sig")
    assert "SAMPLE ETF" in md_path.read_text(encoding="utf-8")
