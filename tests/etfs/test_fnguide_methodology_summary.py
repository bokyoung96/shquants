import json
from pathlib import Path

import pandas as pd

from etfs.fnguide.methodology_summary import (
    build_etf_methodology_summary,
    write_etf_methodology_summary,
)


def test_build_etf_methodology_summary_joins_holdings_rules_and_requirements(tmp_path: Path) -> None:
    holdings_dir = tmp_path / "files"
    holdings_dir.mkdir()
    pd.DataFrame({"as_of": ["2026-06-15", "2026-06-16"], "weight": [0.1, 0.2]}).to_parquet(
        holdings_dir / "holdings_111111.parquet"
    )
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "code": "111111",
                        "name": "ETF A",
                        "status": "downloaded",
                        "file_path": "etfs/output/methodologies/111111.pdf",
                        "page_url": "https://www.fnindex.co.kr/overview/detail/I/FI00.TEST",
                        "rules": {
                            "index_name": "FnGuide Test Index",
                            "review_frequency": "quarterly",
                            "review_months": [3, 6, 9, 12],
                            "rebalance_timing": "D+2 after futures/options expiry",
                            "weighting_scheme": "float_market_cap_weighted",
                            "weight_cap": "25%",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "code": "111111",
                        "methodology_status": "available",
                        "missing_data": ["krx_trading_calendar", "futures_options_expiry_calendar"],
                        "required_data": ["krx_trading_calendar"],
                        "available_data": ["methodology_pdf"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "methodology_audit.json"
    audit_path.write_text(
        json.dumps({"items": [{"index_code": "FI00.TEST", "blockers": ["status=draft_extracted"]}]}),
        encoding="utf-8",
    )

    summary = build_etf_methodology_summary(
        holdings_dir=holdings_dir,
        rules_path=rules_path,
        requirements_path=requirements_path,
        audit_path=audit_path,
    )

    assert summary["count"] == 1
    item = summary["items"][0]
    assert item["etf_code"] == "111111"
    assert item["holdings_latest_as_of"] == "2026-06-16"
    assert item["index_code"] == "FI00.TEST"
    assert item["rebalance_frequency"] == "quarterly"
    assert item["rebalance_months"] == [3, 6, 9, 12]
    assert item["rebalance_date_status"] == "requires_calendar"
    assert item["weight_cap"] == "25%"
    assert item["methodology_structured_status"] == "draft_review_required"
    assert "status=draft_extracted" in item["review_flags"]


def test_write_etf_methodology_summary_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    holdings_dir = tmp_path / "files"
    holdings_dir.mkdir()
    pd.DataFrame({"as_of": ["2026-06-16"]}).to_parquet(holdings_dir / "holdings_222222.parquet")
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"items": []}), encoding="utf-8")
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_text(json.dumps({"items": []}), encoding="utf-8")

    json_path, csv_path, md_path = write_etf_methodology_summary(
        holdings_dir=holdings_dir,
        rules_path=rules_path,
        requirements_path=requirements_path,
        output_dir=tmp_path / "out",
    )

    assert json_path.name == "etf_methodology_summary.json"
    assert csv_path.name == "etf_methodology_summary.csv"
    assert md_path.name == "etf_methodology_summary.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["items"][0]["methodology_pdf_status"] == "missing_rules"
    assert "ETF Methodology Summary" in md_path.read_text(encoding="utf-8")
