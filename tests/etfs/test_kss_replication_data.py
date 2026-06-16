import json
from pathlib import Path

from etfs.fnguide.replication_data import (
    KSS_REQUIRED_DATASETS,
    build_kss_data_requirements,
    require_kss_snapshot_fields,
    write_kss_data_requirements,
)


def test_kss_required_datasets_cover_full_replication_chain() -> None:
    names = [item["name"] for item in KSS_REQUIRED_DATASETS]

    assert names == [
        "methodology_spec",
        "rebalance_calendar",
        "security_master",
        "eligibility_flags",
        "market_snapshot",
        "classification_snapshot",
        "selection_metrics",
        "official_target_weights",
        "etf_holdings_snapshot",
    ]


def test_build_kss_data_requirements_marks_missing_datasets() -> None:
    requirements = build_kss_data_requirements(available_datasets={"methodology_spec", "market_snapshot"})

    assert requirements["index_code"] == "FI00.WLT.KSS"
    assert requirements["replication_stage"] == "data_contract"
    assert requirements["available_datasets"] == ["market_snapshot", "methodology_spec"]
    assert "selection_metrics" in requirements["missing_datasets"]
    assert requirements["full_replication_ready"] is False


def test_require_kss_snapshot_fields_rejects_missing_required_fields() -> None:
    rows = [{"security_code": "A000001", "float_market_cap": 100.0}]

    errors = require_kss_snapshot_fields(rows)

    assert errors == [
        {
            "row": 0,
            "security_code": "A000001",
            "missing_fields": [
                "as_of",
                "is_eligible",
                "is_semiconductor_theme",
                "composite_momentum_score",
            ],
        }
    ]


def test_write_kss_data_requirements_outputs_json(tmp_path: Path) -> None:
    output_path = write_kss_data_requirements(tmp_path, available_datasets={"methodology_spec"})

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "kss_data_requirements.json"
    assert payload["index_code"] == "FI00.WLT.KSS"
    assert payload["full_replication_ready"] is False
    assert "methodology_spec" in payload["available_datasets"]
