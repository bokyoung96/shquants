import json
from pathlib import Path

from etfs.fnguide.methodology_audit import (
    build_methodology_audit,
    build_methodology_review_queue,
    write_methodology_audit,
)


def test_build_methodology_audit_marks_only_verified_specs_engine_ready() -> None:
    specs = [
        {
            "index_code": "FI00.READY",
            "index_name": "Ready Index",
            "status": "methodology_verified",
            "selection": {"total_constituents": 10},
            "open_questions": [],
        },
        {
            "index_code": "FI00.DRAFT",
            "index_name": "Draft Index",
            "status": "draft_extracted",
            "selection": {"total_constituents": None},
            "open_questions": ["selection.total_constituents not extracted with evidence"],
        },
    ]

    audit = build_methodology_audit(specs)

    assert audit["counts"]["total"] == 2
    assert audit["counts"]["engine_ready"] == 1
    assert audit["counts"]["blocked"] == 1
    assert audit["items"][0]["engine_ready"] is True
    assert audit["items"][1]["engine_ready"] is False
    assert audit["items"][1]["blockers"] == ["status=draft_extracted", "selection.total_constituents not extracted with evidence"]


def test_build_methodology_audit_accepts_verified_variable_count_specs() -> None:
    audit = build_methodology_audit(
        [
            {
                "index_code": "FI00.VARIABLE",
                "index_name": "Variable Count Index",
                "status": "methodology_verified",
                "selection": {
                    "total_constituents": None,
                    "variable_count": {"method": "cumulative_weight_threshold", "threshold": 0.95},
                    "min_constituents": 10,
                },
                "open_questions": [],
            }
        ]
    )

    assert audit["counts"]["engine_ready"] == 1
    assert audit["items"][0]["engine_ready"] is True
    assert audit["items"][0]["variable_count"] == {"method": "cumulative_weight_threshold", "threshold": 0.95}
    assert audit["items"][0]["blockers"] == []


def test_write_methodology_audit_outputs_json_and_markdown(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps(
            {
                "indices": [
                    {
                        "index_code": "FI00.DRAFT",
                        "index_name": "Draft Index",
                        "status": "draft_extracted",
                        "selection": {"total_constituents": None},
                        "open_questions": ["missing selection"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    json_path, md_path = write_methodology_audit(specs_path, tmp_path)

    assert json_path.name == "methodology_audit.json"
    assert md_path.name == "methodology_audit.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["counts"]["blocked"] == 1
    assert "Draft Index" in md_path.read_text(encoding="utf-8")


def test_build_methodology_review_queue_classifies_unresolved_specs() -> None:
    specs = [
        {
            "index_code": "FI00.RANGE",
            "index_name": "Range Index",
            "status": "draft_extracted",
            "products": [{"etf_code": "000001", "etf_name": "Range ETF"}],
            "selection": {"total_constituents": None, "min_constituents": 10, "max_constituents": 15},
            "weighting": {"base": "float_market_cap_weighted", "residual": {}, "security_cap": 0.15},
            "open_questions": ["selection.total_constituents not extracted with evidence"],
        },
        {
            "index_code": "FI00.TIER",
            "index_name": "Tiered Cap Index",
            "status": "draft_extracted",
            "products": [],
            "selection": {"total_constituents": 10},
            "weighting": {"base": "float_market_cap_weighted", "residual": {}},
            "open_questions": ["weight cap exists in rules but residual/fixed bucket scope is unresolved"],
        },
    ]

    queue = build_methodology_review_queue(specs)

    assert queue["counts"] == {"total": 2, "by_category": {"range_or_max_count": 1, "tiered_or_unresolved_weight_cap": 1}}
    assert queue["items"][0]["category"] == "range_or_max_count"
    assert queue["items"][0]["next_action"] == "verify whether range/max count is executable or needs a variable-count rule"
    assert queue["items"][0]["products"] == [{"etf_code": "000001", "etf_name": "Range ETF"}]
    assert queue["items"][1]["category"] == "tiered_or_unresolved_weight_cap"
