import json

import pytest

from etfs.fnguide.replication import (
    build_replication_validation,
    write_replication_validation,
)


def test_build_replication_validation_passes_exact_official_targets() -> None:
    calculated = [
        {"security_code": "A000001", "target_weight": 0.25},
        {"security_code": "A000002", "target_weight": 0.25},
    ]
    official = [
        {"security_code": "A000001", "official_weight": 0.25},
        {"security_code": "A000002", "official_weight": 0.25},
    ]

    result = build_replication_validation(
        index_code="FI00.WLT.KSS",
        as_of="2026-05-29",
        validation_source_type="official_target_weights",
        calculated_target_weights=calculated,
        validation_weights=official,
        weight_tolerance=0.0,
    )

    assert result["status"] == "passed"
    assert result["checks"] == {
        "constituent_membership": "passed",
        "weight_tolerance": "passed",
    }
    assert result["metrics"]["max_abs_weight_difference"] == 0.0
    assert result["differences"] == []


def test_build_replication_validation_reports_missing_extra_and_drift() -> None:
    calculated = [
        {"security_code": "A000001", "target_weight": 0.25},
        {"security_code": "A000002", "target_weight": 0.25},
    ]
    official = [
        {"security_code": "A000001", "official_weight": 0.20},
        {"security_code": "A000003", "official_weight": 0.30},
    ]

    result = build_replication_validation(
        index_code="FI00.WLT.KSS",
        as_of="2026-05-29",
        validation_source_type="official_target_weights",
        calculated_target_weights=calculated,
        validation_weights=official,
        weight_tolerance=0.01,
    )

    assert result["status"] == "failed"
    assert result["checks"] == {
        "constituent_membership": "failed",
        "weight_tolerance": "failed",
    }
    assert result["metrics"]["max_abs_weight_difference"] == 0.05
    assert result["metrics"]["total_abs_weight_difference"] == 0.05
    assert result["differences"] == [
        {
            "type": "missing_in_validation",
            "security_code": "A000002",
            "target_weight": 0.25,
        },
        {
            "type": "extra_in_validation",
            "security_code": "A000003",
            "validation_weight": 0.3,
        },
        {
            "type": "weight_difference",
            "security_code": "A000001",
            "target_weight": 0.25,
            "validation_weight": 0.2,
            "difference": 0.05,
        },
    ]


def test_build_replication_validation_marks_missing_validation_source() -> None:
    result = build_replication_validation(
        index_code="FI00.WLT.KSS",
        as_of="2026-05-29",
        validation_source_type="missing",
        calculated_target_weights=[{"security_code": "A000001", "target_weight": 1.0}],
        validation_weights=[],
        weight_tolerance=0.0,
    )

    assert result["status"] == "not_proven"
    assert result["checks"] == {"validation_source": "missing"}
    assert result["differences"] == [
        {
            "type": "validation_source_missing",
            "index_code": "FI00.WLT.KSS",
            "as_of": "2026-05-29",
        }
    ]


def test_build_replication_validation_rejects_duplicate_security_codes() -> None:
    with pytest.raises(ValueError, match="duplicate security_code"):
        build_replication_validation(
            index_code="FI00.WLT.KSS",
            as_of="2026-05-29",
            validation_source_type="official_target_weights",
            calculated_target_weights=[
                {"security_code": "A000001", "target_weight": 0.5},
                {"security_code": "A000001", "target_weight": 0.5},
            ],
            validation_weights=[
                {"security_code": "A000001", "official_weight": 1.0},
            ],
            weight_tolerance=0.0,
        )

    with pytest.raises(ValueError, match="duplicate security_code"):
        build_replication_validation(
            index_code="FI00.WLT.KSS",
            as_of="2026-05-29",
            validation_source_type="official_target_weights",
            calculated_target_weights=[
                {"security_code": "A000001", "target_weight": 1.0},
            ],
            validation_weights=[
                {"security_code": "A000001", "official_weight": 0.5},
                {"security_code": "A000001", "official_weight": 0.5},
            ],
            weight_tolerance=0.0,
        )


def test_build_replication_validation_rejects_missing_or_blank_weights() -> None:
    with pytest.raises(ValueError, match="target_weight is required"):
        build_replication_validation(
            index_code="FI00.WLT.KSS",
            as_of="2026-05-29",
            validation_source_type="official_target_weights",
            calculated_target_weights=[
                {"security_code": "A000001"},
            ],
            validation_weights=[
                {"security_code": "A000001", "official_weight": 1.0},
            ],
            weight_tolerance=0.0,
        )

    with pytest.raises(ValueError, match="official_weight is required"):
        build_replication_validation(
            index_code="FI00.WLT.KSS",
            as_of="2026-05-29",
            validation_source_type="official_target_weights",
            calculated_target_weights=[
                {"security_code": "A000001", "target_weight": 1.0},
            ],
            validation_weights=[
                {"security_code": "A000001", "official_weight": "   "},
            ],
            weight_tolerance=0.0,
        )


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_build_replication_validation_rejects_non_finite_weights(
    weight: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        build_replication_validation(
            index_code="FI00.WLT.KSS",
            as_of="2026-05-29",
            validation_source_type="official_target_weights",
            calculated_target_weights=[
                {"security_code": "A000001", "target_weight": weight},
            ],
            validation_weights=[
                {"security_code": "A000001", "official_weight": 1.0},
            ],
            weight_tolerance=0.0,
        )


def test_build_replication_validation_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="weight_tolerance must be >= 0"):
        build_replication_validation(
            index_code="FI00.WLT.KSS",
            as_of="2026-05-29",
            validation_source_type="official_target_weights",
            calculated_target_weights=[
                {"security_code": "A000001", "target_weight": 1.0},
            ],
            validation_weights=[
                {"security_code": "A000001", "official_weight": 1.0},
            ],
            weight_tolerance=-0.01,
        )


def test_write_replication_validation_writes_stable_json_and_markdown(
    tmp_path,
) -> None:
    report = build_replication_validation(
        index_code="FI00.WLT.KSS",
        as_of="2026-05-29",
        validation_source_type="official_target_weights",
        calculated_target_weights=[
            {"security_code": "A000001", "target_weight": 0.25},
            {"security_code": "A000002", "target_weight": 0.25},
        ],
        validation_weights=[
            {"security_code": "A000001", "official_weight": 0.2},
            {"security_code": "A000003", "official_weight": 0.3},
        ],
        weight_tolerance=0.01,
    )

    json_path, md_path = write_replication_validation(report, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["result"] == report

    assert md_path.read_text(encoding="utf-8") == (
        "# KSS Replication Validation\n"
        "\n"
        "- index_code: FI00.WLT.KSS\n"
        "- as_of: 2026-05-29\n"
        "- validation_source_type: official_target_weights\n"
        "- status: failed\n"
        "\n"
        "## Differences\n"
        "\n"
        "| type | security_code | target_weight | validation_weight | difference | index_code | as_of |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| missing_in_validation | A000002 | 0.25 |  |  |  |  |\n"
        "| extra_in_validation | A000003 |  | 0.3 |  |  |  |\n"
        "| weight_difference | A000001 | 0.25 | 0.2 | 0.05 |  |  |\n"
    )
