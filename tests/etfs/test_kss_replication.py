from etfs.fnguide.replication import build_replication_validation


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
