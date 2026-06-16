from pathlib import Path

import json

import pytest

from etfs.fnguide.methodology_engine import MethodologyNotReadyError
from etfs.fnguide.replication import (
    build_kss_replication,
    build_replication_validation,
    write_kss_replication_artifacts,
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


@pytest.mark.parametrize(
    "weight_tolerance",
    [float("nan"), float("inf"), float("-inf")],
)
def test_build_replication_validation_rejects_non_finite_tolerance(
    weight_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="weight_tolerance must be finite"):
        build_replication_validation(
            index_code="FI00.WLT.KSS",
            as_of="2026-05-29",
            validation_source_type="official_target_weights",
            calculated_target_weights=[
                {"security_code": "A000001", "target_weight": 1.0},
            ],
            validation_weights=[
                {"security_code": "A000001", "official_weight": 0.9},
            ],
            weight_tolerance=weight_tolerance,
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


def test_write_replication_validation_rejects_nan_report_payload(tmp_path) -> None:
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        write_replication_validation(
            {
                "index_code": "FI00.WLT.KSS",
                "as_of": "2026-05-29",
                "validation_source_type": "official_target_weights",
                "status": "failed",
                "checks": {"weight_tolerance": "failed"},
                "metrics": {"max_abs_weight_difference": float("nan")},
                "differences": [],
            },
            tmp_path,
        )


def _snapshot_row(code: str, float_cap: float, momentum: float) -> dict[str, object]:
    return {
        "as_of": "2026-05-29",
        "security_code": code,
        "name": code,
        "is_eligible": True,
        "is_semiconductor_theme": True,
        "float_market_cap": float_cap,
        "composite_momentum_score": momentum,
    }


def _kss_rows() -> list[dict[str, object]]:
    return [
        _snapshot_row("A000001", 1000, 1),
        _snapshot_row("A000002", 900, 2),
        _snapshot_row("A000003", 800, 90),
        _snapshot_row("A000004", 700, 80),
        _snapshot_row("A000005", 600, 70),
        _snapshot_row("A000006", 500, 60),
        _snapshot_row("A000007", 400, 10),
        _snapshot_row("A000008", 300, 20),
        _snapshot_row("A000009", 200, 30),
        _snapshot_row("A000010", 100, 40),
    ]


def test_build_kss_replication_selects_buckets_and_calculates_target_weights() -> None:
    result = build_kss_replication(
        as_of="2026-05-29",
        effective_date="2026-06-14",
        snapshot_rows=_kss_rows(),
        validation_weights=[],
        validation_source_type="missing",
    )

    assert result["index_code"] == "FI00.WLT.KSS"
    assert result["selected_buckets"] == {
        "top2": [
            {
                "bucket": "top2",
                "rank_metric": "float_market_cap",
                "security_code": "A000001",
                "name": "A000001",
                "float_market_cap": 1000.0,
                "composite_momentum_score": 1.0,
            },
            {
                "bucket": "top2",
                "rank_metric": "float_market_cap",
                "security_code": "A000002",
                "name": "A000002",
                "float_market_cap": 900.0,
                "composite_momentum_score": 2.0,
            },
        ],
        "momentum": [
            {
                "bucket": "momentum",
                "rank_metric": "composite_momentum_score",
                "security_code": "A000003",
                "name": "A000003",
                "float_market_cap": 800.0,
                "composite_momentum_score": 90.0,
            },
            {
                "bucket": "momentum",
                "rank_metric": "composite_momentum_score",
                "security_code": "A000004",
                "name": "A000004",
                "float_market_cap": 700.0,
                "composite_momentum_score": 80.0,
            },
            {
                "bucket": "momentum",
                "rank_metric": "composite_momentum_score",
                "security_code": "A000005",
                "name": "A000005",
                "float_market_cap": 600.0,
                "composite_momentum_score": 70.0,
            },
            {
                "bucket": "momentum",
                "rank_metric": "composite_momentum_score",
                "security_code": "A000006",
                "name": "A000006",
                "float_market_cap": 500.0,
                "composite_momentum_score": 60.0,
            },
        ],
        "market_cap_fill": [
            {
                "bucket": "market_cap_fill",
                "rank_metric": "float_market_cap",
                "security_code": "A000007",
                "name": "A000007",
                "float_market_cap": 400.0,
                "composite_momentum_score": 10.0,
            },
            {
                "bucket": "market_cap_fill",
                "rank_metric": "float_market_cap",
                "security_code": "A000008",
                "name": "A000008",
                "float_market_cap": 300.0,
                "composite_momentum_score": 20.0,
            },
            {
                "bucket": "market_cap_fill",
                "rank_metric": "float_market_cap",
                "security_code": "A000009",
                "name": "A000009",
                "float_market_cap": 200.0,
                "composite_momentum_score": 30.0,
            },
            {
                "bucket": "market_cap_fill",
                "rank_metric": "float_market_cap",
                "security_code": "A000010",
                "name": "A000010",
                "float_market_cap": 100.0,
                "composite_momentum_score": 40.0,
            },
        ],
    }
    assert result["target_weight_result"]["checks"] == {
        "constituent_count": "passed",
        "weight_sum": "passed",
    }
    weights = {
        item["security_code"]: item["target_weight"]
        for item in result["target_weight_result"]["target_weights"]
    }
    selected_security_codes = {
        item["security_code"]
        for bucket in result["selected_buckets"].values()
        for item in bucket
    }
    assert set(weights) == selected_security_codes
    assert weights["A000001"] == 0.25
    assert weights["A000002"] == 0.25
    assert result["validation"]["status"] == "not_proven"
    assert result["full_replication_status"] == "not_proven"


def test_build_kss_replication_raises_controlled_error_when_kss_spec_is_missing(
    tmp_path: Path,
) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps({"indices": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        MethodologyNotReadyError,
        match="FI00.WLT.KSS is missing or not engine-ready",
    ):
        build_kss_replication(
            as_of="2026-05-29",
            effective_date="2026-06-14",
            snapshot_rows=_kss_rows(),
            validation_weights=[],
            validation_source_type="missing",
            specs_path=specs_path,
        )

    with pytest.raises(
        MethodologyNotReadyError,
        match=str(specs_path).replace("\\", "\\\\"),
    ):
        build_kss_replication(
            as_of="2026-05-29",
            effective_date="2026-06-14",
            snapshot_rows=_kss_rows(),
            validation_weights=[],
            validation_source_type="missing",
            specs_path=specs_path,
        )


def test_build_kss_replication_raises_controlled_error_for_missing_specs_path(
    tmp_path: Path,
) -> None:
    specs_path = tmp_path / "missing_specs.json"

    with pytest.raises(
        MethodologyNotReadyError,
        match="FI00.WLT.KSS",
    ):
        build_kss_replication(
            as_of="2026-05-29",
            effective_date="2026-06-14",
            snapshot_rows=_kss_rows(),
            validation_weights=[],
            validation_source_type="missing",
            specs_path=specs_path,
        )

    with pytest.raises(
        MethodologyNotReadyError,
        match=str(specs_path).replace("\\", "\\\\"),
    ):
        build_kss_replication(
            as_of="2026-05-29",
            effective_date="2026-06-14",
            snapshot_rows=_kss_rows(),
            validation_weights=[],
            validation_source_type="missing",
            specs_path=specs_path,
        )


@pytest.mark.parametrize("contents", ["", "{"])
def test_build_kss_replication_raises_controlled_error_for_invalid_specs_file(
    tmp_path: Path,
    contents: str,
) -> None:
    specs_path = tmp_path / "invalid_specs.json"
    specs_path.write_text(contents, encoding="utf-8")

    with pytest.raises(
        MethodologyNotReadyError,
        match="FI00.WLT.KSS",
    ):
        build_kss_replication(
            as_of="2026-05-29",
            effective_date="2026-06-14",
            snapshot_rows=_kss_rows(),
            validation_weights=[],
            validation_source_type="missing",
            specs_path=specs_path,
        )

    with pytest.raises(
        MethodologyNotReadyError,
        match=str(specs_path).replace("\\", "\\\\"),
    ):
        build_kss_replication(
            as_of="2026-05-29",
            effective_date="2026-06-14",
            snapshot_rows=_kss_rows(),
            validation_weights=[],
            validation_source_type="missing",
            specs_path=specs_path,
        )


def test_build_kss_replication_etf_holdings_validation_does_not_mark_proven() -> None:
    base_result = build_kss_replication(
        as_of="2026-05-29",
        effective_date="2026-06-14",
        snapshot_rows=_kss_rows(),
        validation_weights=[],
        validation_source_type="missing",
    )
    validation_weights = [
        {
            "security_code": item["security_code"],
            "holding_weight": item["target_weight"],
        }
        for item in base_result["target_weight_result"]["target_weights"]
    ]

    result = build_kss_replication(
        as_of="2026-05-29",
        effective_date="2026-06-14",
        snapshot_rows=_kss_rows(),
        validation_weights=validation_weights,
        validation_source_type="etf_holdings_snapshot",
    )

    assert result["validation"]["status"] == "passed"
    assert result["full_replication_status"] == "not_proven"


def test_build_kss_replication_official_validation_marks_proven() -> None:
    base_result = build_kss_replication(
        as_of="2026-05-29",
        effective_date="2026-06-14",
        snapshot_rows=_kss_rows(),
        validation_weights=[],
        validation_source_type="missing",
    )
    validation_weights = [
        {
            "security_code": item["security_code"],
            "official_weight": item["target_weight"],
        }
        for item in base_result["target_weight_result"]["target_weights"]
    ]

    result = build_kss_replication(
        as_of="2026-05-29",
        effective_date="2026-06-14",
        snapshot_rows=_kss_rows(),
        validation_weights=validation_weights,
        validation_source_type="official_target_weights",
    )

    assert result["validation"]["status"] == "passed"
    assert result["full_replication_status"] == "proven"


def test_write_kss_replication_artifacts_outputs_selected_weights_and_validation(
    tmp_path: Path,
) -> None:
    result = build_kss_replication(
        as_of="2026-05-29",
        effective_date="2026-06-14",
        snapshot_rows=_kss_rows(),
        validation_weights=[],
        validation_source_type="missing",
    )

    outputs = write_kss_replication_artifacts(result, tmp_path)

    assert outputs == {
        "selected_buckets": (tmp_path / "kss_selected_buckets.json").as_posix(),
        "target_weights": (tmp_path / "kss_target_weights.json").as_posix(),
        "replication_validation": (tmp_path / "kss_replication_validation.json").as_posix(),
        "replication_validation_md": (tmp_path / "kss_replication_validation.md").as_posix(),
    }

    selected_payload = json.loads((tmp_path / "kss_selected_buckets.json").read_text(encoding="utf-8"))
    target_payload = json.loads((tmp_path / "kss_target_weights.json").read_text(encoding="utf-8"))
    validation_payload = json.loads(
        (tmp_path / "kss_replication_validation.json").read_text(encoding="utf-8")
    )

    assert selected_payload["effective_date"] == "2026-06-14"
    assert selected_payload["selected_buckets"] == result["selected_buckets"]
    assert target_payload["result"] == result["target_weight_result"]
    assert target_payload["result"]["effective_date"] == "2026-06-14"
    assert validation_payload["result"]["status"] == "not_proven"
