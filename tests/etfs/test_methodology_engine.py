import json
from pathlib import Path

import pytest

import etfs.fnguide.methodology_engine as methodology_engine
from etfs.fnguide.methodology_engine import (
    MethodologyNotReadyError,
    build_engine_promotion_candidates,
    build_engine_support_matrix,
    build_methodology_replication_report,
    build_parser,
    calculate_capped_float_market_cap_target_weights,
    calculate_capped_metric_target_weights,
    calculate_equal_weight_target_weights,
    calculate_fixed_plus_residual_target_weights,
    calculate_top2_plus_target_weights,
    load_engine_ready_specs,
    require_engine_ready_spec,
    write_engine_input_requirements,
    write_engine_input_template,
    write_engine_promotion_candidates,
    write_engine_support_matrix,
    write_methodology_replication_report,
    write_target_weights,
)


def _top2_plus_spec() -> dict[str, object]:
    return {
        "index_code": "FI00.TOP2.PLUS",
        "index_name": "Generic Top2 Plus Index",
        "status": "methodology_verified",
        "selection": {
            "total_constituents": 10,
            "buckets": [
                {"name": "top2", "count": 2, "weight": {"type": "fixed", "value": 0.25}},
                {"name": "momentum", "count": 4},
                {"name": "market_cap_fill", "count": 4},
            ],
        },
        "weighting": {
            "base": "float_market_cap_weighted",
            "residual": {
                "applies_to_buckets": ["momentum", "market_cap_fill"],
                "total_weight": 0.5,
                "base": "float_market_cap",
                "cap": 0.15,
                "redistribution": "iterative_pro_rata",
            },
        },
        "open_questions": [],
    }


def _top2_constituents() -> dict[str, list[dict[str, object]]]:
    return {
        "top2": [
            {"security_code": "A000001", "float_market_cap": 1000},
            {"security_code": "A000002", "float_market_cap": 900},
        ],
        "momentum": [
            {"security_code": "A000003", "float_market_cap": 80},
            {"security_code": "A000004", "float_market_cap": 70},
            {"security_code": "A000005", "float_market_cap": 60},
            {"security_code": "A000006", "float_market_cap": 50},
        ],
        "market_cap_fill": [
            {"security_code": "A000007", "float_market_cap": 40},
            {"security_code": "A000008", "float_market_cap": 30},
            {"security_code": "A000009", "float_market_cap": 20},
            {"security_code": "A000010", "float_market_cap": 10},
        ],
    }


def _equal_weight_spec() -> dict[str, object]:
    return {
        "index_code": "FI00.EQUAL",
        "index_name": "Equal Weight Index",
        "status": "methodology_verified",
        "selection": {"total_constituents": 3, "buckets": []},
        "weighting": {"base": "equal_weighted", "residual": {}},
        "open_questions": [],
    }


def _float_cap_spec() -> dict[str, object]:
    return {
        "index_code": "FI00.FLOAT",
        "index_name": "Float Cap Index",
        "status": "methodology_verified",
        "selection": {"total_constituents": 3, "buckets": []},
        "weighting": {"base": "float_market_cap_weighted", "residual": {}, "security_cap": 0.5},
        "open_questions": [],
    }


def _metric_cap_spec() -> dict[str, object]:
    return {
        "index_code": "FI00.METRIC",
        "index_name": "Metric Cap Index",
        "status": "methodology_verified",
        "selection": {"total_constituents": 3, "buckets": []},
        "weighting": {
            "base": "metric_weighted",
            "metric": "dividend_amount",
            "residual": {},
            "security_cap": 0.5,
        },
        "open_questions": [],
    }


def _fixed_residual_equal_spec() -> dict[str, object]:
    return {
        "index_code": "FI00.FIXED.EQUAL",
        "index_name": "Fixed Plus Equal Residual Index",
        "status": "methodology_verified",
        "selection": {
            "total_constituents": 20,
            "buckets": [
                {"name": "top3", "count": 3, "weight": {"type": "fixed", "value": 0.25}},
                {"name": "residual", "count": 17},
            ],
        },
        "weighting": {
            "base": "fixed_plus_residual",
            "residual": {"applies_to_buckets": ["residual"], "total_weight": 0.25, "base": "equal_weighted"},
        },
        "open_questions": [],
    }


def _fixed_residual_float_cap_spec() -> dict[str, object]:
    return {
        "index_code": "FI00.FIXED.FLOAT",
        "index_name": "Fixed Plus Float Residual Index",
        "status": "methodology_verified",
        "selection": {
            "total_constituents": 10,
            "buckets": [
                {"name": "top3", "count": 3, "weight": {"type": "fixed", "value": 0.2}},
                {"name": "plus", "count": 7},
            ],
        },
        "weighting": {
            "base": "fixed_plus_residual",
            "residual": {
                "applies_to_buckets": ["plus"],
                "total_weight": 0.4,
                "base": "float_market_cap",
                "cap": 0.07,
                "redistribution": "iterative_pro_rata",
            },
        },
        "open_questions": [],
    }


def test_require_engine_ready_spec_rejects_draft_specs() -> None:
    draft = {
        "index_code": "FI00.DRAFT",
        "status": "draft_extracted",
        "selection": {"total_constituents": 10},
        "open_questions": [],
    }

    with pytest.raises(MethodologyNotReadyError, match="status=draft_extracted"):
        require_engine_ready_spec(draft)


def test_load_engine_ready_specs_returns_only_verified_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps({"indices": [_top2_plus_spec(), {"index_code": "FI00.DRAFT", "status": "draft_extracted"}]}),
        encoding="utf-8",
    )

    specs = load_engine_ready_specs(specs_path)

    assert list(specs) == ["FI00.TOP2.PLUS"]
    assert specs["FI00.TOP2.PLUS"]["index_name"] == "Generic Top2 Plus Index"


def test_calculate_top2_plus_target_weights_applies_fixed_top2_and_residual_pro_rata() -> None:
    weights = calculate_top2_plus_target_weights(_top2_plus_spec(), _top2_constituents())

    assert weights["A000001"] == pytest.approx(0.25)
    assert weights["A000002"] == pytest.approx(0.25)
    assert weights["A000003"] == pytest.approx(0.5 * 80 / 360)
    assert weights["A000010"] == pytest.approx(0.5 * 10 / 360)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_top2_plus_target_weights_caps_residual_and_redistributes_excess() -> None:
    constituents = _top2_constituents()
    constituents["momentum"][0]["float_market_cap"] = 900

    weights = calculate_top2_plus_target_weights(_top2_plus_spec(), constituents)

    assert weights["A000003"] == pytest.approx(0.15)
    assert weights["A000004"] == pytest.approx(0.35 * 70 / 280)
    assert weights["A000010"] == pytest.approx(0.35 * 10 / 280)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_top2_plus_target_weights_rejects_invalid_buckets() -> None:
    constituents = _top2_constituents()
    constituents["momentum"] = constituents["momentum"][:3]

    with pytest.raises(ValueError, match="momentum requires 4 constituents"):
        calculate_top2_plus_target_weights(_top2_plus_spec(), constituents)

    constituents = _top2_constituents()
    constituents["market_cap_fill"][0]["security_code"] = "A000003"
    with pytest.raises(ValueError, match="duplicate security_code"):
        calculate_top2_plus_target_weights(_top2_plus_spec(), constituents)


def test_weight_engines_cover_equal_float_metric_and_fixed_residual_methods() -> None:
    assert calculate_equal_weight_target_weights(
        _equal_weight_spec(),
        [{"security_code": "A000001"}, {"security_code": "A000002"}, {"security_code": "A000003"}],
    ) == {"A000001": pytest.approx(1 / 3), "A000002": pytest.approx(1 / 3), "A000003": pytest.approx(1 / 3)}

    float_weights = calculate_capped_float_market_cap_target_weights(
        _float_cap_spec(),
        [
            {"security_code": "A000001", "float_market_cap": 900},
            {"security_code": "A000002", "float_market_cap": 70},
            {"security_code": "A000003", "float_market_cap": 30},
        ],
    )
    assert float_weights["A000001"] == pytest.approx(0.5)
    assert sum(float_weights.values()) == pytest.approx(1.0)

    metric_weights = calculate_capped_metric_target_weights(
        _metric_cap_spec(),
        [
            {"security_code": "A000001", "dividend_amount": 900},
            {"security_code": "A000002", "dividend_amount": 70},
            {"security_code": "A000003", "dividend_amount": 30},
        ],
    )
    assert metric_weights["A000001"] == pytest.approx(0.5)
    assert sum(metric_weights.values()) == pytest.approx(1.0)

    fixed_equal = calculate_fixed_plus_residual_target_weights(
        _fixed_residual_equal_spec(),
        {
            "top3": [{"security_code": "A000001"}, {"security_code": "A000002"}, {"security_code": "A000003"}],
            "residual": [{"security_code": f"A1{i:05d}"} for i in range(17)],
        },
    )
    assert fixed_equal["A000001"] == pytest.approx(0.25)
    assert fixed_equal["A100000"] == pytest.approx(0.25 / 17)

    fixed_float = calculate_fixed_plus_residual_target_weights(
        _fixed_residual_float_cap_spec(),
        {
            "top3": [{"security_code": "A000001"}, {"security_code": "A000002"}, {"security_code": "A000003"}],
            "plus": [
                {"security_code": "A000004", "float_market_cap": 900},
                {"security_code": "A000005", "float_market_cap": 60},
                {"security_code": "A000006", "float_market_cap": 50},
                {"security_code": "A000007", "float_market_cap": 40},
                {"security_code": "A000008", "float_market_cap": 30},
                {"security_code": "A000009", "float_market_cap": 20},
                {"security_code": "A000010", "float_market_cap": 10},
            ],
        },
    )
    assert fixed_float["A000004"] == pytest.approx(0.07)
    assert sum(fixed_float.values()) == pytest.approx(1.0)


def test_write_target_weights_outputs_file_from_explicit_engine_inputs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_top2_plus_spec()]}), encoding="utf-8")
    inputs_path = tmp_path / "engine_inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "index_code": "FI00.TOP2.PLUS",
                        "as_of": "2026-05-29",
                        "methodology": "top2_plus",
                        "constituents_by_bucket": _top2_constituents(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = write_target_weights(inputs_path, specs_path, tmp_path)

    result = json.loads(output_path.read_text(encoding="utf-8"))["results"][0]
    assert output_path.name == "target_weights.json"
    assert result["index_code"] == "FI00.TOP2.PLUS"
    assert result["checks"] == {"constituent_count": "passed", "weight_sum": "passed"}
    assert result["target_weights"][0] == {"security_code": "A000001", "target_weight": 0.25}


def test_write_target_weights_supports_non_bucket_methods(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_equal_weight_spec(), _metric_cap_spec()]}), encoding="utf-8")
    inputs_path = tmp_path / "engine_inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "index_code": "FI00.EQUAL",
                        "as_of": "2026-05-29",
                        "methodology": "equal_weighted",
                        "constituents": [{"security_code": "A000001"}, {"security_code": "A000002"}, {"security_code": "A000003"}],
                    },
                    {
                        "index_code": "FI00.METRIC",
                        "as_of": "2026-05-29",
                        "methodology": "metric_weighted",
                        "constituents": [
                            {"security_code": "A000001", "dividend_amount": 900},
                            {"security_code": "A000002", "dividend_amount": 70},
                            {"security_code": "A000003", "dividend_amount": 30},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(write_target_weights(inputs_path, specs_path, tmp_path).read_text(encoding="utf-8"))

    assert [item["methodology"] for item in payload["results"]] == ["equal_weighted", "metric_weighted"]
    assert [item["metrics"] for item in payload["results"]] == [
        {"constituent_count": 3, "weight_sum": 1.0},
        {"constituent_count": 3, "weight_sum": 1.0},
    ]


def test_write_engine_input_requirements_describes_generic_verified_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps({"indices": [_top2_plus_spec(), _equal_weight_spec(), _float_cap_spec(), _metric_cap_spec(), _fixed_residual_equal_spec()]}),
        encoding="utf-8",
    )

    payload = json.loads(write_engine_input_requirements(specs_path, tmp_path).read_text(encoding="utf-8"))
    requirements = {item["index_code"]: item for item in payload["requirements"]}

    assert requirements["FI00.TOP2.PLUS"]["methodology"] == "top2_plus"
    assert requirements["FI00.TOP2.PLUS"]["required_buckets"] == [
        {"name": "top2", "count": 2, "weighting": "fixed"},
        {"name": "momentum", "count": 4, "weighting": "residual_float_market_cap"},
        {"name": "market_cap_fill", "count": 4, "weighting": "residual_float_market_cap"},
    ]
    assert requirements["FI00.EQUAL"]["required_fields"] == ["security_code"]
    assert requirements["FI00.FLOAT"]["weighting"] == {"security_cap": 0.5, "redistribution": "iterative_pro_rata"}
    assert requirements["FI00.METRIC"]["required_fields"] == ["security_code", "dividend_amount"]
    assert requirements["FI00.FIXED.EQUAL"]["required_buckets"] == [
        {"name": "top3", "count": 3, "weighting": "fixed"},
        {"name": "residual", "count": 17, "weighting": "residual_equal_weighted"},
    ]


def test_engine_support_and_promotion_reports_are_generic(tmp_path: Path) -> None:
    draft_supported = {**_equal_weight_spec(), "index_code": "FI00.DRAFT.EQUAL", "status": "draft_extracted"}
    missing_count = {
        **_equal_weight_spec(),
        "index_code": "FI00.MISSING.COUNT",
        "status": "draft_extracted",
        "selection": {"total_constituents": None, "buckets": []},
        "open_questions": ["selection.total_constituents not extracted with evidence"],
    }
    unsupported = {**_equal_weight_spec(), "index_code": "FI00.CUSTOM", "weighting": {"base": "custom_weighted", "residual": {}}}

    matrix = build_engine_support_matrix([_top2_plus_spec(), draft_supported, missing_count, unsupported])

    assert matrix["counts"] == {
        "total": 4,
        "engine_ready": 1,
        "supported_after_review": 1,
        "blocked_by_methodology_evidence": 1,
        "unsupported_methodology": 1,
    }
    items = {item["index_code"]: item for item in matrix["items"]}
    assert items["FI00.TOP2.PLUS"]["engine_support_status"] == "engine_ready"
    assert items["FI00.DRAFT.EQUAL"]["engine_support_status"] == "supported_after_review"

    candidates = build_engine_promotion_candidates([_top2_plus_spec(), draft_supported, missing_count])
    assert candidates["counts"] == {"total": 1, "by_methodology": {"equal_weighted": 1}}
    assert candidates["items"][0]["index_code"] == "FI00.DRAFT.EQUAL"

    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_top2_plus_spec(), _equal_weight_spec()]}), encoding="utf-8")
    support_json, support_md = write_engine_support_matrix(specs_path, tmp_path)
    promotion_json, promotion_md = write_engine_promotion_candidates(specs_path, tmp_path)
    assert support_json.name == "engine_support_matrix.json"
    assert promotion_json.name == "engine_promotion_candidates.json"
    assert "FnGuide Engine Support Matrix" in support_md.read_text(encoding="utf-8")
    assert "FnGuide Engine Promotion Candidates" in promotion_md.read_text(encoding="utf-8")


def test_write_engine_input_template_outputs_fillable_requests_for_ready_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_top2_plus_spec(), _equal_weight_spec()]}), encoding="utf-8")

    payload = json.loads(write_engine_input_template(specs_path, tmp_path).read_text(encoding="utf-8"))

    assert payload["template_only"] is True
    assert payload["count"] == 2
    top2_request = payload["requests"][0]
    assert top2_request["index_code"] == "FI00.TOP2.PLUS"
    assert list(top2_request["constituents_by_bucket"]) == ["top2", "momentum", "market_cap_fill"]
    assert top2_request["constituents_by_bucket"]["top2"][0] == {"security_code": "", "float_market_cap": None}
    assert payload["requests"][1]["constituents"] == [{"security_code": ""}, {"security_code": ""}, {"security_code": ""}]


def test_build_methodology_replication_report_smoke_tests_engine_ready_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_top2_plus_spec(), _equal_weight_spec()]}), encoding="utf-8")

    report = build_methodology_replication_report(specs_path)

    assert report["counts"]["total_specs"] == 2
    assert report["counts"]["engine_ready"] == 2
    assert report["counts"]["target_weight_replication_passed"] == 2
    assert report["counts"]["target_weight_replication_failed"] == 0
    assert report["counts"]["full_methodology_replication_proven"] == 0
    ready_items = [item for item in report["items"] if item["engine_support_status"] == "engine_ready"]
    assert {item["target_weight_replication_status"] for item in ready_items} == {"passed"}
    assert {item["full_methodology_replication_status"] for item in ready_items} == {"not_proven"}


def test_build_methodology_replication_report_never_promotes_full_replication_from_named_artifacts(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_top2_plus_spec()]}), encoding="utf-8")

    report = build_methodology_replication_report(specs_path)

    item = report["items"][0]
    assert item["target_weight_replication_status"] == "passed"
    assert item["full_methodology_replication_status"] == "not_proven"
    assert item["full_methodology_replication_evidence"] == ""
    assert item["full_methodology_replication_blockers"] == [
        "constituent universe and bucket selection are supplied as explicit engine inputs",
        "official rebalance target weights are not available for direct comparison",
    ]


def test_build_methodology_replication_report_keeps_full_methodology_not_proven_when_target_weight_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_top2_plus_spec()]}), encoding="utf-8")
    monkeypatch.setattr(
        methodology_engine,
        "_target_weight_result",
        lambda request, ready_specs: {"checks": {"weight_tolerance": "failed"}, "metrics": {}},
    )

    report = build_methodology_replication_report(specs_path)

    item = report["items"][0]
    assert item["target_weight_replication_status"] == "failed"
    assert item["full_methodology_replication_status"] == "not_proven"


def test_write_methodology_replication_report_outputs_json_and_markdown(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_top2_plus_spec(), {**_equal_weight_spec(), "status": "draft_extracted"}]}), encoding="utf-8")

    json_path, md_path = write_methodology_replication_report(specs_path, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.name == "methodology_replication_report.json"
    assert md_path.name == "methodology_replication_report.md"
    assert payload["counts"]["target_weight_replication_passed"] == 1
    assert payload["counts"]["full_methodology_replication_proven"] == 0
    assert "FnGuide Methodology Replication Report" in md_path.read_text(encoding="utf-8")


def test_write_target_weights_rejects_requests_for_non_engine_ready_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [{"index_code": "FI00.DRAFT", "status": "draft_extracted", "selection": {"total_constituents": 10}}]}), encoding="utf-8")
    inputs_path = tmp_path / "engine_inputs.json"
    inputs_path.write_text(json.dumps({"requests": [{"index_code": "FI00.DRAFT", "as_of": "2026-05-29", "methodology": "top2_plus"}]}), encoding="utf-8")

    with pytest.raises(MethodologyNotReadyError, match="FI00.DRAFT"):
        write_target_weights(inputs_path, specs_path, tmp_path)


def test_methodology_engine_parser_uses_grouped_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.inputs == "etfs/output/methodology/fnguide/engine_inputs.json"
    assert args.specs == "etfs/output/methodology/fnguide/methodology_specs.json"
    assert args.output_dir == "etfs/output/methodology/fnguide"
    assert args.write_requirements is False
    assert args.write_template is False
    assert args.write_replication_report is False


def test_methodology_engine_parser_accepts_template_and_replication_report_modes() -> None:
    assert build_parser().parse_args(["--write-template"]).write_template is True
    assert build_parser().parse_args(["--write-replication-report"]).write_replication_report is True
