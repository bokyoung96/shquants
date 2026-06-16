import json
from pathlib import Path

import pytest

import etfs.fnguide.methodology_engine as methodology_engine
from etfs.fnguide.methodology_engine import (
    MethodologyNotReadyError,
    build_methodology_replication_report,
    build_engine_support_matrix,
    build_engine_promotion_candidates,
    build_parser,
    calculate_fixed_plus_residual_target_weights,
    calculate_capped_float_market_cap_target_weights,
    calculate_capped_metric_target_weights,
    calculate_equal_weight_target_weights,
    calculate_top2_plus_target_weights,
    write_engine_input_requirements,
    write_engine_input_template,
    write_engine_promotion_candidates,
    write_methodology_replication_report,
    load_engine_ready_specs,
    require_engine_ready_spec,
    write_target_weights,
    write_engine_support_matrix,
)


def _kss_spec() -> dict[str, object]:
    return {
        "index_code": "FI00.WLT.KSS",
        "index_name": "FnGuide AI 반도체 TOP2 Plus 지수",
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


def _constituents() -> dict[str, list[dict[str, object]]]:
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
        "index_code": "FI00.FIXED_EQUAL",
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
        "index_code": "FI00.FIXED_FLOAT",
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
        json.dumps(
            {
                "indices": [
                    _kss_spec(),
                    {
                        "index_code": "FI00.DRAFT",
                        "status": "draft_extracted",
                        "selection": {"total_constituents": 10},
                        "open_questions": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_engine_ready_specs(specs_path)

    assert list(specs) == ["FI00.WLT.KSS"]
    assert specs["FI00.WLT.KSS"]["index_name"] == "FnGuide AI 반도체 TOP2 Plus 지수"


def test_calculate_top2_plus_target_weights_applies_fixed_top2_and_residual_pro_rata() -> None:
    weights = calculate_top2_plus_target_weights(_kss_spec(), _constituents())

    assert weights["A000001"] == pytest.approx(0.25)
    assert weights["A000002"] == pytest.approx(0.25)
    assert weights["A000003"] == pytest.approx(0.5 * 80 / 360)
    assert weights["A000010"] == pytest.approx(0.5 * 10 / 360)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_top2_plus_target_weights_caps_residual_and_redistributes_excess() -> None:
    constituents = _constituents()
    constituents["momentum"][0]["float_market_cap"] = 900

    weights = calculate_top2_plus_target_weights(_kss_spec(), constituents)

    assert weights["A000003"] == pytest.approx(0.15)
    assert weights["A000004"] == pytest.approx(0.35 * 70 / 280)
    assert weights["A000010"] == pytest.approx(0.35 * 10 / 280)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_top2_plus_target_weights_rejects_bucket_count_mismatches() -> None:
    constituents = _constituents()
    constituents["momentum"] = constituents["momentum"][:3]

    with pytest.raises(ValueError, match="momentum requires 4 constituents"):
        calculate_top2_plus_target_weights(_kss_spec(), constituents)


def test_calculate_top2_plus_target_weights_rejects_duplicate_securities_across_buckets() -> None:
    constituents = _constituents()
    constituents["market_cap_fill"][0]["security_code"] = "A000003"

    with pytest.raises(ValueError, match="duplicate security_code"):
        calculate_top2_plus_target_weights(_kss_spec(), constituents)


def test_calculate_equal_weight_target_weights_assigns_equal_weights() -> None:
    weights = calculate_equal_weight_target_weights(
        _equal_weight_spec(),
        [
            {"security_code": "A000001"},
            {"security_code": "A000002"},
            {"security_code": "A000003"},
        ],
    )

    assert weights == {"A000001": pytest.approx(1 / 3), "A000002": pytest.approx(1 / 3), "A000003": pytest.approx(1 / 3)}


def test_calculate_equal_weight_target_weights_rejects_count_mismatches() -> None:
    with pytest.raises(ValueError, match="requires 3 constituents"):
        calculate_equal_weight_target_weights(_equal_weight_spec(), [{"security_code": "A000001"}])


def test_calculate_capped_float_market_cap_target_weights_caps_and_redistributes() -> None:
    weights = calculate_capped_float_market_cap_target_weights(
        _float_cap_spec(),
        [
            {"security_code": "A000001", "float_market_cap": 900},
            {"security_code": "A000002", "float_market_cap": 70},
            {"security_code": "A000003", "float_market_cap": 30},
        ],
    )

    assert weights["A000001"] == pytest.approx(0.5)
    assert weights["A000002"] == pytest.approx(0.5 * 70 / 100)
    assert weights["A000003"] == pytest.approx(0.5 * 30 / 100)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_capped_float_market_cap_target_weights_rejects_duplicate_securities() -> None:
    with pytest.raises(ValueError, match="duplicate security_code"):
        calculate_capped_float_market_cap_target_weights(
            _float_cap_spec(),
            [
                {"security_code": "A000001", "float_market_cap": 10},
                {"security_code": "A000001", "float_market_cap": 20},
                {"security_code": "A000003", "float_market_cap": 30},
            ],
        )


def test_calculate_capped_metric_target_weights_caps_and_redistributes() -> None:
    weights = calculate_capped_metric_target_weights(
        _metric_cap_spec(),
        [
            {"security_code": "A000001", "dividend_amount": 900},
            {"security_code": "A000002", "dividend_amount": 70},
            {"security_code": "A000003", "dividend_amount": 30},
        ],
    )

    assert weights["A000001"] == pytest.approx(0.5)
    assert weights["A000002"] == pytest.approx(0.5 * 70 / 100)
    assert weights["A000003"] == pytest.approx(0.5 * 30 / 100)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_fixed_plus_residual_target_weights_supports_equal_residual_bucket() -> None:
    weights = calculate_fixed_plus_residual_target_weights(
        _fixed_residual_equal_spec(),
        {
            "top3": [{"security_code": "A000001"}, {"security_code": "A000002"}, {"security_code": "A000003"}],
            "residual": [{"security_code": f"A1{i:05d}"} for i in range(17)],
        },
    )

    assert weights["A000001"] == pytest.approx(0.25)
    assert weights["A100000"] == pytest.approx(0.25 / 17)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_fixed_plus_residual_target_weights_supports_float_cap_residual_bucket() -> None:
    weights = calculate_fixed_plus_residual_target_weights(
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

    assert weights["A000001"] == pytest.approx(0.2)
    assert weights["A000004"] == pytest.approx(0.07)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_write_target_weights_outputs_file_from_explicit_engine_inputs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_kss_spec()]}), encoding="utf-8")
    inputs_path = tmp_path / "engine_inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "index_code": "FI00.WLT.KSS",
                        "as_of": "2026-05-29",
                        "methodology": "top2_plus",
                        "constituents_by_bucket": _constituents(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = write_target_weights(inputs_path, specs_path, tmp_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    assert output_path.name == "target_weights.json"
    assert result["index_code"] == "FI00.WLT.KSS"
    assert result["as_of"] == "2026-05-29"
    assert result["checks"] == {"constituent_count": "passed", "weight_sum": "passed"}
    assert result["metrics"] == {"constituent_count": 10, "weight_sum": 1.0}
    assert result["target_weights"][0] == {"security_code": "A000001", "target_weight": 0.25}


def test_write_target_weights_supports_equal_weight_methodology(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_equal_weight_spec()]}), encoding="utf-8")
    inputs_path = tmp_path / "engine_inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "index_code": "FI00.EQUAL",
                        "as_of": "2026-05-29",
                        "methodology": "equal_weighted",
                        "constituents": [
                            {"security_code": "A000001"},
                            {"security_code": "A000002"},
                            {"security_code": "A000003"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = write_target_weights(inputs_path, specs_path, tmp_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["methodology"] == "equal_weighted"
    assert payload["results"][0]["metrics"] == {"constituent_count": 3, "weight_sum": 1.0}


def test_write_target_weights_supports_metric_weighted_methodology(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_metric_cap_spec()]}), encoding="utf-8")
    inputs_path = tmp_path / "engine_inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "index_code": "FI00.METRIC",
                        "as_of": "2026-05-29",
                        "methodology": "metric_weighted",
                        "constituents": [
                            {"security_code": "A000001", "dividend_amount": 900},
                            {"security_code": "A000002", "dividend_amount": 70},
                            {"security_code": "A000003", "dividend_amount": 30},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = write_target_weights(inputs_path, specs_path, tmp_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["methodology"] == "metric_weighted"
    assert payload["results"][0]["metrics"] == {"constituent_count": 3, "weight_sum": 1.0}


def test_write_target_weights_supports_fixed_plus_residual_methodology(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_fixed_residual_equal_spec()]}), encoding="utf-8")
    inputs_path = tmp_path / "engine_inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "index_code": "FI00.FIXED_EQUAL",
                        "as_of": "2026-05-29",
                        "methodology": "fixed_plus_residual",
                        "constituents_by_bucket": {
                            "top3": [
                                {"security_code": "A000001"},
                                {"security_code": "A000002"},
                                {"security_code": "A000003"},
                            ],
                            "residual": [{"security_code": f"A1{i:05d}"} for i in range(17)],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = write_target_weights(inputs_path, specs_path, tmp_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["methodology"] == "fixed_plus_residual"
    assert payload["results"][0]["metrics"] == {"constituent_count": 20, "weight_sum": 1.0}


def test_write_engine_input_requirements_describes_generic_verified_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps({"indices": [_equal_weight_spec(), _float_cap_spec(), _metric_cap_spec(), _fixed_residual_equal_spec()]}),
        encoding="utf-8",
    )

    output_path = write_engine_input_requirements(specs_path, tmp_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    requirements = {item["index_code"]: item for item in payload["requirements"]}
    assert requirements["FI00.EQUAL"]["methodology"] == "equal_weighted"
    assert requirements["FI00.EQUAL"]["required_fields"] == ["security_code"]
    assert requirements["FI00.FLOAT"]["methodology"] == "float_market_cap_weighted"
    assert requirements["FI00.FLOAT"]["required_fields"] == ["security_code", "float_market_cap"]
    assert requirements["FI00.FLOAT"]["weighting"] == {"security_cap": 0.5, "redistribution": "iterative_pro_rata"}
    assert requirements["FI00.METRIC"]["methodology"] == "metric_weighted"
    assert requirements["FI00.METRIC"]["required_fields"] == ["security_code", "dividend_amount"]
    assert requirements["FI00.METRIC"]["weighting"] == {
        "metric": "dividend_amount",
        "security_cap": 0.5,
        "redistribution": "iterative_pro_rata",
    }
    assert requirements["FI00.FIXED_EQUAL"]["methodology"] == "fixed_plus_residual"
    assert requirements["FI00.FIXED_EQUAL"]["required_buckets"] == [
        {"name": "top3", "count": 3, "weighting": "fixed"},
        {"name": "residual", "count": 17, "weighting": "residual_equal_weighted"},
    ]


def test_build_engine_support_matrix_separates_readiness_from_methodology_support() -> None:
    draft_supported = {
        **_equal_weight_spec(),
        "index_code": "FI00.DRAFT_EQUAL",
        "status": "draft_extracted",
    }
    missing_count = {
        **_equal_weight_spec(),
        "index_code": "FI00.MISSING_COUNT",
        "status": "draft_extracted",
        "selection": {"total_constituents": None, "buckets": []},
        "open_questions": ["selection.total_constituents not extracted with evidence"],
    }
    unsupported = {
        **_equal_weight_spec(),
        "index_code": "FI00.CUSTOM",
        "weighting": {"base": "custom_weighted", "residual": {}},
    }

    matrix = build_engine_support_matrix([_kss_spec(), draft_supported, missing_count, unsupported])

    assert matrix["counts"] == {
        "total": 4,
        "engine_ready": 1,
        "supported_after_review": 1,
        "blocked_by_methodology_evidence": 1,
        "unsupported_methodology": 1,
    }
    items = {item["index_code"]: item for item in matrix["items"]}
    assert items["FI00.WLT.KSS"]["engine_support_status"] == "engine_ready"
    assert items["FI00.DRAFT_EQUAL"]["engine_support_status"] == "supported_after_review"
    assert items["FI00.DRAFT_EQUAL"]["methodology"] == "equal_weighted"
    assert items["FI00.MISSING_COUNT"]["engine_support_status"] == "blocked_by_methodology_evidence"
    assert items["FI00.CUSTOM"]["engine_support_status"] == "unsupported_methodology"


def test_write_engine_support_matrix_outputs_json_and_markdown(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_kss_spec(), _equal_weight_spec()]}), encoding="utf-8")

    json_path, md_path = write_engine_support_matrix(specs_path, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.name == "engine_support_matrix.json"
    assert md_path.name == "engine_support_matrix.md"
    assert payload["counts"]["engine_ready"] == 2
    assert "FnGuide Engine Support Matrix" in md_path.read_text(encoding="utf-8")


def test_build_engine_promotion_candidates_lists_supported_review_only_specs() -> None:
    draft_supported = {
        **_equal_weight_spec(),
        "index_code": "FI00.DRAFT_EQUAL",
        "status": "draft_extracted",
    }
    missing_count = {
        **_equal_weight_spec(),
        "index_code": "FI00.MISSING_COUNT",
        "status": "draft_extracted",
        "selection": {"total_constituents": None, "buckets": []},
        "open_questions": ["selection.total_constituents not extracted with evidence"],
    }

    candidates = build_engine_promotion_candidates([_kss_spec(), draft_supported, missing_count])

    assert candidates["counts"] == {"total": 1, "by_methodology": {"equal_weighted": 1}}
    assert candidates["items"][0]["index_code"] == "FI00.DRAFT_EQUAL"
    assert candidates["items"][0]["required_review"] == [
        "verify PDF evidence supports extracted total_constituents",
        "verify PDF evidence supports weighting method and cap scope",
        "promote status only after evidence review",
    ]


def test_write_engine_promotion_candidates_outputs_json_and_markdown(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_kss_spec(), {**_equal_weight_spec(), "status": "draft_extracted"}]}), encoding="utf-8")

    json_path, md_path = write_engine_promotion_candidates(specs_path, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.name == "engine_promotion_candidates.json"
    assert md_path.name == "engine_promotion_candidates.md"
    assert payload["counts"]["total"] == 1
    assert "FnGuide Engine Promotion Candidates" in md_path.read_text(encoding="utf-8")


def test_write_target_weights_rejects_requests_for_non_engine_ready_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps(
            {
                "indices": [
                    {
                        "index_code": "FI00.DRAFT",
                        "status": "draft_extracted",
                        "selection": {"total_constituents": 10},
                        "open_questions": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    inputs_path = tmp_path / "engine_inputs.json"
    inputs_path.write_text(
        json.dumps({"requests": [{"index_code": "FI00.DRAFT", "as_of": "2026-05-29", "methodology": "top2_plus"}]}),
        encoding="utf-8",
    )

    with pytest.raises(MethodologyNotReadyError, match="FI00.DRAFT"):
        write_target_weights(inputs_path, specs_path, tmp_path)


def test_write_engine_input_requirements_describes_only_engine_ready_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps(
            {
                "indices": [
                    _kss_spec(),
                    {
                        "index_code": "FI00.DRAFT",
                        "status": "draft_extracted",
                        "selection": {"total_constituents": 10},
                        "open_questions": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = write_engine_input_requirements(specs_path, tmp_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "engine_input_requirements.json"
    assert payload["count"] == 1
    requirement = payload["requirements"][0]
    assert requirement["index_code"] == "FI00.WLT.KSS"
    assert requirement["methodology"] == "top2_plus"
    assert requirement["required_fields"] == ["security_code", "float_market_cap"]
    assert requirement["required_buckets"] == [
        {"name": "top2", "count": 2, "weighting": "fixed"},
        {"name": "momentum", "count": 4, "weighting": "residual_float_market_cap"},
        {"name": "market_cap_fill", "count": 4, "weighting": "residual_float_market_cap"},
    ]


def test_write_engine_input_template_outputs_fillable_requests_for_ready_specs(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_kss_spec(), _equal_weight_spec()]}), encoding="utf-8")

    output_path = write_engine_input_template(specs_path, tmp_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "engine_inputs.template.json"
    assert payload["template_only"] is True
    assert payload["count"] == 2
    kss_request = payload["requests"][0]
    assert kss_request["index_code"] == "FI00.WLT.KSS"
    assert kss_request["methodology"] == "top2_plus"
    assert list(kss_request["constituents_by_bucket"]) == ["top2", "momentum", "market_cap_fill"]
    assert len(kss_request["constituents_by_bucket"]["top2"]) == 2
    assert kss_request["constituents_by_bucket"]["top2"][0] == {"security_code": "", "float_market_cap": None}
    equal_request = payload["requests"][1]
    assert equal_request["index_code"] == "FI00.EQUAL"
    assert equal_request["constituents"] == [{"security_code": ""}, {"security_code": ""}, {"security_code": ""}]


def test_build_methodology_replication_report_smoke_tests_real_engine_ready_specs() -> None:
    report = build_methodology_replication_report(Path("etfs/output/extractions/fnguide/methodology_specs.json"))

    assert report["counts"]["total_specs"] == 59
    assert report["counts"]["engine_ready"] == 12
    assert report["counts"]["target_weight_replication_passed"] == 12
    assert report["counts"]["target_weight_replication_failed"] == 0
    assert report["counts"]["full_methodology_replication_proven"] == 0
    ready_items = [item for item in report["items"] if item["engine_support_status"] == "engine_ready"]
    assert {item["target_weight_replication_status"] for item in ready_items} == {"passed"}
    assert {item["full_methodology_replication_status"] for item in ready_items} == {"not_proven"}


def test_build_methodology_replication_report_marks_kss_proven_when_artifact_proves_validation(tmp_path: Path) -> None:
    replication_path = tmp_path / "kss_replication_validation.json"
    replication_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result": {
                    "index_code": "FI00.WLT.KSS",
                    "as_of": "2026-05-29",
                    "validation_source_type": "official_target_weights",
                    "status": "passed",
                    "checks": {"constituent_membership": "passed", "weight_tolerance": "passed"},
                    "metrics": {},
                    "differences": [],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_methodology_replication_report(
        Path("etfs/output/extractions/fnguide/methodology_specs.json"),
        kss_replication_validation_path=replication_path,
    )

    kss = next(item for item in report["items"] if item["index_code"] == "FI00.WLT.KSS")
    assert kss["full_methodology_replication_status"] == "proven"
    assert kss["full_methodology_replication_evidence"] == replication_path.as_posix()
    assert report["counts"]["full_methodology_replication_proven"] == 1


def test_build_methodology_replication_report_does_not_crash_on_malformed_kss_validation_artifact(tmp_path: Path) -> None:
    replication_path = tmp_path / "kss_replication_validation.json"
    replication_path.write_text("{not-json", encoding="utf-8")

    report = build_methodology_replication_report(
        Path("etfs/output/extractions/fnguide/methodology_specs.json"),
        kss_replication_validation_path=replication_path,
    )

    kss = next(item for item in report["items"] if item["index_code"] == "FI00.WLT.KSS")
    assert kss["full_methodology_replication_status"] == "not_proven"
    assert any("unusable KSS validation artifact" in blocker for blocker in kss["full_methodology_replication_blockers"])
    assert report["counts"]["full_methodology_replication_proven"] == 0


@pytest.mark.parametrize(
    ("artifact_result", "case_name"),
    [
        (
            {
                "index_code": "FI00.WLT.KSS",
                "as_of": "2026-05-29",
                "validation_source_type": "etf_holdings",
                "status": "passed",
                "checks": {"constituent_membership": "passed", "weight_tolerance": "passed"},
                "metrics": {},
                "differences": [],
            },
            "wrong source type",
        ),
        (
            {
                "index_code": "FI00.NOT.KSS",
                "as_of": "2026-05-29",
                "validation_source_type": "official_target_weights",
                "status": "passed",
                "checks": {"constituent_membership": "passed", "weight_tolerance": "passed"},
                "metrics": {},
                "differences": [],
            },
            "wrong index code",
        ),
        (
            {
                "index_code": "FI00.WLT.KSS",
                "as_of": "2026-05-29",
                "validation_source_type": "official_target_weights",
                "status": "failed",
                "checks": {"constituent_membership": "passed", "weight_tolerance": "failed"},
                "metrics": {},
                "differences": [{"security_code": "A000001"}],
            },
            "failed status",
        ),
    ],
)
def test_build_methodology_replication_report_requires_official_kss_validation_pass(
    tmp_path: Path,
    artifact_result: dict[str, object],
    case_name: str,
) -> None:
    replication_path = tmp_path / f"kss_replication_validation_{case_name.replace(' ', '_')}.json"
    replication_path.write_text(json.dumps({"schema_version": "1.0", "result": artifact_result}), encoding="utf-8")

    report = build_methodology_replication_report(
        Path("etfs/output/extractions/fnguide/methodology_specs.json"),
        kss_replication_validation_path=replication_path,
    )

    kss = next(item for item in report["items"] if item["index_code"] == "FI00.WLT.KSS")
    assert kss["full_methodology_replication_status"] == "not_proven"
    assert report["counts"]["full_methodology_replication_proven"] == 0


def test_build_methodology_replication_report_keeps_kss_not_proven_when_target_weight_not_run(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [{**_kss_spec(), "status": "draft_extracted"}]}), encoding="utf-8")
    replication_path = tmp_path / "kss_replication_validation.json"
    replication_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result": {
                    "index_code": "FI00.WLT.KSS",
                    "as_of": "2026-05-29",
                    "validation_source_type": "official_target_weights",
                    "status": "passed",
                    "checks": {"constituent_membership": "passed", "weight_tolerance": "passed"},
                    "metrics": {},
                    "differences": [],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_methodology_replication_report(
        specs_path,
        kss_replication_validation_path=replication_path,
    )

    kss = next(item for item in report["items"] if item["index_code"] == "FI00.WLT.KSS")
    assert kss["target_weight_replication_status"] == "not_run"
    assert kss["full_methodology_replication_status"] == "not_proven"
    assert report["counts"]["full_methodology_replication_proven"] == 0


def test_build_methodology_replication_report_keeps_kss_not_proven_when_target_weight_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replication_path = tmp_path / "kss_replication_validation.json"
    replication_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result": {
                    "index_code": "FI00.WLT.KSS",
                    "as_of": "2026-05-29",
                    "validation_source_type": "official_target_weights",
                    "status": "passed",
                    "checks": {"constituent_membership": "passed", "weight_tolerance": "passed"},
                    "metrics": {},
                    "differences": [],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        methodology_engine,
        "_target_weight_result",
        lambda request, ready_specs: {"checks": {"weight_tolerance": "failed"}, "metrics": {}},
    )

    report = build_methodology_replication_report(
        Path("etfs/output/extractions/fnguide/methodology_specs.json"),
        kss_replication_validation_path=replication_path,
    )

    kss = next(item for item in report["items"] if item["index_code"] == "FI00.WLT.KSS")
    assert kss["target_weight_replication_status"] == "failed"
    assert kss["full_methodology_replication_status"] == "not_proven"
    assert report["counts"]["full_methodology_replication_proven"] == 0


def test_write_methodology_replication_report_outputs_json_and_markdown(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps({"indices": [_kss_spec(), {**_equal_weight_spec(), "status": "draft_extracted"}]}), encoding="utf-8")

    json_path, md_path = write_methodology_replication_report(specs_path, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.name == "methodology_replication_report.json"
    assert md_path.name == "methodology_replication_report.md"
    assert payload["counts"]["target_weight_replication_passed"] == 1
    assert payload["counts"]["full_methodology_replication_proven"] == 0
    assert "FnGuide Methodology Replication Report" in md_path.read_text(encoding="utf-8")


def test_methodology_engine_parser_uses_grouped_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.inputs == "etfs/output/engine/fnguide/engine_inputs.json"
    assert args.specs == "etfs/output/extractions/fnguide/methodology_specs.json"
    assert args.output_dir == "etfs/output/engine/fnguide"
    assert args.write_requirements is False
    assert args.write_template is False
    assert args.write_replication_report is False


def test_methodology_engine_parser_accepts_template_mode() -> None:
    args = build_parser().parse_args(["--write-template"])

    assert args.write_template is True


def test_methodology_engine_parser_accepts_replication_report_mode() -> None:
    args = build_parser().parse_args(["--write-replication-report"])

    assert args.write_replication_report is True
