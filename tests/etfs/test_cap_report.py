import json
from pathlib import Path

from etfs.common.cap import (
    build_cap_candidate_report,
    build_parser,
    cap_policy_from_methodology_spec,
    write_cap_candidate_report,
)
from etfs import paths
from etfs.fnguide.validation import ValidationFixture
from etfs.common.holdings import ValidationHolding, ValidationSnapshot


def _top_bucket_cap_spec() -> dict[str, object]:
    return {
        "index_code": "FI00.EXAMPLE.TOP",
        "index_name": "Example Top Bucket Index",
        "methodology_version": "v1.2",
        "selection": {
            "total_constituents": 3,
            "buckets": [
                {"name": "leaders", "count": 2, "weight": {"type": "fixed", "value": 0.25}},
                {"name": "residual", "count": 1},
            ],
        },
        "weighting": {"base": "fixed_plus_residual", "residual": {"cap": 0.15}},
    }


def test_cap_policy_from_methodology_spec_uses_the_highest_explicit_security_level_cap() -> None:
    policy = cap_policy_from_methodology_spec(_top_bucket_cap_spec())

    assert policy.index_code == "FI00.EXAMPLE.TOP"
    assert policy.index_name == "Example Top Bucket Index"
    assert policy.methodology_version == "v1.2"
    assert policy.regular_security_cap == 0.25


def test_build_cap_candidate_report_uses_latest_holdings_snapshot_with_quantity_and_market_value() -> None:
    fixture = ValidationFixture(
        schema_version="1.0",
        source_type="etf_portfolio_component_xlsx",
        etf_code="123456",
        etf_code_raw="A123456",
        etf_name="Example ETF",
        index_code="FI00.EXAMPLE.TOP",
        source={"path": "holdings.xlsx"},
        snapshots=[
            ValidationSnapshot(
                as_of="2026-06-14",
                equity_holdings=[ValidationHolding("000001", "A000001", "Old", 1, 10, 0.40)],
                cash={},
            ),
            ValidationSnapshot(
                as_of="2026-06-16",
                equity_holdings=[
                    ValidationHolding("000001", "A000001", "Above Cap", 11, 330000, 0.2575),
                    ValidationHolding("000002", "A000002", "Below Cap", 7, 210000, 0.2475),
                ],
                cash={"weight": 0.001},
            ),
        ],
    )

    report = build_cap_candidate_report([fixture], [_top_bucket_cap_spec()])

    assert report["count"] == 1
    assert report["items"] == [
        {
            "etf_code": "123456",
            "etf_name": "Example ETF",
            "index_code": "FI00.EXAMPLE.TOP",
            "index_name": "Example Top Bucket Index",
            "as_of": "2026-06-16",
            "event_type": "regular_cap_excess",
            "security_code": "000001",
            "security_name": "Above Cap",
            "quantity": 11.0,
            "market_value": 330000.0,
            "weight": 0.2575,
            "cap": 0.25,
            "excess_weight": 0.0075,
            "effective_date": "",
        }
    ]


def test_write_cap_candidate_report_writes_json_and_markdown(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "validation_fixtures.json"
    specs_path = tmp_path / "methodology_specs.json"
    fixtures_path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "schema_version": "1.0",
                        "source_type": "etf_portfolio_component_xlsx",
                        "etf_code": "123456",
                        "etf_code_raw": "A123456",
                        "etf_name": "Example | ETF",
                        "index_code": "FI00.EXAMPLE.TOP",
                        "source": {"path": "holdings.xlsx"},
                        "snapshots": [
                            {
                                "as_of": "2026-06-16",
                                "equity_holdings": [
                                    {
                                        "ticker": "000001",
                                        "ticker_raw": "A000001",
                                        "name": "Above | Cap",
                                        "quantity": 11,
                                        "amount": 330000,
                                        "weight": 0.2575,
                                    }
                                ],
                                "cash": {},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs_path.write_text(json.dumps({"indices": [_top_bucket_cap_spec()]}), encoding="utf-8")

    json_path, markdown_path = write_cap_candidate_report(fixtures_path, specs_path, tmp_path / "out")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert json_path.name == "cap_candidates.json"
    assert markdown_path.name == "cap_candidates.md"
    assert payload["items"][0]["security_code"] == "000001"
    assert "Example \\| ETF" in markdown
    assert "Above \\| Cap" in markdown


def test_cap_parser_uses_grouped_validation_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.fixtures == paths.VALIDATION_FIXTURES_JSON.as_posix()
    assert args.specs == paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix()
    assert args.output_dir == paths.VALIDATION_OUTPUT_DIR.as_posix()
