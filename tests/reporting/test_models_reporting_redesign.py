from pathlib import Path

import pandas as pd

from backtesting.reporting.models import BenchmarkConfig, ComparisonBundle, ReportKind, ReportSpec, TearsheetBundle


def test_report_spec_defaults_to_tearsheet_for_single_run() -> None:
    spec = ReportSpec(name="single", run_ids=("run-a",))
    assert spec.kind is ReportKind.TEARSHEET
    assert spec.benchmark.code == "IKS200"
    assert spec.benchmark.name == "KOSPI200"


def test_report_spec_defaults_to_comparison_for_multiple_runs() -> None:
    spec = ReportSpec(name="compare", run_ids=("run-a", "run-b"))
    assert spec.kind is ReportKind.COMPARISON


def test_report_spec_normalizes_string_kind_to_enum() -> None:
    spec = ReportSpec(name="single", run_ids=("run-a",), kind="tearsheet")

    assert spec.kind is ReportKind.TEARSHEET


def test_report_spec_positional_arguments_remain_backward_compatible() -> None:
    spec = ReportSpec("legacy", ("run-a",), "Legacy Title", False, False, True, ("pdf",), None, None)

    assert spec.name == "legacy"
    assert spec.run_ids == ("run-a",)
    assert spec.title == "Legacy Title"
    assert spec.include_factor is False
    assert spec.include_validation is False
    assert spec.include_is_oos is True
    assert spec.formats == ("pdf",)
    assert spec.kind is ReportKind.TEARSHEET


def test_report_spec_rejects_empty_run_ids() -> None:
    try:
        ReportSpec(name="empty", run_ids=())
    except ValueError as exc:
        assert "run_ids" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_report_spec_rejects_comparison_with_one_run() -> None:
    try:
        ReportSpec(name="single", run_ids=("run-a",), kind=ReportKind.COMPARISON)
    except ValueError as exc:
        assert "COMPARISON" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_report_spec_rejects_tearsheet_with_multiple_runs() -> None:
    try:
        ReportSpec(name="compare", run_ids=("run-a", "run-b"), kind=ReportKind.TEARSHEET)
    except ValueError as exc:
        assert "TEARSHEET" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_report_spec_rejects_invalid_kind_values() -> None:
    for invalid_kind in ("not-a-kind", 123):
        try:
            ReportSpec(name="single", run_ids=("run-a",), kind=invalid_kind)
        except ValueError as exc:
            assert "invalid report kind" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_bundles_expose_display_metadata(tmp_path: Path) -> None:
    bundle = TearsheetBundle(
        spec=ReportSpec(name="single", run_ids=("run-a",)),
        out_dir=tmp_path,
        run_id="run-a",
        display_name="Momentum",
        pages={"executive": tmp_path / "executive.png"},
        tables={"summary": pd.DataFrame([{"metric": "CAGR", "value": 0.1}])},
        notes=(),
    )
    assert bundle.display_name == "Momentum"
    assert "executive" in bundle.pages


def test_comparison_bundle_exposes_basic_metadata(tmp_path: Path) -> None:
    bundle = ComparisonBundle(
        spec=ReportSpec(name="compare", run_ids=("run-a", "run-b")),
        out_dir=tmp_path,
        display_names=("Momentum", "Mean Reversion"),
        pages={"summary": tmp_path / "summary.png"},
        tables={"summary": pd.DataFrame([{"metric": "CAGR", "value": 0.1}])},
    )

    assert bundle.display_names == ("Momentum", "Mean Reversion")
    assert "summary" in bundle.pages
    assert bundle.notes == ()
