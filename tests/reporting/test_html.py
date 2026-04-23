from pathlib import Path

import pandas as pd

from backtesting.reporting.composers import ComparisonComposer, TearsheetComposer
from backtesting.reporting.html import HtmlRenderer
from backtesting.reporting.models import BenchmarkConfig, ComparisonBundle, ReportBundle, ReportSpec, SavedRun, TearsheetBundle


def _write_asset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".html":
        path.write_text("<html><body>plot</body></html>", encoding="utf-8")
    else:
        path.write_bytes(b"png")
    return path


def test_html_renderer_uses_tearsheet_template(tmp_path: Path) -> None:
    bundle = TearsheetBundle(
        spec=ReportSpec(
            name="single-report",
            run_ids=("run-a",),
            title="Momentum Tearsheet",
            benchmark=BenchmarkConfig.default_kospi200(),
        ),
        out_dir=tmp_path / "single-report",
        run_id="run-a",
        display_name="Momentum",
        pages={
            "performance": _write_asset(tmp_path / "single-report" / "pages" / "performance.png"),
        },
        tables={
            "performance_summary": pd.DataFrame(
                [
                    {"metric_key": "cagr", "metric": "CAGR", "value": 0.172},
                    {"metric_key": "sharpe", "metric": "Sharpe", "value": 1.1},
                    {"metric_key": "beta", "metric": "Beta", "value": 1.01},
                    {"metric_key": "final_equity", "metric": "Final Equity", "value": 1234567.0},
                ]
            ),
            "drawdown_episodes": pd.DataFrame([{"start": "2022-01-01", "drawdown": "-12.3%"}]),
            "top_holdings": pd.DataFrame([{"symbol": "AAA", "weight": "25.0%"}]),
            "sector_weights": pd.DataFrame([{"sector": "Tech", "weight": "40.0%"}]),
            "validation_appendix": pd.DataFrame([{"note": "missing_factor:run-a"}]),
        },
        notes=("missing_factor:run-a",),
    )

    path = HtmlRenderer().render(bundle)

    html = path.read_text(encoding="utf-8")
    assert path.exists()
    assert path.parent.joinpath("styles.css").exists()
    assert '<section class="report-cover cover">' in html
    assert '<section class="report-section executive-spread">' in html
    assert '<div class="executive-spread">' not in html
    assert html.index('<section class="report-cover cover">') < html.index('<section class="report-section executive-spread">')
    assert "Momentum Tearsheet" in html
    assert "Single-Run Tearsheet" in html
    assert "Momentum" in html
    assert "single-report" in html
    assert "PDF-first single-run performance summary" in html
    assert "KOSPI200" in html
    assert "Executive Summary" in html
    assert "Appendix" in html
    assert "performance.png" in html
    assert "Top Holdings" in html
    assert "17.2%" in html
    assert "1.10" in html
    assert "1.01" in html
    assert "1,234,567" in html
    assert "Document Scope" not in html
    assert "PDF-First Layout" not in html
    assert "Metric Cards" not in html
    assert "Open interactive chart" not in html
    assert "110.0%" not in html
    assert "101.0%" not in html


def test_html_renderer_supports_strategy_only_reports(tmp_path: Path) -> None:
    bundle = TearsheetBundle(
        spec=ReportSpec(
            name="single-report",
            run_ids=("run-a",),
            title="Absolute Tearsheet",
            benchmark=None,
            profile="absolute",
        ),
        out_dir=tmp_path / "single-report",
        run_id="run-a",
        display_name="Absolute",
        pages={"performance": _write_asset(tmp_path / "single-report" / "pages" / "performance.png")},
        tables={"performance_summary": pd.DataFrame([{"metric_key": "cagr", "metric": "CAGR", "value": 0.172}])},
        notes=(),
    )

    path = HtmlRenderer().render(bundle)

    html = path.read_text(encoding="utf-8")
    assert "Strategy Only" in html


def test_html_renderer_uses_comparison_template(tmp_path: Path) -> None:
    bundle = ComparisonBundle(
        spec=ReportSpec(
            name="compare-report",
            run_ids=("run-a", "run-b"),
            title="Strategy Comparison",
            benchmark=BenchmarkConfig.default_kospi200(),
        ),
        out_dir=tmp_path / "compare-report",
        display_names=("Momentum", "Momentum Variant"),
        pages={
            "executive": _write_asset(tmp_path / "compare-report" / "pages" / "executive.png"),
            "performance": _write_asset(tmp_path / "compare-report" / "pages" / "performance.html"),
        },
        tables={
            "ranked_summary": pd.DataFrame(
                [
                    {"display_name": "Momentum", "cagr": 0.172, "sharpe": 1.10},
                    {"display_name": "Momentum Variant", "cagr": 0.150, "sharpe": 1.35},
                ]
            ),
            "benchmark_relative": pd.DataFrame([{"display_name": "Momentum", "alpha": "3.2%"}]),
            "exposure_summary": pd.DataFrame([{"display_name": "Momentum", "holdings_count": "20"}]),
            "sector_summary": pd.DataFrame([{"display_name": "Momentum", "top_sector": "Tech"}]),
        },
        notes=("missing_split:run-b",),
    )

    path = HtmlRenderer().render(bundle)

    html = path.read_text(encoding="utf-8")
    assert path.exists()
    assert '<section class="report-cover cover">' in html
    assert '<section class="report-section executive-spread">' in html
    assert '<div class="executive-spread">' not in html
    assert html.index('<section class="report-cover cover">') < html.index('<section class="report-section executive-spread">')
    assert "Comparison Report" in html
    assert "Strategy Comparison" in html
    assert "KOSPI200" in html
    assert "compare-report" in html
    assert "Cross-strategy comparison optimized for PDF review" in html
    assert "Momentum" in html
    assert "Momentum Variant" in html
    assert "pages/performance.html" in html
    assert '<iframe class="plot-frame"' in html
    assert "Ranked Summary" in html
    assert "Benchmark Relative Metrics" in html
    assert "Holdings And Sector Comparison" in html
    assert "Participants" not in html
    assert "Research-Style Comparison" not in html
    assert "Metric Cards" not in html
    assert "Top CAGR" in html
    assert "Momentum · 17.2%" in html
    assert "Top Sharpe" in html
    assert "Momentum Variant · 1.35" in html
    assert "missing_split:run-b" in html


def test_html_renderer_keeps_composed_report_asset_paths_relative(tmp_path: Path) -> None:
    out_dir = tmp_path / "nested" / "comparison-report"
    bundle = ComparisonBundle(
        spec=ReportSpec(
            name="comparison-report",
            run_ids=("run-a", "run-b"),
            title="Nested Comparison",
            benchmark=BenchmarkConfig.default_kospi200(),
        ),
        out_dir=out_dir,
        display_names=("Momentum", "Momentum Variant"),
        pages={
            "executive": _write_asset(out_dir / "pages" / "executive.png"),
            "performance": _write_asset(out_dir / "pages" / "performance.html"),
        },
        tables={
            "ranked_summary": pd.DataFrame(
                [
                    {"display_name": "Momentum", "cagr": 0.172, "sharpe": 1.10},
                    {"display_name": "Momentum Variant", "cagr": 0.150, "sharpe": 1.35},
                ]
            ),
            "benchmark_relative": pd.DataFrame(),
            "exposure_summary": pd.DataFrame(),
            "sector_summary": pd.DataFrame(),
        },
        notes=(),
    )

    path = HtmlRenderer().render(bundle)

    html = path.read_text(encoding="utf-8")
    assert "pages/executive.png" in html
    assert "pages/performance.html" in html
    assert str(out_dir) not in html


def test_tearsheet_composer_builds_pdf_first_context(tmp_path: Path) -> None:
    bundle = TearsheetBundle(
        spec=ReportSpec(
            name="single-report",
            run_ids=("run-a",),
            title="Momentum Tearsheet",
            benchmark=BenchmarkConfig.default_kospi200(),
        ),
        out_dir=tmp_path / "single-report",
        run_id="run-a",
        display_name="Momentum",
        pages={
            "performance": _write_asset(tmp_path / "single-report" / "pages" / "performance.png"),
        },
        tables={
            "performance_summary": pd.DataFrame(
                [
                    {"metric_key": "cagr", "metric": "CAGR", "value": 0.172},
                    {"metric_key": "sharpe", "metric": "Sharpe", "value": 1.1},
                    {"metric_key": "max_drawdown", "metric": "Max Drawdown", "value": -0.221},
                    {"metric_key": "tracking_error", "metric": "Tracking Error", "value": 0.081},
                    {"metric_key": "final_equity", "metric": "Final Equity", "value": 1234567.0},
                ]
            ),
            "drawdown_episodes": pd.DataFrame([{"start": "2022-01-01", "drawdown": -0.123}]),
            "top_holdings": pd.DataFrame([{"symbol": "AAA", "weight": 0.25}]),
            "sector_weights": pd.DataFrame([{"sector": "Tech", "weight": 0.40}]),
            "validation_appendix": pd.DataFrame([{"note": "missing_factor:run-a"}]),
        },
        notes=("missing_factor:run-a",),
    )

    report = TearsheetComposer().compose(bundle)

    assert report.cover.report_type == "Single-Run Tearsheet"
    assert report.cover.title == "Momentum Tearsheet"
    assert report.cover.benchmark_name == "KOSPI200"
    assert report.cover.report_name == "single-report"
    assert report.cover.descriptor
    assert [(item.label, item.value) for item in report.executive_metrics][:1] == [("CAGR", "17.2%")]
    assert tuple(page.key for page in report.executive_pages) == ("performance",)
    assert tuple(table.key for table in report.executive_tables) == ("performance_summary", "drawdown_episodes")
    assert tuple(section.title for section in report.sections) == (
        "Holdings And Sectors",
        "Appendix",
    )
    assert tuple(page.key for page in report.sections[0].pages) == ()
    assert tuple(table.key for table in report.sections[0].tables) == ("top_holdings", "sector_weights")
    assert tuple(page.key for page in report.sections[1].pages) == ()
    assert tuple(table.key for table in report.sections[1].tables) == ("validation_appendix",)
    assert report.notes == ("missing_factor:run-a",)
    assert len(report.executive_metrics) == 5


def test_comparison_composer_builds_pdf_first_context(tmp_path: Path) -> None:
    bundle = ComparisonBundle(
        spec=ReportSpec(
            name="compare-report",
            run_ids=("run-a", "run-b"),
            title="Strategy Comparison",
            benchmark=BenchmarkConfig.default_kospi200(),
        ),
        out_dir=tmp_path / "compare-report",
        display_names=("Momentum", "Momentum Variant"),
        pages={
            "executive": _write_asset(tmp_path / "compare-report" / "pages" / "executive.png"),
            "performance": _write_asset(tmp_path / "compare-report" / "pages" / "performance.png"),
            "rolling": _write_asset(tmp_path / "compare-report" / "pages" / "rolling.png"),
            "exposure": _write_asset(tmp_path / "compare-report" / "pages" / "exposure.png"),
        },
        tables={
            "ranked_summary": pd.DataFrame(
                [
                    {"display_name": "Momentum", "cagr": 0.172, "sharpe": 1.10},
                    {"display_name": "Momentum Variant", "cagr": 0.150, "sharpe": 1.35},
                ]
            ),
            "benchmark_relative": pd.DataFrame([{"display_name": "Momentum", "alpha": 0.032, "beta": 0.88}]),
            "exposure_summary": pd.DataFrame([{"display_name": "Momentum", "holdings_count": 20}]),
            "sector_summary": pd.DataFrame([{"display_name": "Momentum", "top_sector": "Tech"}]),
        },
        notes=("missing_split:run-b",),
    )

    report = ComparisonComposer().compose(bundle)

    assert report.cover.report_type == "Comparison Report"
    assert report.cover.title == "Strategy Comparison"
    assert report.cover.benchmark_name == "KOSPI200"
    assert report.cover.report_name == "compare-report"
    assert report.cover.descriptor
    assert [item.label for item in report.executive_metrics] == ["Top CAGR", "Top Sharpe"]
    assert report.executive_metrics[0].value == "Momentum · 17.2%"
    assert report.executive_metrics[1].value == "Momentum Variant · 1.35"
    assert tuple(page.key for page in report.executive_pages) == ("executive", "performance")
    assert tuple(table.key for table in report.executive_tables) == ("ranked_summary", "benchmark_relative")
    assert tuple(section.title for section in report.sections) == (
        "Rolling And Relative Diagnostics",
        "Holdings And Sector Comparison",
    )
    assert tuple(page.key for page in report.sections[0].pages) == ("rolling",)
    assert tuple(table.key for table in report.sections[0].tables) == ()
    assert tuple(page.key for page in report.sections[1].pages) == ("exposure",)
    assert tuple(table.key for table in report.sections[1].tables) == ("exposure_summary", "sector_summary")
    assert report.notes == ("missing_split:run-b",)
    assert report.participants == ("Momentum", "Momentum Variant")


def test_html_renderer_keeps_legacy_reportbundle_path_styled(tmp_path: Path) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    run = SavedRun(
        run_id="legacy-run",
        path=tmp_path / "legacy-run",
        config={"strategy": "momentum"},
        summary={"cagr": 0.1},
        equity=pd.Series([100.0, 110.0], index=index),
        returns=pd.Series([0.0, 0.1], index=index),
        turnover=pd.Series([0.0, 0.05], index=index),
        weights=pd.DataFrame({"A": [1.0, 1.0]}, index=index),
        qty=pd.DataFrame({"A": [10.0, 10.0]}, index=index),
    )
    plot_path = _write_asset(tmp_path / "legacy-report" / "plots" / "equity.png")
    bundle = ReportBundle(
        spec=ReportSpec(name="legacy-report", run_ids=("legacy-run",)),
        out_dir=tmp_path / "legacy-report",
        runs=(run,),
        summary=pd.DataFrame([{"run_id": "legacy-run", "cagr": 0.1}]),
        appendix=pd.DataFrame([{"run_id": "legacy-run"}]),
        plots={"equity": plot_path},
        notes=(),
    )

    path = HtmlRenderer().render(bundle)

    html = path.read_text(encoding="utf-8")
    css = path.parent.joinpath("styles.css").read_text(encoding="utf-8")
    assert '<div class="plot-grid">' in html
    assert ".report-cover" in css
    assert ".executive-spread" in css
    assert ".metric-strip" in css
    assert ".compact-table-block" in css
    assert ".plot-grid" in css


def test_html_renderer_supports_html_page_asset_fallback_for_new_templates(tmp_path: Path) -> None:
    bundle = TearsheetBundle(
        spec=ReportSpec(
            name="fallback-report",
            run_ids=("run-a",),
            title="Fallback Tearsheet",
            benchmark=BenchmarkConfig.default_kospi200(),
        ),
        out_dir=tmp_path / "fallback-report",
        run_id="run-a",
        display_name="Momentum",
        pages={
            "performance": _write_asset(tmp_path / "fallback-report" / "pages" / "performance.html"),
        },
        tables={
            "performance_summary": pd.DataFrame([{"metric_key": "cagr", "metric": "CAGR", "value": 0.172}]),
            "drawdown_episodes": pd.DataFrame(),
            "top_holdings": pd.DataFrame(),
            "sector_weights": pd.DataFrame(),
            "validation_appendix": pd.DataFrame(),
        },
        notes=(),
    )

    path = HtmlRenderer().render(bundle)

    html = path.read_text(encoding="utf-8")
    assert '<iframe class="plot-frame"' in html
    assert "pages/performance.html" in html
