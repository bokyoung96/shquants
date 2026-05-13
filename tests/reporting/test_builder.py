from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backtesting.reporting.builder import ReportBuilder
from backtesting.reporting.models import ComparisonBundle, ReportBundle, ReportKind, ReportProfile, ReportSpec, SavedRun, TearsheetBundle


def _sample_run(tmp_path: Path, run_id: str, strategy: str = "trend_rank", name: str | None = None) -> SavedRun:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return SavedRun(
        run_id=run_id,
        path=tmp_path / run_id,
        config={"strategy": strategy, "name": name or strategy.replace("_", " ").title()},
        summary={"cagr": 0.1, "mdd": -0.2, "sharpe": 1.0, "final_equity": 110.0, "avg_turnover": 0.05},
        equity=pd.Series([100.0, 110.0], index=index),
        returns=pd.Series([0.0, 0.1], index=index),
        turnover=pd.Series([0.0, 0.05], index=index),
        weights=pd.DataFrame({"A": [1.0, 1.0]}, index=index),
        qty=pd.DataFrame({"A": [10.0, 10.0]}, index=index),
    )


def test_report_builder_creates_tearsheet_bundle_and_persists_tables(tmp_path: Path, monkeypatch) -> None:
    run = _sample_run(tmp_path, "sample")

    class _FakeFactory:
        def build(self, run_obj, benchmark, profile=None):  # type: ignore[no-untyped-def]
            return SimpleNamespace(run_id=run_obj.run_id, display_name="Trend Rank")

    class _FakeTearsheetFigureBuilder:
        def __init__(self, out_dir: Path) -> None:
            self.out_dir = out_dir

        def build(self, snapshot, *, require_png=False):  # type: ignore[no-untyped-def]
            path = self.out_dir / "performance.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
            return {"performance": path}

    class _FakeTearsheetTableBuilder:
        def build(self, snapshot, *, notes=()):  # type: ignore[no-untyped-def]
            return {"performance_summary": pd.DataFrame([{"metric_key": "cagr", "metric": "CAGR", "value": 0.1}])}

    monkeypatch.setattr("backtesting.reporting.builder.BenchmarkRepository.default", lambda: object())
    monkeypatch.setattr("backtesting.reporting.builder.SectorRepository.default", lambda: object())
    monkeypatch.setattr("backtesting.reporting.builder.PerformanceSnapshotFactory", lambda *args, **kwargs: _FakeFactory())
    monkeypatch.setattr("backtesting.reporting.builder.TearsheetFigureBuilder", _FakeTearsheetFigureBuilder)
    monkeypatch.setattr("backtesting.reporting.builder.TearsheetTableBuilder", _FakeTearsheetTableBuilder)

    bundle = ReportBuilder(tmp_path).build(ReportSpec(name="sample-report", run_ids=("sample",)), [run])

    assert isinstance(bundle, TearsheetBundle)
    assert bundle.spec.kind is ReportKind.TEARSHEET
    assert bundle.run_id == "sample"
    assert bundle.display_name == "Trend Rank"
    assert set(bundle.pages) == {"performance"}
    assert set(bundle.tables) == {"performance_summary"}
    assert (bundle.out_dir / "tables" / "performance_summary.csv").exists()


def test_report_builder_uses_universe_specific_repositories(tmp_path: Path, monkeypatch) -> None:
    run = _sample_run(tmp_path, "sample")
    run = SavedRun(
        run_id=run.run_id,
        path=run.path,
        config={**run.config, "universe_id": "kosdaq150"},
        summary=run.summary,
        equity=run.equity,
        returns=run.returns,
        turnover=run.turnover,
        weights=run.weights,
        qty=run.qty,
    )

    benchmark_repo = object()
    sector_repo = object()
    observed: list[tuple[object, object, object]] = []

    def _fake_resolver(universe_id: str | None) -> tuple[object, object]:
        assert universe_id == "kosdaq150"
        return benchmark_repo, sector_repo

    class _FakeFactory:
        def __init__(self, *, benchmark_repo, sector_repo):  # type: ignore[no-untyped-def]
            observed.append((benchmark_repo, sector_repo, "init"))

        def build(self, run_obj, benchmark, profile=None):  # type: ignore[no-untyped-def]
            observed.append((run_obj.config["universe_id"], benchmark.code, benchmark.name, profile))
            return SimpleNamespace(run_id=run_obj.run_id, display_name="Trend Rank")

    monkeypatch.setattr("backtesting.reporting.builder.default_repositories_for_universe", _fake_resolver, raising=False)
    monkeypatch.setattr("backtesting.reporting.builder.PerformanceSnapshotFactory", _FakeFactory)
    monkeypatch.setattr("backtesting.reporting.builder.TearsheetFigureBuilder", lambda out_dir: SimpleNamespace(build=lambda snapshot, require_png=False: {"performance": out_dir / "performance.png"}))
    monkeypatch.setattr("backtesting.reporting.builder.TearsheetTableBuilder", lambda: SimpleNamespace(build=lambda snapshot, notes=(): {"performance_summary": pd.DataFrame([{"metric_key": "cagr", "metric": "CAGR", "value": 0.1}])}))

    bundle = ReportBuilder(tmp_path).build(ReportSpec(name="sample-report", run_ids=("sample",)), [run])

    assert isinstance(bundle, TearsheetBundle)
    assert observed[0][:2] == (benchmark_repo, sector_repo)
    assert observed[1] == ("kosdaq150", "IKS200", "KOSPI200", None)


def test_report_builder_creates_comparison_bundle_for_multiple_runs(tmp_path: Path, monkeypatch) -> None:
    runs = [_sample_run(tmp_path, "run-a", "trend_rank"), _sample_run(tmp_path, "run-b", "trend_rank", name="Trend Rank Variant")]

    class _FakeFactory:
        def build(self, run_obj, benchmark, profile=None):  # type: ignore[no-untyped-def]
            return SimpleNamespace(run_id=run_obj.run_id, display_name=str(run_obj.config["name"]))

    class _FakeComparisonFigureBuilder:
        def __init__(self, out_dir: Path) -> None:
            self.out_dir = out_dir

        def build(self, snapshots, *, require_png=False):  # type: ignore[no-untyped-def]
            path = self.out_dir / "performance.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
            return {"performance": path}

    class _FakeComparisonTableBuilder:
        def build(self, snapshots):  # type: ignore[no-untyped-def]
            return {"ranked_summary": pd.DataFrame([{"display_name": "Trend Rank", "cagr": 0.1}])}

    monkeypatch.setattr("backtesting.reporting.builder.BenchmarkRepository.default", lambda: object())
    monkeypatch.setattr("backtesting.reporting.builder.SectorRepository.default", lambda: object())
    monkeypatch.setattr("backtesting.reporting.builder.PerformanceSnapshotFactory", lambda *args, **kwargs: _FakeFactory())
    monkeypatch.setattr("backtesting.reporting.builder.ComparisonFigureBuilder", _FakeComparisonFigureBuilder)
    monkeypatch.setattr("backtesting.reporting.builder.ComparisonTableBuilder", _FakeComparisonTableBuilder)

    bundle = ReportBuilder(tmp_path).build(ReportSpec(name="compare-report", run_ids=("run-a", "run-b")), runs)

    assert isinstance(bundle, ComparisonBundle)
    assert bundle.spec.kind is ReportKind.COMPARISON
    assert bundle.display_names == ("Trend Rank", "Trend Rank Variant")
    assert set(bundle.pages) == {"performance"}
    assert set(bundle.tables) == {"ranked_summary"}
    assert (bundle.out_dir / "tables" / "ranked_summary.csv").exists()


def test_report_builder_passes_profile_to_snapshot_factory(tmp_path: Path, monkeypatch) -> None:
    run = _sample_run(tmp_path, "sample")
    observed: list[object] = []

    class _FakeFactory:
        def __init__(self, *, benchmark_repo, sector_repo):  # type: ignore[no-untyped-def]
            pass

        def build(self, run_obj, benchmark, profile=None):  # type: ignore[no-untyped-def]
            observed.append(profile)
            return SimpleNamespace(run_id=run_obj.run_id, display_name="Trend Rank")

    monkeypatch.setattr("backtesting.reporting.builder.PerformanceSnapshotFactory", _FakeFactory)
    monkeypatch.setattr("backtesting.reporting.builder.default_repositories_for_universe", lambda universe_id: (object(), object()), raising=False)
    monkeypatch.setattr("backtesting.reporting.builder.TearsheetFigureBuilder", lambda out_dir: SimpleNamespace(build=lambda snapshot, require_png=False: {"performance": out_dir / "performance.png"}))
    monkeypatch.setattr("backtesting.reporting.builder.TearsheetTableBuilder", lambda: SimpleNamespace(build=lambda snapshot, notes=(): {"performance_summary": pd.DataFrame([{"metric_key": "cagr", "metric": "CAGR", "value": 0.1}])}))

    bundle = ReportBuilder(tmp_path).build(
        ReportSpec(name="sample-report", run_ids=("sample",), profile=ReportProfile.INDEX),
        [run],
    )

    assert isinstance(bundle, TearsheetBundle)
    assert observed == [ReportProfile.INDEX]


def test_report_builder_falls_back_when_default_repositories_are_unavailable(tmp_path: Path, monkeypatch) -> None:
    run = _sample_run(tmp_path, "sample")

    monkeypatch.setattr(
        "backtesting.reporting.builder.default_repositories_for_universe",
        lambda universe_id: (_ for _ in ()).throw(FileNotFoundError("missing raw dataset")),
        raising=False,
    )

    bundle = ReportBuilder(tmp_path).build(
        ReportSpec(name="sample-report", run_ids=("sample",), benchmark=None, profile=ReportProfile.ABSOLUTE),
        [run],
    )

    assert isinstance(bundle, TearsheetBundle)
    assert bundle.pages["performance"].exists()


def test_report_builder_legacy_path_remains_available(tmp_path: Path, monkeypatch) -> None:
    run = _sample_run(tmp_path, "sample")

    plot_dir = tmp_path / "legacy-report" / "plots"
    plot_paths = {
        "equity": plot_dir / "equity.png",
        "drawdown": plot_dir / "drawdown.png",
        "turnover": plot_dir / "turnover.png",
        "top_weights": plot_dir / "top_weights.png",
        "monthly_heatmap": plot_dir / "monthly_heatmap.png",
    }

    def _make_plotter(method_name: str):
        def _plot(self, runs, *, require_png=False):  # type: ignore[no-untyped-def]
            path = plot_paths[method_name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(method_name, encoding="utf-8")
            return path

        return _plot

    for method_name in plot_paths:
        monkeypatch.setattr(
            "backtesting.reporting.plots.PlotLibrary." + method_name,
            _make_plotter(method_name),
        )

    bundle = ReportBuilder(tmp_path).build_legacy(ReportSpec(name="legacy-report", run_ids=("sample",)), [run])

    assert isinstance(bundle, ReportBundle)
    assert bundle.plots == plot_paths
    assert bundle.summary.iloc[0]["run_id"] == "sample"
