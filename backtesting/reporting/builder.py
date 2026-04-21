from __future__ import annotations

from pathlib import Path

import pandas as pd

from .benchmarks import BenchmarkRepository, SectorRepository
from .benchmarks import default_repositories_for_universe
from .comparison_figures import ComparisonFigureBuilder
from .models import ComparisonBundle, ReportBundle, ReportKind, ReportSpec, SavedRun, TearsheetBundle
from .figures import TearsheetFigureBuilder
from .plots import PlotLibrary
from .snapshots import PerformanceSnapshotFactory
from .tables import build_appendix_table, build_latest_qty_table, build_latest_weights_table, build_summary_table
from .tables_comparison import ComparisonTableBuilder
from .tables_single import TearsheetTableBuilder

__all__ = ("ReportBuilder",)


class ReportBuilder:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)

    def build(self, spec: ReportSpec, runs: list[SavedRun]) -> TearsheetBundle | ComparisonBundle:
        out_dir = self.root_dir / spec.name
        pages_dir = out_dir / "pages"
        tables_dir = out_dir / "tables"
        for path in (out_dir, pages_dir, tables_dir):
            path.mkdir(parents=True, exist_ok=True)

        notes = self._build_notes(spec, runs)
        snapshots = self._build_snapshots(runs, spec)
        if spec.kind is ReportKind.TEARSHEET:
            return self._build_tearsheet_bundle(spec, out_dir, pages_dir, tables_dir, snapshots[0], notes)
        return self._build_comparison_bundle(spec, out_dir, pages_dir, tables_dir, snapshots, notes)

    def build_legacy(self, spec: ReportSpec, runs: list[SavedRun]) -> ReportBundle:
        out_dir = self.root_dir / spec.name
        plots_dir = out_dir / "plots"
        tables_dir = out_dir / "tables"
        for path in (out_dir, plots_dir, tables_dir):
            path.mkdir(parents=True, exist_ok=True)

        plotter = PlotLibrary(plots_dir)
        plots = {
            "equity": plotter.equity(runs),
            "drawdown": plotter.drawdown(runs),
            "turnover": plotter.turnover(runs),
            "top_weights": plotter.top_weights(runs),
            "monthly_heatmap": plotter.monthly_heatmap(runs),
        }

        summary = build_summary_table(runs)
        appendix = build_appendix_table(runs)
        self._write_legacy_table(tables_dir / "summary.csv", summary)
        self._write_legacy_table(tables_dir / "appendix.csv", appendix)
        for run in runs:
            self._write_legacy_table(tables_dir / f"{run.run_id}_latest_weights.csv", build_latest_weights_table(run))
            self._write_legacy_table(tables_dir / f"{run.run_id}_latest_qty.csv", build_latest_qty_table(run))

        return ReportBundle(
            spec=spec,
            out_dir=out_dir,
            runs=tuple(runs),
            summary=summary,
            appendix=appendix,
            plots=plots,
            notes=self._build_notes(spec, runs),
        )

    def _build_snapshots(self, runs: list[SavedRun], spec: ReportSpec) -> list[object]:
        snapshots: list[object] = []
        for run in runs:
            benchmark_repo, sector_repo = default_repositories_for_universe(
                str(run.config.get("universe_id")) if run.config.get("universe_id") is not None else None
            )
            factory = PerformanceSnapshotFactory(
                benchmark_repo=benchmark_repo,
                sector_repo=sector_repo,
            )
            snapshots.append(factory.build(run, spec.benchmark))
        return snapshots

    def _build_tearsheet_bundle(
        self,
        spec: ReportSpec,
        out_dir: Path,
        pages_dir: Path,
        tables_dir: Path,
        snapshot: object,
        notes: tuple[str, ...],
    ) -> TearsheetBundle:
        pages = TearsheetFigureBuilder(pages_dir).build(snapshot)
        tables = TearsheetTableBuilder().build(snapshot, notes=notes)
        self._write_tables(tables_dir, tables)
        return TearsheetBundle(
            spec=spec,
            out_dir=out_dir,
            run_id=str(snapshot.run_id),
            display_name=str(snapshot.display_name),
            pages=pages,
            tables=tables,
            notes=notes,
        )

    def _build_comparison_bundle(
        self,
        spec: ReportSpec,
        out_dir: Path,
        pages_dir: Path,
        tables_dir: Path,
        snapshots: list[object],
        notes: tuple[str, ...],
    ) -> ComparisonBundle:
        pages = ComparisonFigureBuilder(pages_dir).build(snapshots)
        tables = ComparisonTableBuilder().build(snapshots)
        self._write_tables(tables_dir, tables)
        return ComparisonBundle(
            spec=spec,
            out_dir=out_dir,
            display_names=tuple(str(snapshot.display_name) for snapshot in snapshots),
            pages=pages,
            tables=tables,
            notes=notes,
        )

    @staticmethod
    def _write_tables(tables_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
        for name, table in tables.items():
            table.to_csv(tables_dir / f"{name}.csv", index=False)

    @staticmethod
    def _write_legacy_table(path: Path, table: pd.DataFrame) -> None:
        table.to_csv(path, index=False)

    @staticmethod
    def _build_notes(spec: ReportSpec, runs: list[SavedRun]) -> tuple[str, ...]:
        notes: list[str] = []
        for run in runs:
            if spec.include_validation and run.validation is None:
                notes.append(f"missing_validation:{run.run_id}")
            if spec.include_is_oos and run.split is None:
                notes.append(f"missing_split:{run.run_id}")
            if spec.include_factor and run.factor is None:
                notes.append(f"missing_factor:{run.run_id}")
        return tuple(notes)
