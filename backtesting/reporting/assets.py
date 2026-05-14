from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    from backtesting.run import RunReport


_EMPTY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBgL9enZsAAAAASUVORK5CYII="
)


@dataclass(slots=True)
class ReportAssetWriter:
    plot_series: Callable[[Path, pd.Series, str, str], None]

    def write(self, report: "RunReport", run_dir: Path) -> None:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        self.plot_series(
            plots_dir / "equity.png",
            report.result.equity,
            "Equity Curve",
            "Equity",
        )
        self.plot_series(
            plots_dir / "drawdown.png",
            self._drawdown(report.result.equity),
            "Drawdown",
            "Drawdown",
        )
        self._write_performance_page(report, run_dir)

    def _write_performance_page(self, report: "RunReport", run_dir: Path) -> None:
        pages_dir = run_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        performance_path = pages_dir / "performance.png"
        try:
            from .builder import ReportBuilder
            from .figures import TearsheetFigureBuilder
            from .models import SavedRun
            from .snapshots import PerformanceSnapshotFactory
            from .writer import RunWriter

            saved_run = SavedRun(
                run_id=run_dir.name,
                path=run_dir,
                config=asdict(report.config),
                summary=report.summary,
                equity=report.result.equity,
                returns=report.result.returns,
                turnover=report.result.turnover,
                weights=report.result.weights,
                qty=report.result.qty,
                monthly_returns=RunWriter._monthly_returns(report.result.returns),
                latest_qty=RunWriter._latest_qty(report),
                latest_weights=RunWriter._latest_weights(report),
                bucket_ledger=None if report.position_plan is None else RunWriter._bucket_ledger(report),
                validation=getattr(report, "validation", None) or {"warnings": []},
                split=getattr(report, "split", None) or {"is": None, "oos": None},
                factor=getattr(report, "factor", None) or {"metrics": {}},
            )
            benchmark = self._benchmark_config(report)
            benchmark_repo, sector_repo = ReportBuilder._repositories_for_run(saved_run)
            snapshot = PerformanceSnapshotFactory(
                benchmark_repo=benchmark_repo,
                sector_repo=sector_repo,
            ).build(saved_run, benchmark, None)
            TearsheetFigureBuilder(pages_dir).build(snapshot, require_png=True)
        except Exception:
            performance_path.write_bytes(_EMPTY_PNG)

    @staticmethod
    def _benchmark_config(report: "RunReport"):
        code = report.config.benchmark_code
        name = report.config.benchmark_name
        if not code or not name:
            return None
        from .models import BenchmarkConfig

        return BenchmarkConfig(
            code=str(code),
            name=str(name),
            dataset=str(report.config.benchmark_dataset or "qw_BM"),
        )

    @staticmethod
    def _drawdown(equity: pd.Series) -> pd.Series:
        peak = equity.cummax()
        return equity.div(peak).sub(1.0).astype(float)
