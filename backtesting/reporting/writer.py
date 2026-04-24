from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from backtesting.run import RunReport


_EMPTY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBgL9enZsAAAAASUVORK5CYII="
)


@dataclass(slots=True)
class RunWriter:
    root_dir: Path

    def write(self, report: "RunReport") -> Path:
        run_dir = self._run_dir(report)
        series_dir = run_dir / "series"
        positions_dir = run_dir / "positions"
        plots_dir = run_dir / "plots"

        for path in (run_dir, series_dir, positions_dir, plots_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._write_json(run_dir / "config.json", asdict(report.config))
        self._write_json(run_dir / "summary.json", report.summary)
        self._write_json(run_dir / "validation.json", getattr(report, "validation", None) or {"warnings": []})
        self._write_json(run_dir / "split.json", getattr(report, "split", None) or {"is": None, "oos": None})
        self._write_json(run_dir / "factor.json", getattr(report, "factor", None) or {"metrics": {}})
        if getattr(report, "resolved_spec", None) is not None:
            self._write_json(run_dir / "resolved_execution_spec.json", asdict(report.resolved_spec))
        if getattr(report, "execution_resolution", None) is not None:
            self._write_json(run_dir / "execution_resolution.json", report.execution_resolution)

        report.result.equity.rename("equity").to_csv(series_dir / "equity.csv", index_label="date")
        report.result.returns.rename("returns").to_csv(series_dir / "returns.csv", index_label="date")
        report.result.turnover.rename("turnover").to_csv(series_dir / "turnover.csv", index_label="date")
        self._monthly_returns(report.result.returns).rename("monthly_returns").to_csv(
            series_dir / "monthly_returns.csv",
            index_label="date",
        )

        report.result.weights.to_parquet(positions_dir / "weights.parquet")
        report.result.qty.to_parquet(positions_dir / "qty.parquet")
        if report.position_plan is not None:
            self._bucket_ledger(report).to_parquet(positions_dir / "bucket_ledger.parquet")
        self._latest_qty(report).to_csv(positions_dir / "latest_qty.csv", index=False)
        self._latest_weights(report).to_csv(positions_dir / "latest_weights.csv", index=False)

        self._plot_series(
            path=plots_dir / "equity.png",
            series=report.result.equity,
            title="Equity Curve",
            ylabel="Equity",
        )
        self._plot_series(
            path=plots_dir / "drawdown.png",
            series=self._drawdown(report.result.equity),
            title="Drawdown",
            ylabel="Drawdown",
        )
        self._write_performance_page(report, run_dir)
        return run_dir

    def _run_dir(self, report: "RunReport") -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = report.config.name or f"{report.config.strategy}_{report.config.start}_{report.config.end}"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()
        return self.root_dir / f"{slug}_{stamp}"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_performance_page(self, report: "RunReport", run_dir: Path) -> None:
        pages_dir = run_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        performance_path = pages_dir / "performance.png"
        try:
            from .builder import ReportBuilder
            from .figures import TearsheetFigureBuilder
            from .models import SavedRun
            from .snapshots import PerformanceSnapshotFactory

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
                monthly_returns=self._monthly_returns(report.result.returns),
                latest_qty=self._latest_qty(report),
                latest_weights=self._latest_weights(report),
                bucket_ledger=None if report.position_plan is None else self._bucket_ledger(report),
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
    def _monthly_returns(returns: pd.Series) -> pd.Series:
        monthly = (1.0 + returns).resample("ME").prod().sub(1.0)
        monthly.index = monthly.index.normalize()
        return monthly.astype(float)

    @staticmethod
    def _drawdown(equity: pd.Series) -> pd.Series:
        peak = equity.cummax()
        return equity.div(peak).sub(1.0).astype(float)

    @staticmethod
    def _latest_qty(report: "RunReport") -> pd.DataFrame:
        last_qty = report.result.qty.iloc[-1]
        frame = pd.DataFrame(
            {
                "symbol": last_qty.index,
                "qty": last_qty.values,
            }
        )
        frame = frame.loc[frame["qty"].ne(0.0)].copy()
        frame["abs_qty"] = frame["qty"].abs()
        return frame.sort_values(["abs_qty", "symbol"], ascending=[False, True]).reset_index(drop=True)

    @staticmethod
    def _latest_weights(report: "RunReport") -> pd.DataFrame:
        last_weight = report.result.weights.iloc[-1]
        frame = pd.DataFrame(
            {
                "symbol": last_weight.index,
                "target_weight": last_weight.values,
            }
        )
        frame = frame.loc[frame["target_weight"].ne(0.0)].copy()
        frame["abs_weight"] = frame["target_weight"].abs()
        return frame.sort_values(["abs_weight", "symbol"], ascending=[False, True]).reset_index(drop=True)

    @staticmethod
    def _bucket_ledger(report: "RunReport") -> pd.DataFrame:
        ledger = report.position_plan.bucket_ledger.copy()
        if ledger.empty or report.result.equity.empty:
            return ledger

        start = report.result.equity.index.min()
        end = report.result.equity.index.max()
        ledger_dates = pd.to_datetime(ledger["date"])
        return ledger.loc[ledger_dates.between(start, end)].copy()

    @staticmethod
    def _plot_series(path: Path, series: pd.Series, title: str, ylabel: str) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            path.write_bytes(_EMPTY_PNG)
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        series.plot(ax=ax, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
