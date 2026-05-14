from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .assets import ReportAssetWriter, _EMPTY_PNG

if TYPE_CHECKING:
    from backtesting.run import RunReport


@dataclass(slots=True)
class RunWriter:
    root_dir: Path
    write_report_assets: bool = True

    def write(self, report: "RunReport") -> Path:
        run_dir = self._run_dir(report)
        series_dir = run_dir / "series"
        positions_dir = run_dir / "positions"

        output_dirs = (run_dir, series_dir, positions_dir)
        for path in output_dirs:
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
        if getattr(report, "timing", None) is not None:
            self.write_timing(run_dir, report.timing)

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

        if self.write_report_assets:
            ReportAssetWriter(plot_series=self._plot_series).write(report, run_dir)
        return run_dir

    def write_timing(self, run_dir: Path, timing: dict[str, float]) -> None:
        self._write_json(run_dir / "timing.json", timing)

    def _run_dir(self, report: "RunReport") -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = report.config.name or f"{report.config.strategy}_{report.config.start}_{report.config.end}"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()
        return self.root_dir / f"{slug}_{stamp}"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _monthly_returns(returns: pd.Series) -> pd.Series:
        monthly = (1.0 + returns).resample("ME").prod().sub(1.0)
        monthly.index = monthly.index.normalize()
        return monthly.astype(float)

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
