from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .engine import INVESTABLE_ASSETS, BacktestResult
from .plots import plot_drawdown, plot_equity, plot_summary, plot_weights


@dataclass(frozen=True, slots=True)
class BacktestReportPaths:
    spec_path: Path
    metrics_path: Path
    returns_path: Path
    equity_path: Path
    drawdown_path: Path
    weights_path: Path
    summary_plot_path: Path
    equity_plot_path: Path
    drawdown_plot_path: Path
    weights_plot_path: Path


@dataclass(slots=True)
class BacktestReportWriter:
    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    def write(self, run_name: str, result: BacktestResult) -> BacktestReportPaths:
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        spec_path = run_dir / "spec.json"
        metrics_path = run_dir / "performance.json"
        returns_path = run_dir / "returns.parquet"
        equity_path = run_dir / "equity.parquet"
        drawdown_path = run_dir / "drawdown.parquet"
        weights_path = run_dir / "weights.parquet"
        summary_plot_path = run_dir / "summary.png"
        equity_plot_path = run_dir / "equity_curve.png"
        drawdown_plot_path = run_dir / "drawdown.png"
        weights_plot_path = run_dir / "weights.png"

        equity_asset, bond_asset = INVESTABLE_ASSETS
        spec = {
            "run_name": run_name,
            "strategy_assets": list(INVESTABLE_ASSETS),
            "benchmark_name": f"75% {equity_asset} / 25% {bond_asset}",
            "benchmark_weights": result.benchmark_weights.to_dict(),
            "strategy_weight_bounds": {asset: [0.0, 1.0] for asset in INVESTABLE_ASSETS},
            "rebalance": "weekly W-FRI model weight, forward-filled daily and applied to next-day returns",
            "walk_forward": {
                "train_window": result.metrics.get("train_window"),
                "test_window": result.metrics.get("test_window"),
                "purge_window": result.metrics.get("purge_window"),
                "oos_start": result.metrics.get("oos_start"),
                "oos_end": result.metrics.get("oos_end"),
            },
            "transaction_cost_bps": result.metrics["transaction_cost_bps"],
            "transaction_cost_method": "daily turnover multiplied by transaction_cost_bps; initial position entry is included",
            "return_timing": "weights at date t are multiplied by asset returns from t to t+1",
            "performance_frequency": "daily",
        }
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        pd.Series(result.metrics).to_json(metrics_path, indent=2)
        pd.concat(
            [
                result.returns.rename("strategy_net"),
                result.gross_returns.rename("strategy_gross"),
                result.benchmark_returns.rename("benchmark_net"),
            ],
            axis=1,
        ).to_parquet(returns_path, engine="pyarrow")
        pd.concat(
            [
                result.equity.rename("strategy_net"),
                result.gross_equity.rename("strategy_gross"),
                result.benchmark_equity.rename("benchmark_net"),
            ],
            axis=1,
        ).to_parquet(equity_path, engine="pyarrow")
        pd.concat(
            [
                result.drawdown.rename("strategy_net"),
                result.gross_drawdown.rename("strategy_gross"),
                result.benchmark_drawdown.rename("benchmark_net"),
            ],
            axis=1,
        ).to_parquet(drawdown_path, engine="pyarrow")
        result.weights.to_parquet(weights_path, engine="pyarrow")
        plot_equity(
            pd.concat(
                [
                    result.equity.rename("strategy_net"),
                    result.gross_equity.rename("strategy_gross"),
                    result.benchmark_equity.rename("benchmark_net"),
                ],
                axis=1,
            ),
            equity_plot_path,
        )
        plot_drawdown(
            pd.concat(
                [
                    result.drawdown.rename("strategy_net"),
                    result.gross_drawdown.rename("strategy_gross"),
                    result.benchmark_drawdown.rename("benchmark_net"),
                ],
                axis=1,
            ),
            drawdown_plot_path,
        )
        plot_weights(result.weights, weights_plot_path)
        plot_summary(
            equity=pd.concat(
                [
                    result.equity.rename("strategy_net"),
                    result.gross_equity.rename("strategy_gross"),
                    result.benchmark_equity.rename("benchmark_net"),
                ],
                axis=1,
            ),
            drawdown=pd.concat(
                [
                    result.drawdown.rename("strategy_net"),
                    result.gross_drawdown.rename("strategy_gross"),
                    result.benchmark_drawdown.rename("benchmark_net"),
                ],
                axis=1,
            ),
            weights=result.weights,
            path=summary_plot_path,
        )

        return BacktestReportPaths(
            spec_path=spec_path,
            metrics_path=metrics_path,
            returns_path=returns_path,
            equity_path=equity_path,
            drawdown_path=drawdown_path,
            weights_path=weights_path,
            summary_plot_path=summary_plot_path,
            equity_plot_path=equity_plot_path,
            drawdown_plot_path=drawdown_plot_path,
            weights_plot_path=weights_plot_path,
        )
