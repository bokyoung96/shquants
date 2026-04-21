from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from backtesting.reporting.analytics import ROLLING_WINDOW, SECTOR_CONTRIBUTION_METHOD_WEIGHTED_ASSET_RETURNS
from backtesting.reporting.benchmarks import default_repositories_for_universe
from backtesting.reporting.models import BenchmarkConfig, SavedRun
from backtesting.reporting.reader import RunReader
from backtesting.reporting.snapshots import PerformanceSnapshot, PerformanceSnapshotFactory
from backtesting.universe import UniverseRegistry
from dashboard.backend.schemas import (
    BenchmarkModel,
    DashboardContextModel,
    DashboardExposureModel,
    DashboardLaunchModel,
    DashboardMetricModel,
    DashboardPayloadModel,
    DashboardPerformanceModel,
    DashboardResearchModel,
    DashboardRollingModel,
    LaunchBenchmarkContextModel,
    LaunchStrategyBenchmarkModel,
    ResearchFocusModel,
    RollingSeriesModel,
)
from dashboard.backend.serializers import (
    serialize_category_series,
    serialize_distribution,
    serialize_drawdown_episodes,
    serialize_heatmap,
    serialize_latest_holdings,
    serialize_latest_holdings_performance,
    serialize_named_series,
    serialize_named_values,
    serialize_value_points,
)
from dashboard.backend.services.run_index import RunIndexService
from dashboard.strategies import DEFAULT_LAUNCH_CONFIG, enabled_strategy_presets
from root import ROOT


class DashboardPayloadService:
    def __init__(
        self,
        runs_root: Path | None = None,
        *,
        run_index_service: RunIndexService | None = None,
        run_reader: RunReader | None = None,
        snapshot_factory: PerformanceSnapshotFactory | None = None,
    ) -> None:
        self.runs_root = runs_root or (ROOT.results_path / "backtests")
        self.run_index_service = run_index_service or RunIndexService(self.runs_root)
        self.run_reader = run_reader or RunReader()
        self.snapshot_factory = snapshot_factory
        self.benchmark = BenchmarkConfig.default_kospi200()

    def build(self, run_ids: list[str]) -> DashboardPayloadModel:
        selected_runs = [self._read_run(run_id) for run_id in run_ids]
        snapshots = [self._snapshot_factory_for_run(run).build(run, self._resolve_benchmark(run)) for run in selected_runs]

        return DashboardPayloadModel(
            mode="single" if len(run_ids) == 1 else "multi",
            selected_run_ids=run_ids,
            available_runs=self.run_index_service.list_runs(),
            launch=self._serialize_launch(snapshots),
            metrics={snapshot.run_id: self._serialize_metrics(snapshot) for snapshot in snapshots},
            context={snapshot.run_id: self._serialize_context(snapshot) for snapshot in snapshots},
            performance=DashboardPerformanceModel(
                series=[self._serialize_series(snapshot, snapshot.strategy_equity) for snapshot in snapshots],
                benchmark=serialize_value_points(snapshots[0].benchmark_equity) if len(snapshots) == 1 else None,
                benchmarks=[self._serialize_benchmark(snapshot) for snapshot in snapshots],
                drawdowns=[self._serialize_series(snapshot, snapshot.drawdowns.underwater) for snapshot in snapshots],
            ),
            rolling=DashboardRollingModel(
                rolling_sharpe=self._serialize_rolling_series(snapshots, "rolling_sharpe"),
                rolling_beta=self._serialize_rolling_series(snapshots, "rolling_beta"),
                rolling_correlation=self._serialize_rolling_correlation(snapshots),
            ),
            exposure=DashboardExposureModel(
                holdings_count=[self._serialize_series(snapshot, snapshot.exposure.holdings_count) for snapshot in snapshots],
                latest_holdings={
                    snapshot.run_id: serialize_latest_holdings(snapshot.exposure.latest_holdings) for snapshot in snapshots
                },
                latest_holdings_winners={
                    snapshot.run_id: serialize_latest_holdings_performance(snapshot.exposure.latest_holdings_winners)
                    for snapshot in snapshots
                },
                latest_holdings_losers={
                    snapshot.run_id: serialize_latest_holdings_performance(snapshot.exposure.latest_holdings_losers)
                    for snapshot in snapshots
                },
                sector_weights={
                    snapshot.run_id: serialize_named_values(snapshot.sectors.latest_weighted) for snapshot in snapshots
                },
            ),
            research=self._serialize_research(snapshots),
        )

    @staticmethod
    def _serialize_launch(snapshots: list[PerformanceSnapshot]) -> DashboardLaunchModel:
        config = DEFAULT_LAUNCH_CONFIG.global_config
        as_of_date = None
        if snapshots:
            as_of_date = max(snapshot.strategy_equity.index.max() for snapshot in snapshots).date().isoformat()
        return DashboardLaunchModel(
            configured_start_date=config.start,
            configured_end_date=config.end,
            capital=config.capital,
            schedule=config.schedule,
            fill_mode=config.fill_mode,
            fee=config.fee,
            sell_tax=config.sell_tax,
            slippage=config.slippage,
            benchmark=DashboardPayloadService._serialize_launch_benchmark_context(),
            as_of_date=as_of_date,
        )

    @staticmethod
    def _serialize_launch_benchmark_context() -> LaunchBenchmarkContextModel:
        presets = sorted(
            enabled_strategy_presets(DEFAULT_LAUNCH_CONFIG.strategies),
            key=lambda preset: (preset.strategy_name, preset.display_label),
        )
        if not presets:
            default = BenchmarkConfig.default_kospi200()
            return LaunchBenchmarkContextModel(
                kind="shared",
                shared=BenchmarkModel(code=default.code, name=default.name),
                strategies=[],
            )

        strategies = [
            LaunchStrategyBenchmarkModel(
                strategy=preset.strategy_name,
                label=preset.display_label,
                benchmark=BenchmarkModel(code=preset.benchmark.code, name=preset.benchmark.name),
            )
            for preset in presets
        ]
        unique_benchmarks = {(entry.benchmark.code, entry.benchmark.name) for entry in strategies}
        shared = strategies[0].benchmark if len(unique_benchmarks) == 1 else None
        return LaunchBenchmarkContextModel(
            kind="shared" if shared is not None else "strategy-specific",
            shared=shared,
            strategies=strategies,
        )

    def _read_run(self, run_id: str) -> SavedRun:
        run_dir = self.runs_root / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        try:
            return self.run_reader.read(run_dir)
        except OSError as exc:
            raise HTTPException(status_code=404, detail=f"unable to read run_id: {run_id}") from exc

    def _serialize_context(self, snapshot: PerformanceSnapshot) -> DashboardContextModel:
        return DashboardContextModel(
            label=snapshot.display_name,
            strategy=snapshot.strategy_name,
            benchmark=BenchmarkModel(code=snapshot.benchmark.code, name=snapshot.benchmark.name),
            start_date=snapshot.strategy_equity.index.min().date().isoformat(),
            end_date=snapshot.strategy_equity.index.max().date().isoformat(),
            as_of_date=snapshot.strategy_equity.index.max().date().isoformat(),
        )

    @staticmethod
    def _serialize_metrics(snapshot: PerformanceSnapshot) -> DashboardMetricModel:
        metrics = snapshot.metrics
        return DashboardMetricModel(
            label=snapshot.display_name,
            cumulative_return=metrics.cumulative_return,
            cagr=metrics.cagr,
            annual_volatility=metrics.annual_volatility,
            sharpe=metrics.sharpe,
            sortino=metrics.sortino,
            calmar=metrics.calmar,
            max_drawdown=metrics.max_drawdown,
            final_equity=metrics.final_equity,
            avg_turnover=metrics.avg_turnover,
            alpha=metrics.alpha,
            beta=metrics.beta,
            tracking_error=metrics.tracking_error,
            information_ratio=metrics.information_ratio,
        )

    @staticmethod
    def _serialize_series(snapshot: PerformanceSnapshot, series: object) -> object:
        return serialize_named_series(
            series,
            run_id=snapshot.run_id,
            label=snapshot.display_name,
        )

    @staticmethod
    def _serialize_benchmark(snapshot: PerformanceSnapshot) -> object:
        return serialize_named_series(
            snapshot.benchmark_equity,
            run_id=snapshot.run_id,
            label=snapshot.benchmark.name,
        )

    @staticmethod
    def _serialize_rolling_series(snapshots: list[PerformanceSnapshot], name: str) -> list[object]:
        return [
            serialize_named_series(
                snapshot.rolling.series[name],
                run_id=snapshot.run_id,
                label=snapshot.display_name,
            )
            for snapshot in snapshots
            if not snapshot.rolling.series[name].dropna().empty
        ]

    @staticmethod
    def _serialize_rolling_correlation(snapshots: list[PerformanceSnapshot]) -> list[RollingSeriesModel]:
        return [
            RollingSeriesModel(
                run_id=snapshot.run_id,
                label=snapshot.display_name,
                benchmark=BenchmarkModel(code=snapshot.benchmark.code, name=snapshot.benchmark.name),
                window=snapshot.rolling.window if snapshot.rolling.window else ROLLING_WINDOW,
                points=serialize_value_points(snapshot.rolling.series["rolling_correlation"]),
            )
            for snapshot in snapshots
            if not snapshot.rolling.series["rolling_correlation"].dropna().empty
        ]

    @staticmethod
    def _serialize_research(snapshots: list[PerformanceSnapshot]) -> DashboardResearchModel:
        return DashboardResearchModel(
            focus=ResearchFocusModel(kind="all-selected", label="All Selected", value=None),
            sector_contribution_method=SECTOR_CONTRIBUTION_METHOD_WEIGHTED_ASSET_RETURNS,
            monthly_heatmap={
                snapshot.run_id: serialize_heatmap(snapshot.research.monthly_heatmap) for snapshot in snapshots
            },
            return_distribution={
                snapshot.run_id: serialize_distribution(snapshot.research.return_distribution) for snapshot in snapshots
            },
            monthly_return_distribution={
                snapshot.run_id: serialize_distribution(snapshot.research.monthly_return_distribution)
                for snapshot in snapshots
            },
            yearly_excess_returns={
                snapshot.run_id: serialize_named_series(
                    snapshot.research.yearly_excess_returns,
                    run_id=snapshot.run_id,
                    label=snapshot.display_name,
                ).points
                for snapshot in snapshots
            },
            sector_contribution_series={
                snapshot.run_id: serialize_category_series(snapshot.research.sector_contribution) for snapshot in snapshots
            },
            sector_weight_series={
                snapshot.run_id: serialize_category_series(snapshot.research.sector_weights) for snapshot in snapshots
            },
            drawdown_episodes={
                snapshot.run_id: serialize_drawdown_episodes(snapshot.research.drawdown_episodes) for snapshot in snapshots
            },
        )

    def _snapshot_factory_for_run(self, run: SavedRun) -> PerformanceSnapshotFactory:
        if self.snapshot_factory is not None:
            return self.snapshot_factory
        universe_id = run.config.get("universe_id")
        benchmark_repo, sector_repo = default_repositories_for_universe(
            str(universe_id) if universe_id is not None else None
        )
        return PerformanceSnapshotFactory(
            benchmark_repo=benchmark_repo,
            sector_repo=sector_repo,
        )

    def _resolve_benchmark(self, run: SavedRun) -> BenchmarkConfig:
        raw = run.config.get("benchmark")
        if isinstance(raw, dict):
            code = str(raw.get("code") or self.benchmark.code)
            name = str(raw.get("name") or code or self.benchmark.name)
            dataset = str(raw.get("dataset") or self.benchmark.dataset)
            return BenchmarkConfig(code=code, name=name, dataset=dataset)

        code = run.config.get("benchmark_code")
        name = run.config.get("benchmark_name")
        dataset = run.config.get("benchmark_dataset")
        if code is None and name is None and dataset is None:
            universe_id = run.config.get("universe_id")
            if universe_id is not None:
                try:
                    universe_spec = UniverseRegistry.default().get(str(universe_id))
                    return BenchmarkConfig(
                        code=universe_spec.default_benchmark_code,
                        name=universe_spec.default_benchmark_name,
                        dataset=universe_spec.default_benchmark_dataset,
                    )
                except KeyError:
                    pass
            return self.benchmark

        resolved_code = str(code or self.benchmark.code)
        resolved_name = str(name or resolved_code or self.benchmark.name)
        resolved_dataset = str(dataset or self.benchmark.dataset)
        return BenchmarkConfig(code=resolved_code, name=resolved_name, dataset=resolved_dataset)
