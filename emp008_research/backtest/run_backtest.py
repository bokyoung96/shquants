from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .engine import BacktestEngine
from .report import BacktestReport, build_active_weight_outputs, write_backtest_outputs
from .spec import BacktestSpec


def run_backtest(spec: BacktestSpec) -> BacktestReport:
    weights = read_target_weights_csv(spec.weights_csv)
    filtered_weights = filter_dates(weights, start=spec.start, end=spec.end)
    if filtered_weights.empty:
        raise ValueError("no target weight dates remain after applying start/end filters")

    close = load_frame(spec.frame_path(spec.close_filename))
    open_ = load_frame(spec.frame_path(spec.open_filename)) if spec.fill_mode == "next_open" else None
    tradable = load_optional_bool_frame(spec.frame_path(spec.tradable_filename))
    exit_tradable = load_optional_bool_frame(spec.frame_path(spec.exit_tradable_filename))

    start = pd.Timestamp(spec.start) if spec.start else filtered_weights.index.min()
    end = pd.Timestamp(spec.end) if spec.end else filtered_weights.index.max()
    close = close.loc[(close.index >= start) & (close.index <= end)]
    if close.empty:
        raise ValueError("close price frame is empty for requested date range")
    if open_ is not None:
        open_ = open_.reindex(close.index)
    if tradable is not None:
        tradable = tradable.reindex(close.index)
    if exit_tradable is not None:
        exit_tradable = exit_tradable.reindex(close.index)

    schedule = close.index.to_series().isin(filtered_weights.index)
    schedule.index = close.index

    engine = BacktestEngine(cost=spec.cost)
    result = engine.run(
        close=close,
        open_=open_,
        weights=filtered_weights.reindex(close.index).ffill().fillna(0.0),
        capital=spec.capital,
        tradable=tradable,
        exit_tradable=exit_tradable,
        schedule=schedule,
        fill_mode=spec.fill_mode,
        allow_fractional=spec.allow_fractional,
    )

    benchmark_weights = load_optional_frame(spec.frame_path(spec.benchmark_weights_filename))
    if benchmark_weights is not None:
        benchmark_weights = filter_dates(benchmark_weights, start=spec.start, end=spec.end)
        benchmark_weights = benchmark_weights.reindex(filtered_weights.index).fillna(0.0)
    active_weights, active_share = build_active_weight_outputs(filtered_weights, benchmark_weights)

    return write_backtest_outputs(
        config=spec,
        result=result,
        output_dir=spec.output_dir,
        active_weights=active_weights,
        active_share=active_share,
    )


def read_target_weights_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing weights CSV: {path}")
    frame = pd.read_csv(path, index_col=0)
    if frame.empty and len(frame.columns) == 0:
        raise ValueError(f"target weights CSV has no symbol columns: {path}")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.fillna(0.0).astype(float)
    return frame


def load_frame(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"missing parquet frame: {path}")
    frame = pd.read_parquet(path)
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(f"expected a flat ticker-column frame: {path}")
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def load_optional_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return load_frame(path)


def load_optional_bool_frame(path: Path | None) -> pd.DataFrame | None:
    frame = load_optional_frame(path)
    if frame is None:
        return None
    return frame.astype(bool)


def filter_dates(frame: pd.DataFrame, *, start: str | None, end: str | None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    mask = pd.Series(True, index=frame.index)
    if start_ts is not None:
        mask &= frame.index >= start_ts
    if end_ts is not None:
        mask &= frame.index <= end_ts
    return frame.loc[mask]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone EMP008 target-weight backtest.")
    parser.add_argument("--name", default="emp008_research")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument("--fill-mode", choices=("close", "next_open"), default="close")
    parser.add_argument("--fee", type=float, default=0.0)
    parser.add_argument("--sell-tax", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--borrow-fee-annual", type=float, default=0.0)
    parser.add_argument("--short-cash-collateral-ratio", type=float, default=1.0)
    parser.add_argument("--no-fractional", action="store_true")
    parser.add_argument("--close-file", default="qw_adj_c.parquet")
    parser.add_argument("--open-file", default="qw_adj_o.parquet")
    parser.add_argument("--tradable-file")
    parser.add_argument("--exit-tradable-file")
    parser.add_argument("--benchmark-weights-file", default="qw_bm_weights.parquet")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = run_backtest(
        BacktestSpec(
            name=args.name,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            weights_csv=args.weights_csv,
            start=args.start,
            end=args.end,
            capital=args.capital,
            fill_mode=args.fill_mode,
            allow_fractional=not args.no_fractional,
            close_filename=args.close_file,
            open_filename=args.open_file,
            tradable_filename=args.tradable_file,
            exit_tradable_filename=args.exit_tradable_file,
            benchmark_weights_filename=args.benchmark_weights_file,
            fee=args.fee,
            sell_tax=args.sell_tax,
            slippage=args.slippage,
            borrow_fee_annual=args.borrow_fee_annual,
            short_cash_collateral_ratio=args.short_cash_collateral_ratio,
        )
    )
    print(pd.Series(report.summary).to_json(force_ascii=False))


if __name__ == "__main__":
    main()
