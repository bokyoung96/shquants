from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.reporting.cli import ReportCli
from backtesting.run import BacktestRunner
from backtesting.strategies.emp008.attribution import build_emp008_factor_attribution
from backtesting.strategies.emp008.comparison import build_emp008_comparison
from backtesting.strategies.emp008.run_backtest import active_share_payload, backtest_summary, build_target_weight_spec, resolve_run_output_dirs
from backtesting.strategies.emp008.run_weights import (
    DEFAULT_START,
    build_emp008_config,
    configure_logging,
    latest_common_end,
    timed,
    timestamp,
    weights_summary,
    write_target_weights_csv,
)
from backtesting.strategies.emp008.mfbt_emp008 import run_mfbt_emp008


DEFAULT_NAME = "mfbt_emp008"


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = build_emp008_config(
        tracking_error_annual=args.tracking_error_annual,
        risk_model=args.risk_model,
        factor_set=args.factor_set,
        sector_neutral_dataset=args.sector_neutral_dataset,
    )
    end = args.end or latest_common_end(args.parquet_dir, config)
    run_root, backtests_root, reports_root = resolve_run_output_dirs(
        output_root=args.output_root,
        name=args.name,
        backtests_root=args.backtests_root,
        reports_root=args.reports_root,
    )
    weights_dir = run_root / "weights"
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or logs_dir / f"run_{timestamp()}.log"
    logger = configure_logging(log_path, logger_name="mfbt_emp008_full")

    logger.info("MFBT EMP008 full run started")
    logger.info("start=%s end=%s parquet_dir=%s", args.start, end, args.parquet_dir)
    logger.info(
        "tracking_error_monthly=%s tracking_error_annual=%s risk_model=%s factor_set=%s",
        config.tracking_error,
        args.tracking_error_annual,
        config.risk_model,
        config.factor_set,
    )
    logger.info("output_root=%s backtests_root=%s reports_root=%s", args.output_root, backtests_root, reports_root)
    logger.info("log_file=%s", log_path)

    summary: dict[str, Any] = {
        "name": args.name,
        "start": args.start,
        "end": end,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_path),
        "tracking_error_monthly": config.tracking_error,
        "tracking_error_annual": args.tracking_error_annual,
        "risk_model": config.risk_model,
        "factor_set": config.factor_set,
        "sector_neutral_dataset": None if config.sector_neutral_dataset is None else config.sector_neutral_dataset.value,
    }

    try:
        with timed(logger, "weights"):
            emp008_result = run_mfbt_emp008(
                parquet_dir=args.parquet_dir,
                start=args.start,
                end=end,
                config=config,
                output_dir=weights_dir,
            )
            weights_csv = write_target_weights_csv(emp008_result.target_weights, weights_dir / "target_weights.csv")
            summary["weights"] = weights_summary(emp008_result, weights_csv)
            summary["active_share"] = active_share_payload(weights_csv)
            logger.info("weights_csv=%s", weights_csv)
            logger.info("weights_summary=%s", json.dumps(summary["weights"], ensure_ascii=False))
            logger.info("active_share=%s", json.dumps(summary["active_share"], ensure_ascii=False))

        with timed(logger, "backtest"):
            dates = tuple(pd.to_datetime(emp008_result.target_weights.index).strftime("%Y-%m-%d"))
            spec = build_target_weight_spec(
                name=args.name,
                weights_csv=weights_csv,
                dates=dates,
                end=end,
                fill_mode=args.fill_mode,
                capital=args.capital,
                fee=args.fee,
                sell_tax=args.sell_tax,
                slippage=args.slippage,
                allow_fractional=not args.no_fractional,
            )
            runner = BacktestRunner(
                result_dir=backtests_root,
                write_report_assets=False,
                profile=True,
            )
            backtest_report = runner.run_spec(runner.resolve_spec(spec))
            summary["backtest"] = backtest_summary(backtest_report)
            summary["active_share"] = active_share_payload(
                weights_csv,
                backtest_output_dir=Path(str(backtest_report.output_dir)),
            )
            logger.info("backtest_summary=%s", json.dumps(summary["backtest"], ensure_ascii=False))
            logger.info("active_share=%s", json.dumps(summary["active_share"], ensure_ascii=False))

        with timed(logger, "report"):
            run_id = Path(str(backtest_report.output_dir)).name
            report_payload = ReportCli(
                runs_root=backtests_root,
                reports_root=reports_root,
            ).run(
                [
                    "--runs",
                    run_id,
                    "--name",
                    args.name,
                    "--kind",
                    "tearsheet",
                    "--title",
                    args.title or args.name,
                ]
            )
            summary["report"] = report_payload
            logger.info("report_payload=%s", json.dumps(report_payload, ensure_ascii=False))

        if not args.no_comparison:
            with timed(logger, "costed_backtest"):
                costed_spec = build_target_weight_spec(
                    name=f"{args.name}_costed",
                    weights_csv=weights_csv,
                    dates=dates,
                    end=end,
                    fill_mode=args.fill_mode,
                    capital=args.capital,
                    fee=args.comparison_fee,
                    sell_tax=args.comparison_sell_tax,
                    slippage=args.comparison_slippage,
                    allow_fractional=not args.no_fractional,
                )
                costed_runner = BacktestRunner(
                    result_dir=backtests_root,
                    write_report_assets=False,
                    profile=True,
                )
                costed_report = costed_runner.run_spec(costed_runner.resolve_spec(costed_spec))
                summary["costed_backtest"] = backtest_summary(costed_report)
                summary["costed_active_share"] = active_share_payload(
                    weights_csv,
                    backtest_output_dir=Path(str(costed_report.output_dir)),
                )
                logger.info("costed_backtest_summary=%s", json.dumps(summary["costed_backtest"], ensure_ascii=False))

            with timed(logger, "comparison"):
                comparison_payload = build_emp008_comparison(
                    gross_run_dir=Path(str(backtest_report.output_dir)),
                    costed_run_dir=Path(str(costed_report.output_dir)),
                    active_weights_parquet=weights_dir / "active_weights.parquet",
                    benchmark_parquet=args.benchmark_parquet,
                    output_dir=run_root / "comparison",
                )
                summary["comparison"] = comparison_payload
                logger.info("comparison=%s", json.dumps(comparison_payload, ensure_ascii=False))

        if not args.no_factor_attribution:
            with timed(logger, "factor_attribution"):
                attribution_payload = build_emp008_factor_attribution(
                    parquet_dir=args.parquet_dir,
                    run_root=run_root,
                    config=config,
                )
                summary["factor_attribution"] = attribution_payload
                logger.info("factor_attribution=%s", json.dumps(attribution_payload, ensure_ascii=False))

        summary_path = run_root / "run_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("summary_path=%s", summary_path)
        logger.info("MFBT EMP008 full run completed")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        logger.exception("MFBT EMP008 full run failed")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MFBT EMP008 weights, backtest, and report with logs.")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Requested output start date. Default: {DEFAULT_START}")
    parser.add_argument("--end", help="Requested end date. Default: min latest date across required parquet datasets.")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"Run/report name. Default: {DEFAULT_NAME}")
    parser.add_argument("--title", help="Report title. Default: run name.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("results") / "emp008_runs")
    parser.add_argument("--backtests-root", type=Path, help="Backtest output root. Default: <output-root>/<name>/backtests.")
    parser.add_argument("--reports-root", type=Path, help="Report output root. Default: <output-root>/<name>/reports.")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--fill-mode", choices=("close", "next_open"), default="close")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument("--fee", type=float, default=0.0)
    parser.add_argument("--sell-tax", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--no-fractional", action="store_true")
    parser.add_argument("--tracking-error-annual", type=float, help="Annual tracking error budget, e.g. 0.03.")
    parser.add_argument(
        "--risk-model",
        choices=("factor_idio", "direct_covariance"),
        help="Risk matrix used in TE constraint. Default: factor_idio.",
    )
    parser.add_argument(
        "--factor-set",
        choices=("mfbt", "mfbt_pos", "origin"),
        help="Alpha factor set. Use 'mfbt_pos' to replace price momentum with positivity, or 'origin' for LnMktcap, Momentum_12M, DY.",
    )
    parser.add_argument(
        "--sector-neutral-dataset",
        choices=("default", "wi26", "wics"),
        help="Sector taxonomy for optimizer neutrality. Default keeps WI26; wics uses QW_WICS_SEC_BIG.",
    )
    parser.add_argument("--no-comparison", action="store_true", help="Skip costed backtest and comparison artifacts.")
    parser.add_argument("--comparison-fee", type=float, default=0.0002)
    parser.add_argument("--comparison-sell-tax", type=float, default=0.0015)
    parser.add_argument("--comparison-slippage", type=float, default=0.0005)
    parser.add_argument("--benchmark-parquet", type=Path, default=Path("parquet") / "qw_BM.parquet")
    parser.add_argument("--no-factor-attribution", action="store_true", help="Skip factor attribution artifacts.")
    return parser


if __name__ == "__main__":
    main()
