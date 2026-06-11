from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from backtesting.catalog import DataCatalog
from backtesting.strategies.emp008.mfbt_emp008 import MfbtEmp008Result, run_mfbt_emp008
from backtesting.strategies.emp008.mfbt_emp008_data import MfbtEmp008Config, required_datasets


DEFAULT_START = "2020-01-31"
DEFAULT_NAME = "mfbt_emp008"


def build_emp008_config(*, tracking_error_annual: float | None = None, risk_model: str | None = None) -> MfbtEmp008Config:
    config = MfbtEmp008Config()
    if risk_model is not None:
        if risk_model not in {"factor_idio", "direct_covariance"}:
            raise ValueError("risk_model must be 'factor_idio' or 'direct_covariance'")
        config = replace(config, risk_model=risk_model)
    if tracking_error_annual is None:
        return config
    if tracking_error_annual < 0.0:
        raise ValueError("tracking error must be non-negative")
    return replace(config, tracking_error=tracking_error_annual / (12**0.5))


def latest_common_end(parquet_dir: Path, config: MfbtEmp008Config) -> str:
    catalog = DataCatalog.default()
    ends: list[pd.Timestamp] = []
    for dataset_id in required_datasets(config):
        spec = catalog.get(dataset_id)
        path = parquet_dir / f"{spec.stem}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing required parquet dataset: {path}")
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError(f"empty required parquet dataset: {path}")
        ends.append(pd.to_datetime(frame.index).max())
    return min(ends).date().isoformat()


def write_target_weights_csv(weights: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = weights.copy()
    output.index = pd.to_datetime(output.index).strftime("%Y-%m-%d")
    output.to_csv(path)
    return path


def configure_logging(log_path: Path, *, logger_name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


class timed:
    def __init__(self, logger: logging.Logger, name: str) -> None:
        self.logger = logger
        self.name = name
        self.started_at = 0.0

    def __enter__(self) -> None:
        self.started_at = perf_counter()
        self.logger.info("stage_start=%s", self.name)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        elapsed = perf_counter() - self.started_at
        if exc_type is None:
            self.logger.info("stage_done=%s elapsed_seconds=%.3f", self.name, elapsed)
        else:
            self.logger.error("stage_failed=%s elapsed_seconds=%.3f", self.name, elapsed)


def weights_summary(result: MfbtEmp008Result, weights_csv: Path) -> dict[str, object]:
    weights = result.target_weights
    diagnostics = result.diagnostics
    return {
        "target_weights_parquet": str(weights_csv.with_suffix(".parquet")),
        "target_weights_csv": str(weights_csv),
        "active_weights_parquet": str(weights_csv.parent / "active_weights.parquet"),
        "diagnostics_parquet": str(weights_csv.parent / "diagnostics.parquet"),
        "weights_export_xlsx": str(weights_csv.parent / "weights_export.xlsx"),
        "shape": [int(weights.shape[0]), int(weights.shape[1])],
        "date_start": pd.to_datetime(weights.index).min().date().isoformat() if not weights.empty else None,
        "date_end": pd.to_datetime(weights.index).max().date().isoformat() if not weights.empty else None,
        "row_sum_min": float(weights.sum(axis=1).min()) if not weights.empty else None,
        "row_sum_max": float(weights.sum(axis=1).max()) if not weights.empty else None,
        "diagnostics_rows": int(len(diagnostics)),
        "diagnostics_success_all": bool(diagnostics["success"].all()) if not diagnostics.empty else False,
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = build_emp008_config(tracking_error_annual=args.tracking_error_annual, risk_model=args.risk_model)
    end = args.end or latest_common_end(args.parquet_dir, config)
    run_root = args.output_root / args.name
    weights_dir = run_root / "weights"
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or logs_dir / f"weights_{timestamp()}.log"
    logger = configure_logging(log_path, logger_name="mfbt_emp008_weights")

    logger.info("MFBT EMP008 weights run started")
    logger.info("start=%s end=%s parquet_dir=%s", args.start, end, args.parquet_dir)
    logger.info(
        "tracking_error_monthly=%s tracking_error_annual=%s risk_model=%s",
        config.tracking_error,
        args.tracking_error_annual,
        config.risk_model,
    )
    logger.info("weights_dir=%s log_file=%s", weights_dir, log_path)

    summary = {
        "name": args.name,
        "start": args.start,
        "end": end,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_path),
        "tracking_error_monthly": config.tracking_error,
        "tracking_error_annual": args.tracking_error_annual,
        "risk_model": config.risk_model,
    }
    try:
        with timed(logger, "weights"):
            result = run_mfbt_emp008(
                parquet_dir=args.parquet_dir,
                start=args.start,
                end=end,
                config=config,
                output_dir=weights_dir,
            )
            weights_csv = write_target_weights_csv(result.target_weights, weights_dir / "target_weights.csv")
            summary["weights"] = weights_summary(result, weights_csv)
            logger.info("weights_summary=%s", json.dumps(summary["weights"], ensure_ascii=False))

        summary_path = run_root / "weights_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("summary_path=%s", summary_path)
        logger.info("MFBT EMP008 weights run completed")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        logger.exception("MFBT EMP008 weights run failed")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MFBT EMP008 target weights only.")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Requested output start date. Default: {DEFAULT_START}")
    parser.add_argument("--end", help="Requested end date. Default: min latest date across required parquet datasets.")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"Weights run name. Default: {DEFAULT_NAME}")
    parser.add_argument("--parquet-dir", type=Path, default=Path("parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("results") / "emp008_runs")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--tracking-error-annual", type=float, help="Annual tracking error budget, e.g. 0.03.")
    parser.add_argument(
        "--risk-model",
        choices=("factor_idio", "direct_covariance"),
        help="Risk matrix used in TE constraint. Default: factor_idio.",
    )
    return parser


if __name__ == "__main__":
    main()
