from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backtesting.strategies.emp008.mfbt_emp008_factor_pipeline import load_and_prepare_emp008_factors
from backtesting.strategies.emp008.mfbt_emp008_factor_quantiles import run_emp008_factor_quantiles
from backtesting.strategies.emp008.mfbt_emp008_factor_registry import factor_set_values
from backtesting.strategies.emp008.run_weights import DEFAULT_START, build_emp008_config, latest_common_end


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = build_emp008_config(
        factor_set=args.factor_set,
        sector_neutral_dataset=args.sector_neutral_dataset,
    )
    end = args.end or latest_common_end(args.parquet_dir, config)
    prepared = load_and_prepare_emp008_factors(
        parquet_dir=args.parquet_dir,
        start=args.start,
        end=end,
        config=config,
    )
    result = run_emp008_factor_quantiles(
        prepared=prepared,
        start=args.start,
        end=end,
        q=args.quantiles,
    )
    payload = result.write_outputs(args.output_dir, factor_set=config.factor_set, q=args.quantiles)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_cli(argv: list[str] | None = None) -> int:
    try:
        main(argv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone EMP008 factor quantile evaluation with equal-weight and "
            "total-market-cap-weight portfolio modes."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START, help=f"Requested output start date. Default: {DEFAULT_START}")
    parser.add_argument("--end", help="Requested end date. Default: min latest date across required parquet datasets.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "emp008_factor_quantiles",
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--factor-set",
        choices=factor_set_values(),
        help="Registered EMP008 factor set variant.",
    )
    parser.add_argument(
        "--sector-neutral-dataset",
        choices=("default", "wi26", "wics"),
        help="Sector taxonomy for optimizer-neutralized factor preparation. Default keeps WI26; wics uses QW_WICS_SEC_BIG.",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=5,
        help="Number of quantile buckets to evaluate. Default: 5.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run_cli())
