from __future__ import annotations

import argparse
from pathlib import Path

from .core import RrgConfig, compute_multi_horizon_rrg
from .dashboard import export_multi_horizon_rrg
from .data import load_kospi200_wics_sector_rrg_input


def run_kospi200_wics_example(
    *,
    start: str = "2020-01-02",
    end: str = "2026-03-25",
    output_path: str | Path = "results/rrg/advanced_rrg_3d.html",
) -> Path:
    input_data = load_kospi200_wics_sector_rrg_input(start=start, end=end)
    rrg_frame = compute_multi_horizon_rrg(
        sector_prices=input_data.sector_prices,
        benchmark=input_data.benchmark,
        confidence=input_data.confidence,
        config=RrgConfig(),
    )
    return export_multi_horizon_rrg(rrg_frame, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 3D KOSPI200 WICS RRG HTML file.")
    parser.add_argument("--start", default="2020-01-02")
    parser.add_argument("--end", default="2026-03-25")
    parser.add_argument("--output", default="results/rrg/advanced_rrg_3d.html")
    args = parser.parse_args()
    path = run_kospi200_wics_example(start=args.start, end=args.end, output_path=args.output)
    print(path)


if __name__ == "__main__":
    main()
