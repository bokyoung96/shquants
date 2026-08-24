from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.convert import convert_required


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EMP008 raw CSV/XLSX inputs to parquet.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    args = parser.parse_args()
    for dataset, path in convert_required(raw_dir=args.raw_dir, parquet_dir=args.parquet_dir).items():
        print(f"{dataset}: {path}")


if __name__ == "__main__":
    main()
