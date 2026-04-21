from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .cli import run_watch_until

DEFAULT_WATCH_CHANNELS = ["DOC_POOL", "report_figure_by_offset"]


def build_arg_parser(*, default_base_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysts.run_watcher")
    parser.add_argument("--until", required=True)
    parser.add_argument("--channel", action="append")
    parser.add_argument("--base-dir", default=str(default_base_dir))
    return parser


def main(argv: Sequence[str] | None = None, *, default_base_dir: Path | None = None) -> int:
    parser = build_arg_parser(default_base_dir=default_base_dir or Path("analysts"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    channels = args.channel or list(DEFAULT_WATCH_CHANNELS)
    return run_watch_until(base_dir=Path(args.base_dir), channels=channels, until=args.until)
