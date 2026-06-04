from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from ...downloads.batch import BatchCsvWriter, OutputFile
except ImportError:  # pragma: no cover - direct script compatibility
    from downloads.batch import BatchCsvWriter, OutputFile

from .service import US


class USWorkflow:
    name = "us"

    def run(self, client, args) -> None:
        us = US(client)
        if args.us_command == "current":
            us.save_current(date=args.date, output=args.output, limit=args.limit)
        elif args.us_command == "history":
            us.save_history(output=args.output, limit=args.limit)
        elif args.us_command == "at":
            us.save_at(date=args.date, output=args.output, history_path=args.history)
        elif args.us_command == "latest":
            history = us.clean(pd.read_csv(args.history))
            latest = us.latest_rows(history)
            output = Path(args.output)
            BatchCsvWriter().write(output.parent, (OutputFile("latest", output.name, latest),))
            print(f"latest={len(latest)} {args.output}")
        else:
            raise ValueError(f"unknown US workflow command: {args.us_command}")
