from __future__ import annotations

from backtesting.data import Pipeline

from .catalog import source_registry


class DataWorkflow:
    name = "data"

    def run(self, client, args) -> None:
        tables = [part.strip() for part in args.tables.split(",") if part.strip()] if args.tables else None
        plan = source_registry().plan(args.selections, tables=tables)
        Pipeline(client).save(
            plan,
            output=args.output,
            limit=args.limit,
            chunksize=args.chunksize,
            retries=args.retries,
            overwrite=args.overwrite,
        )

