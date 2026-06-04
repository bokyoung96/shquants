from __future__ import annotations

try:
    from ...downloads.batch import BatchCsvWriter, OutputFile
except ImportError:  # pragma: no cover - direct script compatibility
    from downloads.batch import BatchCsvWriter, OutputFile

from .service import Universe


class UniverseWorkflow:
    name = "universe"

    def run(self, client, args) -> None:
        builder = Universe(client)
        links = builder.links(date=args.date, limit=args.limit)
        universe = builder.build(links)
        BatchCsvWriter().write(
            args.output,
            (
                OutputFile("fscrsplink", "fscrsplink.csv", links),
                OutputFile("universe", "universe.csv", universe),
            ),
        )
        print(f"fscrsplink={len(links)} {args.output / 'fscrsplink.csv'}")
        print(f"universe={len(universe)} {args.output / 'universe.csv'}")
