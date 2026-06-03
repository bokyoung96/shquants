from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from options import Options
from universe import Universe
from us import US


class Flow(Protocol):
    name: str

    def run(self, client, args) -> None:
        ...


class FlowRegistry:
    def __init__(self, flows: Iterable[Flow]) -> None:
        self._flows = {flow.name: flow for flow in flows}

    @classmethod
    def default(cls) -> "FlowRegistry":
        return cls((UniverseFlow(), USFlow(), OptionsFlow()))

    def get(self, name: str) -> Flow:
        try:
            return self._flows[name]
        except KeyError as exc:
            raise ValueError(f"unknown data workflow: {name}") from exc


class UniverseFlow:
    name = "universe"

    def run(self, client, args) -> None:
        builder = Universe(client)
        links = builder.links(date=args.date, limit=args.limit)
        universe = builder.build(links)
        args.output.mkdir(parents=True, exist_ok=True)
        links.to_csv(args.output / "fscrsplink.csv", index=False)
        universe.to_csv(args.output / "universe.csv", index=False)
        print(f"fscrsplink={len(links)} {args.output / 'fscrsplink.csv'}")
        print(f"universe={len(universe)} {args.output / 'universe.csv'}")


class USFlow:
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
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            latest.to_csv(args.output, index=False)
            print(f"latest={len(latest)} {args.output}")
        else:
            raise ValueError(f"unknown US workflow command: {args.us_command}")


class OptionsFlow:
    name = "options"

    def run(self, client, args) -> None:
        options = Options(client)
        if args.options_command == "raw":
            options.save_raw(date=args.date, output=args.output, limit=args.limit)
            return
        raise ValueError(f"unknown options workflow command: {args.options_command}")
