from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from client import Client
from download import Downloader
from options import Options
from us import US
from universe import Universe


def main() -> None:
    args = parse_args()
    if args.command == "us" and args.us_command in {"at", "latest"}:
        us = US(None)
        if args.us_command == "at":
            us.save_at(date=args.date, output=args.output, history_path=args.history)
        else:
            history = us.clean(pd.read_csv(args.history))
            latest = us.latest_rows(history)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            latest.to_csv(args.output, index=False)
            print(f"latest={len(latest)} {args.output}")
        return

    with Client(args.config) as wrds:
        if args.command == "check":
            result = wrds.query("select 1 as ok")
            print(result.to_string(index=False))
        elif args.command == "query":
            path = Downloader(wrds).query(args.sql, args.output)
            print(path)
        elif args.command == "table":
            path = Downloader(wrds).table(args.name, args.output, limit=args.limit)
            print(path)
        elif args.command == "universe":
            builder = Universe(wrds)
            links = builder.links(date=args.date, limit=args.limit)
            universe = builder.build(links)
            args.output.mkdir(parents=True, exist_ok=True)
            links.to_csv(args.output / "fscrsplink.csv", index=False)
            universe.to_csv(args.output / "universe.csv", index=False)
            print(f"fscrsplink={len(links)} {args.output / 'fscrsplink.csv'}")
            print(f"universe={len(universe)} {args.output / 'universe.csv'}")
        elif args.command == "us":
            us = US(wrds)
            if args.us_command == "current":
                us.save_current(date=args.date, output=args.output, limit=args.limit)
            elif args.us_command == "history":
                us.save_history(output=args.output, limit=args.limit)
            elif args.us_command == "at":
                us.save_at(date=args.date, output=args.output, history_path=args.history)
            elif args.us_command == "latest":
                history = us.clean(pd.read_csv(args.history))
                latest = us.latest_rows(history)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                latest.to_csv(args.output, index=False)
                print(f"latest={len(latest)} {args.output}")
        elif args.command == "options":
            options = Options(wrds)
            if args.options_command == "raw":
                options.save_raw(date=args.date, output=args.output, limit=args.limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WRDS login, query, and download helper.")
    parser.add_argument("--config", default="wrds/config.json")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check", help="Log in and run select 1.")

    query = commands.add_parser("query", help="Run SQL and save CSV.")
    query.add_argument("sql")
    query.add_argument("--output", default="wrds/output/query.csv")

    table = commands.add_parser("table", help="Download a table with an optional limit.")
    table.add_argument("name", help="library.table, for example wrdsapps.fscrsplink")
    table.add_argument("--output", default="wrds/output/table.csv")
    table.add_argument("--limit", type=int)

    universe = commands.add_parser("universe", help="Build CRSP-FactSet PERMNO universe.")
    universe.add_argument("--date", default="latest")
    universe.add_argument("--output", default="wrds/output", type=Path)
    universe.add_argument("--limit", type=int)

    us = commands.add_parser("us", help="Build US stock universes with CRSP and FactSet mappings.")
    us_commands = us.add_subparsers(dest="us_command", required=True)

    us_current = us_commands.add_parser("current", help="Build active US universe for one date.")
    us_current.add_argument("--date", default="latest")
    us_current.add_argument("--output", default="wrds/output/us/current", type=Path)
    us_current.add_argument("--limit", type=int)

    us_history = us_commands.add_parser("history", help="Build historical US universe and latest representative rows.")
    us_history.add_argument("--output", default="wrds/output/us/history", type=Path)
    us_history.add_argument("--limit", type=int)

    us_at = us_commands.add_parser("at", help="Build one date's membership from a historical universe file.")
    us_at.add_argument("date")
    us_at.add_argument("--history", default="wrds/output/us/history/history.csv", type=Path)
    us_at.add_argument("--output", default="wrds/output/us/at.csv", type=Path)

    us_latest = us_commands.add_parser("latest", help="Build latest representative rows from a historical universe file.")
    us_latest.add_argument("--history", default="wrds/output/us/history/history.csv", type=Path)
    us_latest.add_argument("--output", default="wrds/output/us/history/latest.csv", type=Path)

    options = commands.add_parser("options", help="Download OptionMetrics raw data.")
    options_commands = options.add_subparsers(dest="options_command", required=True)

    option_raw = options_commands.add_parser("raw", help="Download raw OptionMetrics tables for one date.")
    option_raw.add_argument("date")
    option_raw.add_argument("--output", default="wrds/output/options/raw", type=Path)
    option_raw.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
