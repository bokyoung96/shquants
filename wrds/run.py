from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from client import Client
from data import FlowRegistry, Pipeline, Registry
from download import Downloader


def main() -> None:
    args = parse_args()
    handlers = command_handlers()
    if args.command == "us" and args.us_command in {"at", "latest"}:
        handlers[args.command](None, args)
        return

    with Client(args.config) as wrds:
        handlers[args.command](wrds, args)


Command = Callable[[object, argparse.Namespace], None]


def command_handlers() -> dict[str, Command]:
    return {
        "check": check,
        "query": query,
        "table": table,
        "universe": workflow,
        "us": workflow,
        "options": workflow,
        "data": data,
    }


def check(wrds, args) -> None:
    result = wrds.query("select 1 as ok")
    print(result.to_string(index=False))


def query(wrds, args) -> None:
    path = Downloader(wrds).query(args.sql, args.output)
    print(path)


def table(wrds, args) -> None:
    path = Downloader(wrds).table(args.name, args.output, limit=args.limit)
    print(path)


def workflow(wrds, args) -> None:
    FlowRegistry.default().get(args.command).run(wrds, args)


def data(wrds, args) -> None:
    tables = split_csv(args.tables)
    plan = Registry.default().plan(args.selections, tables=tables)
    Pipeline(wrds).save(
        plan,
        output=args.output,
        limit=args.limit,
        chunksize=args.chunksize,
        retries=args.retries,
        overwrite=args.overwrite,
    )


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
    universe.add_argument("--output", default="wrds/output/datas/universe", type=Path)
    universe.add_argument("--limit", type=int)

    us = commands.add_parser("us", help="Build US stock universes with CRSP and FactSet mappings.")
    us_commands = us.add_subparsers(dest="us_command", required=True)

    us_current = us_commands.add_parser("current", help="Build active US universe for one date.")
    us_current.add_argument("--date", default="latest")
    us_current.add_argument("--output", default="wrds/output/datas/us/current", type=Path)
    us_current.add_argument("--limit", type=int)

    us_history = us_commands.add_parser("history", help="Build historical US universe and latest representative rows.")
    us_history.add_argument("--output", default="wrds/output/datas/us/history", type=Path)
    us_history.add_argument("--limit", type=int)

    us_at = us_commands.add_parser("at", help="Build one date's membership from a historical universe file.")
    us_at.add_argument("date")
    us_at.add_argument("--history", default="wrds/output/datas/us/history/history.csv", type=Path)
    us_at.add_argument("--output", default="wrds/output/datas/us/at.csv", type=Path)

    us_latest = us_commands.add_parser("latest", help="Build latest representative rows from a historical universe file.")
    us_latest.add_argument("--history", default="wrds/output/datas/us/history/history.csv", type=Path)
    us_latest.add_argument("--output", default="wrds/output/datas/us/history/latest.csv", type=Path)

    options = commands.add_parser("options", help="Download OptionMetrics raw data.")
    options_commands = options.add_subparsers(dest="options_command", required=True)

    option_raw = options_commands.add_parser("raw", help="Download raw OptionMetrics tables for one date.")
    option_raw.add_argument("date")
    option_raw.add_argument("--output", default="wrds/output/datas/options/raw", type=Path)
    option_raw.add_argument("--limit", type=int, default=1000)

    data = commands.add_parser("data", help="Download selected high-value WRDS source data.")
    data.add_argument(
        "selections",
        nargs="+",
        help="Rank numbers or library names, for example: 1 2 4 12 or crsp comp ibes crsp_a_indexes.",
    )
    data.add_argument("--output", default="wrds/output/datas", type=Path)
    data.add_argument("--limit", type=int, help="Optional row limit for tests/samples. Omit for full history.")
    data.add_argument("--chunksize", type=int, default=500_000, help="Rows per WRDS chunk for full downloads.")
    data.add_argument("--retries", type=int, default=2, help="Reconnect and retry each table/partition on failure.")
    data.add_argument("--tables", help="Optional comma-separated table filter across selected libraries.")
    data.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files.")
    return parser.parse_args()


def split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
