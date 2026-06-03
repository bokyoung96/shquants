from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backtesting.data import FlowRegistry, SINCE, Source, SourceRegistry, Table
from options import Options
from universe import Universe
from us import US


def source_registry() -> SourceRegistry:
    end = date.today().year
    return SourceRegistry(
        [
            Source(
                1,
                "crsp",
                "Current CRSP CIZ US equity prices, returns, and identifiers.",
                (
                    Table("stkdlysecuritydata", "Daily CIZ stock prices and returns.", "dlycaldt", SINCE, end),
                    Table("stkmthsecuritydata", "Monthly CIZ stock prices and returns.", "mthcaldt", SINCE, end),
                    Table("stksecurityinfohist", "CIZ security identifier history."),
                    Table("msedelist", "Monthly delisting events.", "dlstdt", SINCE, end),
                    Table("dsedelist", "Daily delisting events.", "dlstdt", SINCE, end),
                    Table("dsi", "Daily market indexes.", "date", SINCE, end),
                ),
            ),
            Source(
                2,
                "comp",
                "Compustat fundamentals and company/security metadata.",
                (
                    Table("funda", "Annual fundamentals.", "datadate", SINCE, end),
                    Table("fundq", "Quarterly fundamentals.", "datadate", SINCE, end),
                    Table("company", "Company metadata."),
                    Table("security", "Security metadata."),
                    Table("g_names", "Global name history."),
                ),
            ),
            Source(
                4,
                "ibes",
                "Analyst estimates, actuals, summaries, and identifier maps.",
                (
                    Table("det_epsus", "US EPS detail estimates.", "fpedats", SINCE, end),
                    Table("statsumu_epsus", "US EPS unadjusted summary estimates.", "statpers", SINCE, end),
                    Table("actu_epsus", "US EPS unadjusted actuals.", "pends", SINCE, end),
                    Table("id", "IBES identifier history."),
                    Table("det_xepsus", "US ex-item EPS detail estimates.", "fpedats", SINCE, end),
                    Table("statsumu_xepsus", "US ex-item EPS summary estimates.", "statpers", SINCE, end),
                    Table("actu_xepsus", "US ex-item EPS actuals.", "pends", SINCE, end),
                ),
            ),
            Source(
                12,
                "crsp_a_indexes",
                "CRSP index, S&P 500, and index portfolio series.",
                (
                    Table("dsix", "Daily index series.", "caldt", SINCE, end),
                    Table("msix", "Monthly index series.", "caldt", SINCE, end),
                    Table("dsp500", "Daily S&P 500 returns.", "caldt", SINCE, end),
                    Table("msp500", "Monthly S&P 500 returns.", "caldt", SINCE, end),
                    Table("dsp500list", "Daily S&P 500 membership/list data."),
                    Table("msp500list", "Monthly S&P 500 membership/list data."),
                    Table("inddlyseriesdata_ind", "Daily index series metadata/data.", "dlycaldt", SINCE, end),
                    Table("indmthseriesdata_ind", "Monthly index series metadata/data.", "mthcaldt", SINCE, end),
                ),
            ),
        ]
    )


def flow_registry() -> FlowRegistry:
    return FlowRegistry((UniverseFlow(), USFlow(), OptionsFlow()))


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
