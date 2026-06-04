from __future__ import annotations

from datetime import date

from backtesting.data import SINCE, Source, Table


def sources() -> tuple[Source, ...]:
    end = date.today().year
    return (
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
    )

