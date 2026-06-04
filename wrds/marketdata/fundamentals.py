from __future__ import annotations

from datetime import date

from backtesting.data import SINCE, Source, Table


def sources() -> tuple[Source, ...]:
    end = date.today().year
    return (
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
    )

