from __future__ import annotations

from datetime import date

from backtesting.data import SINCE, Source, Table


def sources() -> tuple[Source, ...]:
    end = date.today().year
    return (
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
    )

