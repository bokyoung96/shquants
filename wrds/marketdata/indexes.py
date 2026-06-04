from __future__ import annotations

from datetime import date

from backtesting.data import SINCE, Source, Table


def sources() -> tuple[Source, ...]:
    end = date.today().year
    return (
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
    )

