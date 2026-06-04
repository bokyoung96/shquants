from __future__ import annotations

from typing import Protocol

import pandas as pd


LINK_COLS = (
    "fsym_id",
    "fsym_id_kind",
    "proper_name",
    "fsym_regional_id",
    "fsym_security_id",
    "fs_perm_sec_id",
    "factset_entity_id",
    "entity_proper_name",
    "cusip_fs",
    "ticker_exchange",
    "permno",
    "permco",
    "hdrcusip",
    "cusip",
    "ticker",
    "issuernm",
    "link_bdate",
    "link_edate",
)


def _limit(sql: str, value: int | None) -> str:
    if value is None:
        return sql
    return f"{sql} limit {int(value)}"


class LinkSource(Protocol):
    def latest(self) -> str:
        ...

    def links(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        ...


class FactSetSource:
    name = "links"

    def __init__(self, client) -> None:
        self.client = client

    def latest(self) -> str:
        frame = self.client.query(
            "select max(link_edate) as date "
            "from wrdsapps.fscrsplink "
            "where link_edate is not null"
        )
        if frame.empty or pd.isna(frame.iloc[0]["date"]):
            raise ValueError("wrdsapps.fscrsplink has no link_edate")
        return str(pd.Timestamp(frame.iloc[0]["date"]).date())

    def links(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        if date == "latest":
            date = self.latest()
        sql = (
            f"select {', '.join(LINK_COLS)} "
            "from wrdsapps.fscrsplink "
            f"where link_bdate <= '{date}' "
            f"and (link_edate is null or link_edate >= '{date}') "
            "and fsym_id_kind = 'R' "
            "order by permno, link_bdate"
        )
        return clean(self.client.query(_limit(sql, limit)))

    @staticmethod
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        return clean(frame)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ("link_bdate", "link_edate"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col]).dt.normalize()
    for col in ("permno", "permco"):
        if col in frame:
            frame[col] = frame[col].astype("Int64")
    return frame

