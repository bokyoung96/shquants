from __future__ import annotations

from typing import Iterable, Protocol

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
UNIVERSE_COLS = (
    "permno",
    "permco",
    "ticker",
    "issuernm",
    "fsym_regional_id",
    "fsym_security_id",
    "factset_entity_id",
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


class BuildStrategy(Protocol):
    def build(self, links: pd.DataFrame) -> pd.DataFrame:
        ...


class Named(Protocol):
    name: str


class UniverseRegistry:
    def __init__(self, items: Iterable[Named]) -> None:
        self._items = {item.name: item for item in items}

    @classmethod
    def default(cls, client) -> "UniverseRegistry":
        return cls((FactSetSource(client), LatestLinkStrategy()))

    def get(self, name: str):
        try:
            return self._items[name]
        except KeyError as exc:
            raise ValueError(f"unknown universe component: {name}") from exc


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
        sql = _limit(sql, limit)
        return self.clean(self.client.query(sql))

    @staticmethod
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        return clean(frame)


class LatestLinkStrategy:
    name = "latest"

    def build(self, links: pd.DataFrame) -> pd.DataFrame:
        if links.empty:
            return pd.DataFrame(columns=UNIVERSE_COLS)
        frame = links.sort_values(["permno", "link_bdate"], ascending=[True, False])
        frame = frame.drop_duplicates("permno", keep="first")
        return frame.loc[:, [col for col in UNIVERSE_COLS if col in frame.columns]].reset_index(drop=True)


class Universe:
    def __init__(
        self,
        client=None,
        *,
        source: LinkSource | None = None,
        strategy: BuildStrategy | None = None,
    ) -> None:
        registry = UniverseRegistry.default(client) if source is None or strategy is None else None
        self.source = source or registry.get("links")
        self.strategy = strategy or registry.get("latest")

    @classmethod
    def from_registry(cls, registry: UniverseRegistry) -> "Universe":
        return cls(source=registry.get("links"), strategy=registry.get("latest"))

    def latest(self) -> str:
        return self.source.latest()

    def links(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        return self.source.links(date=date, limit=limit)

    def build(self, links: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.build(links)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ("link_bdate", "link_edate"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col]).dt.normalize()
    for col in ("permno", "permco"):
        if col in frame:
            frame[col] = frame[col].astype("Int64")
    return frame
