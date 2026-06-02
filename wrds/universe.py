from __future__ import annotations

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


class Universe:
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
        if limit is not None:
            sql += f" limit {int(limit)}"
        return self.clean(self.client.query(sql))

    def build(self, links: pd.DataFrame) -> pd.DataFrame:
        if links.empty:
            return pd.DataFrame(columns=UNIVERSE_COLS)
        frame = links.sort_values(["permno", "link_bdate"], ascending=[True, False])
        frame = frame.drop_duplicates("permno", keep="first")
        return frame.loc[:, [col for col in UNIVERSE_COLS if col in frame.columns]].reset_index(drop=True)

    @staticmethod
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for col in ("link_bdate", "link_edate"):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col]).dt.normalize()
        for col in ("permno", "permco"):
            if col in frame:
                frame[col] = frame[col].astype("Int64")
        return frame
