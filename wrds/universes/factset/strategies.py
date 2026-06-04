from __future__ import annotations

from typing import Protocol

import pandas as pd


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


class BuildStrategy(Protocol):
    def build(self, links: pd.DataFrame) -> pd.DataFrame:
        ...


class LatestLinkStrategy:
    name = "latest"

    def build(self, links: pd.DataFrame) -> pd.DataFrame:
        if links.empty:
            return pd.DataFrame(columns=UNIVERSE_COLS)
        frame = links.sort_values(["permno", "link_bdate"], ascending=[True, False])
        frame = frame.drop_duplicates("permno", keep="first")
        return frame.loc[:, [col for col in UNIVERSE_COLS if col in frame.columns]].reset_index(drop=True)

