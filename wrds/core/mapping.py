from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class Columns:
    dates: tuple[str, ...] = ()
    ints: tuple[str, ...] = ()


class Cleaner:
    def __init__(self, columns: Columns) -> None:
        self.columns = columns

    def frame(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col in self.columns.dates:
            if col in data:
                data[col] = pd.to_datetime(data[col]).dt.normalize()
        for col in self.columns.ints:
            if col in data:
                data[col] = data[col].astype("Int64")
        return data


class Linker:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def current(self, left: pd.DataFrame, right: pd.DataFrame, *, key: str, left_start: str, left_end: str, right_start: str, right_end: str) -> pd.DataFrame:
        left = left.sort_values([key, left_start], ascending=[True, False]).drop_duplicates(key, keep="first")
        right = right.sort_values([key, right_start], ascending=[True, False]).drop_duplicates(key, keep="first")
        frame = left.merge(right.drop(columns=["permco"], errors="ignore"), on=key, how="left")
        frame["market"] = frame["exchange"]
        frame["start_date"] = frame[left_start]
        frame.loc[frame[right_start].notna(), "start_date"] = frame.loc[frame[right_start].notna(), [left_start, right_start]].max(axis=1)
        frame["end_date"] = frame[left_end]
        frame.loc[frame[right_end].notna(), "end_date"] = frame.loc[frame[right_end].notna(), [left_end, right_end]].min(axis=1)
        return self._select(frame)

    def history(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        key: str,
        left_start: str,
        left_end: str,
        right_start: str,
        right_end: str,
        desc: str,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        right_by_key = {value: frame for value, frame in right.groupby(key, sort=False)}
        for _, left_row in tqdm(left.iterrows(), total=len(left), desc=desc, unit="row"):
            links = right_by_key.get(left_row[key])
            if links is None or links.empty:
                rows.append(self._row(left_row, None, left_start, left_end, right_start, right_end))
                continue
            matched = links[
                (links[right_start].fillna(pd.Timestamp.min) <= left_row[left_end])
                & (links[right_end].fillna(pd.Timestamp.max) >= left_row[left_start])
            ]
            if matched.empty:
                rows.append(self._row(left_row, None, left_start, left_end, right_start, right_end))
                continue
            for _, right_row in matched.iterrows():
                rows.append(self._row(left_row, right_row, left_start, left_end, right_start, right_end))
        if not rows:
            return pd.DataFrame(columns=self.columns)
        return pd.concat(rows, ignore_index=True).sort_values([key, "start_date", "end_date"]).reset_index(drop=True)

    def at(self, history: pd.DataFrame, date: str, *, active: bool = True) -> pd.DataFrame:
        date_value = pd.Timestamp(date).normalize()
        frame = history[(history["start_date"] <= date_value) & (history["end_date"] >= date_value)]
        if active and "tradingstatusflg" in frame:
            frame = frame[frame["tradingstatusflg"].eq("A")]
        if active and "conditionaltype" in frame:
            frame = frame[frame["conditionaltype"].eq("RW")]
        frame = frame.sort_values(["permno", "start_date"], ascending=[True, False]).drop_duplicates("permno", keep="first")
        return frame.reset_index(drop=True)

    def latest(self, history: pd.DataFrame) -> pd.DataFrame:
        if history.empty:
            return history
        frame = history.sort_values(["permno", "end_date", "start_date"], ascending=[True, False, False])
        return frame.drop_duplicates("permno", keep="first").reset_index(drop=True)

    def _row(self, left: pd.Series, right: pd.Series | None, left_start: str, left_end: str, right_start: str, right_end: str) -> pd.DataFrame:
        data = left.to_dict()
        if right is None:
            data.update(
                {
                    "factset_ticker": pd.NA,
                    "ticker_exchange": pd.NA,
                    "fsym_regional_id": pd.NA,
                    "fsym_security_id": pd.NA,
                    "factset_entity_id": pd.NA,
                    right_start: pd.NaT,
                    right_end: pd.NaT,
                    "start_date": left[left_start],
                    "end_date": left[left_end],
                }
            )
        else:
            data.update(right.drop(labels=["permco"], errors="ignore").to_dict())
            data["start_date"] = max(left[left_start], right[right_start]) if pd.notna(right[right_start]) else left[left_start]
            data["end_date"] = min(left[left_end], right[right_end]) if pd.notna(right[right_end]) else left[left_end]
        data["market"] = data.get("exchange")
        return pd.DataFrame([{col: data.get(col, pd.NA) for col in self.columns}])

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[:, [col for col in self.columns if col in frame.columns]].reset_index(drop=True)
