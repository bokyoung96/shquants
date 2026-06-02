from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pandas as pd


class Client:
    def __init__(self, config: str | Path = "wrds/config.json") -> None:
        self.config = Path(config)
        self.user = ""
        self.password = ""
        self.db = None

    def __enter__(self) -> "Client":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def login(self) -> None:
        data = json.loads(self.config.read_text())
        self.user = str(data.get("id") or data.get("username") or "")
        self.password = str(data.get("pwd") or data.get("password") or "")
        if not self.user or not self.password:
            raise ValueError("config must include id and pwd")

    def connect(self) -> None:
        if not self.user or not self.password:
            self.login()
        wrds = import_module("wrds")
        self.db = wrds.Connection(wrds_username=self.user, wrds_password=self.password, verbose=False)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None

    def query(self, sql: str) -> pd.DataFrame:
        if self.db is None:
            self.connect()
        return self.db.raw_sql(sql)

    def download(self, sql: str, output: str | Path) -> Path:
        from download import Downloader

        return Downloader(self).query(sql, output)

    def table(self, name: str, output: str | Path, *, limit: int | None = None) -> Path:
        from download import Downloader

        return Downloader(self).table(name, output, limit=limit)

    def latest(self) -> str:
        from universe import Universe

        return Universe(self).latest()

    def links(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        from universe import Universe

        return Universe(self).links(date=date, limit=limit)

    def universe(self, links: pd.DataFrame) -> pd.DataFrame:
        from universe import Universe

        return Universe(self).build(links)
