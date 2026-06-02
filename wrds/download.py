from __future__ import annotations

from pathlib import Path


class Downloader:
    def __init__(self, client) -> None:
        self.client = client

    def query(self, sql: str, output: str | Path) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.query(sql).to_csv(path, index=False)
        return path

    def table(self, name: str, output: str | Path, *, limit: int | None = None) -> Path:
        sql = f"select * from {name}"
        if limit is not None:
            sql += f" limit {int(limit)}"
        return self.query(sql, output)
