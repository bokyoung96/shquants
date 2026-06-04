from __future__ import annotations

from pathlib import Path

try:
    from ..core.sql import limit as limit_sql
except ImportError:  # pragma: no cover - direct script compatibility
    from core.sql import limit as limit_sql


class Downloader:
    def __init__(self, client) -> None:
        self.client = client

    def query(self, sql: str, output: str | Path) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.query(sql).to_csv(path, index=False)
        return path

    def table(self, name: str, output: str | Path, *, limit: int | None = None) -> Path:
        return self.query(limit_sql(f"select * from {name}", limit), output)
