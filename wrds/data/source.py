from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable


SINCE = 2015


@dataclass(frozen=True)
class Table:
    name: str
    about: str = ""
    date: str | None = None
    start: int = SINCE
    end: int | None = None

    @property
    def file(self) -> str:
        return f"{self.name}.csv"

    def sql(self, source: str, *, limit: int | None) -> list[str]:
        if self.date is None or limit is not None:
            return [self._select(source, limit=limit)]
        return [sql for _, sql in self.parts(source)]

    def parts(self, source: str) -> list[tuple[int, str]]:
        if self.date is None:
            return []
        end = self.end or date.today().year
        return [(year, self._select(source, where=self._year(year), limit=None)) for year in range(self.start, end + 1)]

    def split(self, *, limit: int | None) -> bool:
        return self.date is not None and limit is None

    def _year(self, year: int) -> str:
        return f"{self.date} >= '{year}-01-01' and {self.date} < '{year + 1}-01-01'"

    def _select(self, source: str, *, where: str | None = None, limit: int | None) -> str:
        sql = f"select * from {source}.{self.name}"
        if where:
            sql += f" where {where}"
        if limit is not None:
            sql += f" limit {int(limit)}"
        return sql


@dataclass(frozen=True)
class Source:
    rank: int
    name: str
    about: str
    tables: tuple[Table, ...]

    @property
    def description(self) -> str:
        return self.about

    def withtables(self, names: Iterable[str] | None) -> "Source":
        if names is None:
            return self
        selected = set(names)
        return Source(
            rank=self.rank,
            name=self.name,
            about=self.about,
            tables=tuple(table for table in self.tables if table.name in selected),
        )

    def with_tables(self, names: Iterable[str] | None) -> "Source":
        return self.withtables(names)


@dataclass(frozen=True)
class Plan:
    sources: tuple[Source, ...]

    @property
    def libraries(self) -> tuple[Source, ...]:
        return self.sources

    @property
    def table_count(self) -> int:
        return sum(len(source.tables) for source in self.sources)


DataTableSpec = Table
DataLibrarySpec = Source
DataDownloadPlan = Plan
