from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import pandas as pd
from tqdm import tqdm

from .source import Plan, Table
from .writer import Csv, Writer


class Db(Protocol):
    def raw_sql(self, sql: str, **kwargs) -> object:
        ...


class ClientLike(Protocol):
    db: Db | None

    def query(self, sql: str) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class Result:
    rank: int
    library: str
    table: str
    path: Path
    rows: int
    status: str

    @property
    def table_name(self) -> str:
        return self.table.split(".", 1)[1]


class Pipeline:
    def __init__(self, client: ClientLike, *, writer: Writer | None = None) -> None:
        self.client = client
        self.writer = writer or Csv()

    def save(
        self,
        plan: Plan,
        *,
        output: str | Path,
        limit: int | None = None,
        chunksize: int = 500_000,
        retries: int = 2,
        overwrite: bool = False,
    ) -> list[Result]:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        results: list[Result] = []
        steps = tqdm(total=plan.table_count, desc="data", unit="table")
        try:
            for source in plan.sources:
                for table in source.tables:
                    result = self._save(
                        rank=source.rank,
                        library=source.name,
                        table=table,
                        output=output,
                        limit=limit,
                        chunksize=chunksize,
                        retries=retries,
                        overwrite=overwrite,
                    )
                    results.append(result)
                    steps.update()
        finally:
            steps.close()
        self._manifest(output, results)
        self._print(results)
        return results

    def _save(
        self,
        *,
        rank: int,
        library: str,
        table: Table,
        output: Path,
        limit: int | None,
        chunksize: int,
        retries: int,
        overwrite: bool,
    ) -> Result:
        path = output / library / table.file
        if path.exists() and not overwrite and self._has_rows(path):
            return self._result(rank, library, table, path, rows=0, status="skipped")
        if table.split(limit=limit):
            return self._parts(
                rank=rank,
                library=library,
                table=table,
                directory=path.with_suffix(""),
                chunksize=chunksize,
                retries=retries,
                overwrite=overwrite,
            )
        rows = self._write(table.sql(library, limit=limit), path, chunksize=chunksize, retries=retries)
        return self._result(rank, library, table, path, rows=rows, status="saved")

    def _parts(
        self,
        *,
        rank: int,
        library: str,
        table: Table,
        directory: Path,
        chunksize: int,
        retries: int,
        overwrite: bool,
    ) -> Result:
        directory.mkdir(parents=True, exist_ok=True)
        rows = 0
        saved = 0
        skipped = 0
        empty = 0
        for year, sql in table.parts(library):
            path = directory / f"year={year}.csv"
            if path.exists() and not overwrite and self._has_rows(path):
                skipped += 1
                continue
            written = self._write([sql], path, chunksize=chunksize, retries=retries)
            if written == 0 and path.exists():
                path.unlink()
                empty += 1
            rows += written
            if written:
                saved += 1
        status = "saved" if saved else "skipped" if skipped else "empty" if empty else "skipped"
        return self._result(rank, library, table, directory, rows=rows, status=status)

    def _write(self, sql: Iterable[str], path: Path, *, chunksize: int, retries: int) -> int:
        statements = list(sql)
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries + 1):
            try:
                return self.writer.write(self._chunks(statements, chunksize=chunksize), path)
            except Exception:
                if attempt >= retries:
                    raise
                self._reconnect()
        raise RuntimeError("unreachable retry state")

    def _chunks(self, statements: Iterable[str], *, chunksize: int) -> Iterator[pd.DataFrame]:
        for sql in statements:
            yield from self._query(sql, chunksize=chunksize)

    def _query(self, sql: str, *, chunksize: int) -> Iterator[pd.DataFrame]:
        db = getattr(self.client, "db", None)
        if db is not None and hasattr(db, "raw_sql"):
            try:
                result = db.raw_sql(sql, chunksize=chunksize, return_iter=True)
            except TypeError:
                result = db.raw_sql(sql)
            if isinstance(result, pd.DataFrame):
                yield result
            else:
                yield from result
            return
        yield self.client.query(sql)

    def _reconnect(self) -> None:
        close = getattr(self.client, "close", None)
        connect = getattr(self.client, "connect", None)
        if callable(close):
            close()
        if callable(connect):
            connect()

    @staticmethod
    def _result(rank: int, library: str, table: Table, path: Path, *, rows: int, status: str) -> Result:
        return Result(rank, library, f"{library}.{table.name}", path, rows, status)

    @staticmethod
    def _has_rows(path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            next(handle, None)
            return next(handle, None) is not None

    @staticmethod
    def _manifest(output: Path, results: Iterable[Result]) -> None:
        rows = []
        for result in results:
            row = asdict(result)
            row["path"] = result.path.relative_to(output).as_posix()
            row["table"] = result.table_name
            rows.append(row)
        pd.DataFrame(rows, columns=["rank", "library", "table", "path", "rows", "status"]).to_csv(
            output / "manifest.csv",
            index=False,
        )

    @staticmethod
    def _print(results: Iterable[Result]) -> None:
        for result in results:
            print(f"{result.status} {result.table} rows={result.rows} {result.path}")


DataDownloadResult = Result
DataPipeline = Pipeline
