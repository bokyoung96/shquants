from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd


def load_wrds_library() -> ModuleType:
    existing = sys.modules.get("wrds")
    if existing is not None and hasattr(existing, "Connection"):
        return existing

    local_root = Path(__file__).resolve().parent
    project_root = local_root.parent
    sentinel = object()
    restored = sentinel
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None and _under(Path(existing_file).resolve(), local_root):
            restored = existing
            del sys.modules["wrds"]

    original_path = list(sys.path)
    try:
        sys.path = [entry for entry in sys.path if not _local_import_path(entry, local_root, project_root)]
        module = import_module("wrds")
        if not hasattr(module, "Connection"):
            raise ImportError("installed WRDS library does not expose Connection")
        return module
    finally:
        sys.path = original_path
        if restored is not sentinel:
            sys.modules["wrds"] = restored


def _local_import_path(entry: str, local_root: Path, project_root: Path) -> bool:
    resolved = Path(entry or ".").resolve()
    return resolved in {local_root, project_root}


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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
        wrds = load_wrds_library()
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
        try:
            from .download import Downloader
        except ImportError:  # pragma: no cover - direct script compatibility
            from download import Downloader

        return Downloader(self).query(sql, output)

    def table(self, name: str, output: str | Path, *, limit: int | None = None) -> Path:
        try:
            from .download import Downloader
        except ImportError:  # pragma: no cover - direct script compatibility
            from download import Downloader

        return Downloader(self).table(name, output, limit=limit)

    def latest(self) -> str:
        try:
            from .universes.factset.service import Universe
        except ImportError:  # pragma: no cover - direct script compatibility
            from universes.factset.service import Universe

        return Universe(self).latest()

    def links(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        try:
            from .universes.factset.service import Universe
        except ImportError:  # pragma: no cover - direct script compatibility
            from universes.factset.service import Universe

        return Universe(self).links(date=date, limit=limit)

    def universe(self, links: pd.DataFrame) -> pd.DataFrame:
        try:
            from .universes.factset.service import Universe
        except ImportError:  # pragma: no cover - direct script compatibility
            from universes.factset.service import Universe

        return Universe(self).build(links)
