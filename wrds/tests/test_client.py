from __future__ import annotations

import sys
from pathlib import Path

from wrds.client import Client, load_wrds_library


def test_client_loads_login_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"id": "user", "pwd": "secret"}')

    client = Client(config)
    client.login()

    assert client.user == "user"
    assert client.password == "secret"


def test_client_loads_external_wrds_library_when_local_package_exists(tmp_path: Path, monkeypatch) -> None:
    site = tmp_path / "site"
    site.mkdir()
    external = site / "wrds.py"
    external.write_text("class Connection: pass\n", encoding="utf-8")
    import wrds as local_wrds

    monkeypatch.setitem(sys.modules, "wrds", local_wrds)
    monkeypatch.syspath_prepend(str(site))

    module = load_wrds_library()

    assert Path(module.__file__).resolve() == external.resolve()
    assert hasattr(module, "Connection")
