from __future__ import annotations

import tomllib
from pathlib import Path


def test_api_package_data_ships_readme_and_config_example():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    package_data = data["tool"]["setuptools"]["package-data"]

    assert "README.md" in package_data["api"]
    assert "config.example.json" in package_data["api"]
