from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_STARTER = "C:/SHINHAN-i/indi/GiExpertStarter.exe"


@dataclass(frozen=True)
class IndiConfig:
    user: str = ""
    password: str = ""
    cert_password: str = ""
    account: str = ""
    account_password: str = ""
    starter: str = DEFAULT_STARTER


def load_config(path: str | Path = DEFAULT_CONFIG) -> IndiConfig:
    config_path = Path(path)
    if not config_path.exists():
        return IndiConfig()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")

    return IndiConfig(
        user=_text(data, "user"),
        password=_text(data, "password"),
        cert_password=_text(data, "cert_password"),
        account=_text(data, "account"),
        account_password=_text(data, "account_password"),
        starter=_text(data, "starter") or DEFAULT_STARTER,
    )


def _text(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    return "" if value is None else str(value)
