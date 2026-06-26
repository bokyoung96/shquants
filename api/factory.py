from __future__ import annotations

from collections.abc import Callable
import importlib
from pathlib import Path

from api.client import Indi
from api.config import DEFAULT_CONFIG, load_config


Controls = Callable[[], tuple[object, object]]


def make(path: str | Path = DEFAULT_CONFIG, controls: Controls | None = None) -> Indi:
    config = load_config(path)
    tr, real = controls() if controls is not None else live_controls()
    return Indi(tr, real, config)


def live_controls() -> tuple[object, object]:
    try:
        gi = importlib.import_module("GiExpertControl")
    except ImportError as exc:
        raise RuntimeError(
            "GiExpertControl is not importable. Run this on the Windows iIndi "
            "Python environment after installing Shinhan iIndi."
        ) from exc

    if hasattr(gi, "NewGiExpertModule"):
        return gi, gi.NewGiExpertModule()

    return gi, gi
