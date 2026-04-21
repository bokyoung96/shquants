from __future__ import annotations

from pathlib import Path
import re

_BAD = re.compile(r'[\\/:*?"<>|]+')
_SPACE = re.compile(r"\s+")


def name(message_id: str, title: str) -> str:
    clean = _clean(title)
    return message_id if not clean else f"{message_id}-{clean}"


def find(root: Path, *, message_id: str, title: str | None = None) -> Path:
    root = Path(root)
    if title:
        exact = root / name(message_id, title)
        if exact.exists():
            return exact
    legacy = root / message_id
    if legacy.exists():
        return legacy
    matches = sorted(root.glob(f"{message_id}*"))
    if matches:
        return matches[0]
    return root / (name(message_id, title or ""))


def ensure(root: Path, *, message_id: str, title: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name(message_id, title)
    if target.exists():
        return target
    legacy = root / message_id
    if legacy.exists():
        legacy.rename(target)
        return target
    target.mkdir(parents=True, exist_ok=True)
    return target


def _clean(text: str) -> str:
    text = _BAD.sub("_", text.strip())
    text = _SPACE.sub(" ", text).strip()
    text = text.replace(" ", "_")
    text = text[:80].strip("._")
    return text
