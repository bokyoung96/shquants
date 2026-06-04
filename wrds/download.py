from __future__ import annotations

try:
    from .downloads.service import Downloader
except ImportError:  # pragma: no cover - direct script compatibility
    from downloads.service import Downloader

__all__ = ("Downloader",)
