from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from . import dirs


@dataclass(frozen=True)
class WebSnapshot:
    url: str
    html_path: Path
    text_path: Path
    screenshot_path: Path | None


class PlaywrightWebCapturer:
    def __init__(self, *, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def capture(self, *, message_id: str, title: str, url: str, index: int) -> WebSnapshot:
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("Playwright is not installed. Install playwright and browser binaries first.") from exc

        target_dir = dirs.ensure(self.output_root, message_id=message_id, title=title) / "web"
        target_dir.mkdir(parents=True, exist_ok=True)
        html_path = target_dir / f"page-{index}.html"
        text_path = target_dir / f"page-{index}.txt"
        screenshot_path = target_dir / f"page-{index}.png"

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=15000)
            if _needs_redirect_wait(url):
                page.wait_for_timeout(7000)
                page.wait_for_load_state("networkidle", timeout=15000)
            html_path.write_text(page.content())
            text_path.write_text(page.locator("body").inner_text())
            try:
                page.screenshot(path=str(screenshot_path), full_page=True)
            except Exception:  # pragma: no cover - screenshot is best effort
                screenshot_path = None
            browser.close()

        return WebSnapshot(
            url=url,
            html_path=html_path,
            text_path=text_path,
            screenshot_path=screenshot_path,
        )


def _needs_redirect_wait(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in {"lrl.kr"}
