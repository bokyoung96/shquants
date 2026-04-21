from __future__ import annotations

import importlib
from pathlib import Path

__all__ = ("PdfRenderer",)

_PRINT_LAYOUT_OVERRIDE = (
    "<style>@media print {"
    ".report-section,.compact-table-block,.notes-banner{break-inside:auto;page-break-inside:auto;}"
    "}</style>"
)


class PdfRenderer:
    def render(self, html_path: Path) -> Path | None:
        pdf_path, _ = self.render_with_status(html_path)
        return pdf_path

    def render_with_status(self, html_path: Path) -> tuple[Path | None, dict[str, object]]:
        html_path = Path(html_path)
        try:
            weasyprint = importlib.import_module("weasyprint")
        except ImportError as exc:
            return None, {"pdf_ok": False, "pdf_error": f"ImportError: {exc}"}

        pdf_path = html_path.with_suffix(".pdf")
        try:
            html = html_path.read_text(encoding="utf-8")
            html = html.replace("</head>", f"{_PRINT_LAYOUT_OVERRIDE}</head>", 1)
            weasyprint.HTML(string=html, base_url=str(html_path.parent)).write_pdf(str(pdf_path))
        except Exception as exc:
            return None, {"pdf_ok": False, "pdf_error": f"{type(exc).__name__}: {exc}"}
        return pdf_path, {"pdf_ok": True, "pdf_path": str(pdf_path)}
