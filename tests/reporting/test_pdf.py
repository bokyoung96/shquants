from __future__ import annotations

import importlib
from pathlib import Path

from backtesting.reporting.pdf import PdfRenderer


def test_pdf_renderer_keeps_html_when_pdf_export_fails(monkeypatch, tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    html_path.write_text("<html><body>hello</body></html>", encoding="utf-8")

    def _raise_import_error(name: str):
        raise ImportError("weasyprint missing")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)

    renderer = PdfRenderer()
    pdf_path = renderer.render(html_path)
    status_path, status = renderer.render_with_status(html_path)

    assert html_path.exists()
    assert pdf_path is None
    assert status_path is None
    assert status["pdf_ok"] is False
    assert str(status["pdf_error"]).startswith("ImportError:")


def test_pdf_renderer_writes_pdf_when_export_succeeds(monkeypatch, tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    html_path.write_text("<html><body>hello</body></html>", encoding="utf-8")

    seen: dict[str, str] = {}

    class _FakeHtml:
        def __init__(self, **kwargs) -> None:
            seen.update({key: str(value) for key, value in kwargs.items()})

        def write_pdf(self, path: str) -> None:
            Path(path).write_bytes(b"%PDF-1.4")

    class _FakeWeasyPrint:
        HTML = _FakeHtml

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeWeasyPrint())

    pdf_path, status = PdfRenderer().render_with_status(html_path)

    assert pdf_path == html_path.with_suffix(".pdf")
    assert pdf_path.exists()
    assert status == {"pdf_ok": True, "pdf_path": str(pdf_path)}
    assert seen["base_url"] == str(tmp_path)
    assert "hello" in seen["string"]


def test_pdf_renderer_writes_pdf_from_composed_report(monkeypatch, tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    styles_path = tmp_path / "styles.css"
    asset_path = tmp_path / "page.png"
    asset_path.write_bytes(b"png")
    styles_path.write_text(
        """
        .report-cover { page-break-after: avoid; }
        .executive-spread { display: grid; }
        .metric-strip { display: grid; }
        .compact-table-block { break-inside: avoid; }
        """.strip(),
        encoding="utf-8",
    )
    html_path.write_text(
        f"""
        <html>
          <head>
            <link rel="stylesheet" href="styles.css">
          </head>
          <body>
            <main class="report-shell">
              <section class="report-cover cover">
                <div class="hero-main">
                  <h1>Momentum Tearsheet</h1>
                </div>
              </section>
              <section class="report-section executive-spread">
                <div class="metric-strip">
                  <article class="metric-card">
                    <p class="metric-card-label">CAGR</p>
                    <p class="metric-card-value">17.2%</p>
                  </article>
                </div>
                <section class="compact-table-block">
                  <div class="table-wrap">
                    <table>
                      <tr><td>1</td></tr>
                    </table>
                  </div>
                </section>
                <img src="{asset_path.name}" alt="equity">
              </section>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    class _FakeHtml:
        def __init__(self, **kwargs) -> None:
            html = kwargs["string"]
            assert kwargs["base_url"] == str(tmp_path)
            assert '<link rel="stylesheet" href="styles.css">' in html
            assert 'class="report-cover cover"' in html
            assert 'class="report-section executive-spread"' in html
            assert 'class="metric-strip"' in html
            assert 'class="compact-table-block"' in html
            assert "@media print" in html
            assert ".report-section,.compact-table-block,.notes-banner" in html
            assert 'src="page.png"' in html

        def write_pdf(self, path: str) -> None:
            Path(path).write_bytes(b"%PDF-1.4")

    class _FakeWeasyPrint:
        HTML = _FakeHtml

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeWeasyPrint())

    pdf_path, status = PdfRenderer().render_with_status(html_path)

    assert pdf_path == html_path.with_suffix(".pdf")
    assert pdf_path is not None
    assert pdf_path.exists()
    assert status["pdf_ok"] is True


def test_pdf_renderer_injects_print_layout_override_for_composed_reports(monkeypatch, tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    html_path.write_text(
        """
        <html>
          <head>
            <link rel="stylesheet" href="styles.css">
          </head>
          <body>
            <main class="report-shell">
              <section class="report-section">
                <div class="section-stack">
                  <section class="compact-table-block">
                    <div class="table-wrap">
                      <table><tr><td>row</td></tr></table>
                    </div>
                  </section>
                </div>
              </section>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    seen: dict[str, str] = {}

    class _FakeHtml:
        def __init__(self, **kwargs) -> None:
            seen.update({key: str(value) for key, value in kwargs.items()})

        def write_pdf(self, path: str) -> None:
            Path(path).write_bytes(b"%PDF-1.4")

    class _FakeWeasyPrint:
        HTML = _FakeHtml

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeWeasyPrint())

    pdf_path, status = PdfRenderer().render_with_status(html_path)

    assert pdf_path == html_path.with_suffix(".pdf")
    assert pdf_path.exists()
    assert status == {"pdf_ok": True, "pdf_path": str(pdf_path)}
    assert seen["base_url"] == str(tmp_path)
    assert "@media print" in seen["string"]
    assert ".report-section,.compact-table-block,.notes-banner" in seen["string"]
    assert 'href="styles.css"' in seen["string"]
