from __future__ import annotations

from fastapi.testclient import TestClient

from crypto.dashboard.app import create_app


def test_crypto_dashboard_serves_preview_and_html() -> None:
    client = TestClient(create_app())

    preview = client.get("/api/preview")
    assert preview.status_code == 200
    payload = preview.json()
    assert payload["summary"]["candidate_pool_size"] == 40
    assert payload["summary"]["selected_basket_size"] == 10
    assert len(payload["registry"]) == 10

    homepage = client.get("/")
    assert homepage.status_code == 200
    assert "Crypto Factory Dashboard" in homepage.text
    assert "/api/preview" in homepage.text
