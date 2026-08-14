"""API-only ``GET /`` is an HTML page, not a JSON stub."""

from __future__ import annotations

from fastapi.testclient import TestClient

from molexp.server.app import _API_ONLY_PAGE, create_app


def test_api_only_root_is_html() -> None:
    client = TestClient(create_app(serve_static=False))
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "molexp API only" in response.text
    assert "/api/docs" in response.text
    assert "npm run build:web" in response.text
    assert response.text == _API_ONLY_PAGE
