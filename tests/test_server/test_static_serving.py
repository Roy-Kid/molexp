"""Smoke tests for bundled frontend static asset serving."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import _find_bundled_webapp, create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_webapp(tmp_path: Path) -> Path:
    """Minimal directory that mimics the rsbuild build output."""
    root = tmp_path / "webapp"
    root.mkdir()
    (root / "index.html").write_text(
        "<!DOCTYPE html><html><body>MolExp UI</body></html>"
    )
    (root / "favicon.ico").write_bytes(b"\x00")  # dummy favicon

    static = root / "static" / "js"
    static.mkdir(parents=True)
    (static / "main.abc123.js").write_text("console.log('molexp');")

    css = root / "static" / "css"
    css.mkdir(parents=True)
    (css / "main.abc123.css").write_text("body{margin:0}")

    return root


# ---------------------------------------------------------------------------
# API-only mode (dev / no bundle)
# ---------------------------------------------------------------------------


class TestApiOnlyMode:
    def test_root_returns_service_info(self) -> None:
        app = create_app(serve_static=False)
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert data["service"] == "molexp"

    def test_health_endpoint(self) -> None:
        app = create_app(serve_static=False)
        with TestClient(app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Production mode with bundled (mock) webapp
# ---------------------------------------------------------------------------


class TestStaticServing:
    def test_index_served_at_root(self, mock_webapp: Path) -> None:
        app = create_app(static_dir=mock_webapp)
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert "MolExp UI" in resp.text

    def test_spa_fallback_for_client_routes(self, mock_webapp: Path) -> None:
        app = create_app(static_dir=mock_webapp)
        with TestClient(app) as client:
            resp = client.get("/projects/abc/experiments/123")
            assert resp.status_code == 200
            assert "MolExp UI" in resp.text

    def test_static_js_served(self, mock_webapp: Path) -> None:
        app = create_app(static_dir=mock_webapp)
        with TestClient(app) as client:
            resp = client.get("/static/js/main.abc123.js")
            assert resp.status_code == 200
            assert "molexp" in resp.text

    def test_static_css_served(self, mock_webapp: Path) -> None:
        app = create_app(static_dir=mock_webapp)
        with TestClient(app) as client:
            resp = client.get("/static/css/main.abc123.css")
            assert resp.status_code == 200

    def test_root_file_served_directly(self, mock_webapp: Path) -> None:
        """Files that exist at the webapp root (favicon etc.) are served as-is."""
        app = create_app(static_dir=mock_webapp)
        with TestClient(app) as client:
            resp = client.get("/favicon.ico")
            assert resp.status_code == 200

    def test_api_takes_priority(self, mock_webapp: Path) -> None:
        """API routes are not shadowed by the SPA fallback."""
        app = create_app(static_dir=mock_webapp)
        with TestClient(app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Bundle detection
# ---------------------------------------------------------------------------


class TestBundleDetection:
    def test_returns_none_when_no_webapp(self) -> None:
        result = _find_bundled_webapp()
        # In a dev checkout without a built wheel this is None.
        # In CI after a wheel build it might be a real path.
        assert result is None or (result.is_dir() and (result / "index.html").exists())
