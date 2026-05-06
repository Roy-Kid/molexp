"""Tests for the per-plugin static-asset mount after the 07 split.

Each entry-point-discovered UI bundle directory is mounted at
``/api/plugins/<id>/`` (no ``/static`` segment) so:

* ``GET /api/plugins/<id>/manifest.json`` → 200 with bundle's manifest;
* ``GET /api/plugins/<id>/index.js`` → 200 with bundle's entry script;
* the URLs returned by ``GET /api/plugins`` resolve 1:1 to those files;
* unknown plugin ids 404.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app


def _make_bundle(root: Path, plugin_id: str) -> Path:
    bundle = root / plugin_id
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "id": plugin_id,
                "name": f"Test {plugin_id}",
                "version": "0.0.1",
                "api_version": "1",
                "entry": "index.js",
            }
        )
    )
    (bundle / "index.js").write_text(f"export default {{ id: {plugin_id!r}, register() {{}} }};\n")
    return bundle


def _patch_discover(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, Path]) -> None:
    monkeypatch.setattr(
        "molexp.plugins.ui.discover_ui_plugin_dirs",
        lambda: dict(mapping),
    )
    import molexp.server.app as app_mod
    import molexp.server.routes.registry as registry_mod

    if hasattr(app_mod, "discover_ui_plugin_dirs"):
        monkeypatch.setattr(app_mod, "discover_ui_plugin_dirs", lambda: dict(mapping))
    if hasattr(registry_mod, "discover_ui_plugin_dirs"):
        monkeypatch.setattr(registry_mod, "discover_ui_plugin_dirs", lambda: dict(mapping))


# ── ac-007 ─────────────────────────────────────────────────────────────


def test_manifest_json_served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "alpha")
    _patch_discover(monkeypatch, {"alpha": bundle})

    client = TestClient(create_app(serve_static=False))
    resp = client.get("/api/plugins/alpha/manifest.json")
    assert resp.status_code == 200, resp.text

    on_disk = json.loads((bundle / "manifest.json").read_text())
    assert resp.json() == on_disk


def test_index_js_served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "alpha")
    _patch_discover(monkeypatch, {"alpha": bundle})

    client = TestClient(create_app(serve_static=False))
    resp = client.get("/api/plugins/alpha/index.js")
    assert resp.status_code == 200, resp.text
    assert resp.content == (bundle / "index.js").read_bytes()


def test_listed_urls_resolve_to_actual_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The manifestUrl/entryUrl returned by /api/plugins must each 200."""
    bundle = _make_bundle(tmp_path, "alpha")
    _patch_discover(monkeypatch, {"alpha": bundle})

    client = TestClient(create_app(serve_static=False))
    listing = client.get("/api/plugins").json()["plugins"]
    assert len(listing) == 1
    entry = listing[0]

    m = client.get(entry["manifestUrl"])
    e = client.get(entry["entryUrl"])
    assert m.status_code == 200, m.text
    assert e.status_code == 200, e.text


def test_unknown_plugin_id_404(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_discover(monkeypatch, {})

    client = TestClient(create_app(serve_static=False))
    resp = client.get("/api/plugins/does-not-exist/manifest.json")
    assert resp.status_code == 404


def test_old_static_path_no_longer_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The legacy ``/api/plugins/<id>/static/index.js`` path from spec 06
    is gone — files are now at ``/api/plugins/<id>/index.js``."""
    bundle = _make_bundle(tmp_path, "alpha")
    _patch_discover(monkeypatch, {"alpha": bundle})

    client = TestClient(create_app(serve_static=False))
    resp = client.get("/api/plugins/alpha/static/index.js")
    assert resp.status_code == 404
