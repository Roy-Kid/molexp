"""Tests for ``GET /api/plugins`` after the 07 split.

The new contract is:

* response shape is ``{plugins: [{id, manifestUrl, entryUrl}, ...], total}``;
* ``manifestUrl`` is ``"/api/plugins/<id>/manifest.json"`` and
  ``entryUrl`` is ``"/api/plugins/<id>/index.js"``;
* built-in ``core``/``metrics``/``molq`` ids do NOT appear — they are
  statically imported on the frontend, not entry-point-discovered;
* ``UiPluginResponse`` carries no UI semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app


def _make_bundle(root: Path, plugin_id: str) -> Path:
    """Create a minimal valid bundle directory at ``root / plugin_id``."""
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
    """Force ``discover_ui_plugin_dirs`` to return *mapping*."""
    monkeypatch.setattr(
        "molexp.plugins.ui.discover_ui_plugin_dirs",
        lambda: dict(mapping),
    )
    # Server module imports the symbol at import time — patch the bound name
    # everywhere it surfaces so app boot picks it up.
    import molexp.server.app as app_mod
    import molexp.server.routes.registry as registry_mod

    if hasattr(app_mod, "discover_ui_plugin_dirs"):
        monkeypatch.setattr(app_mod, "discover_ui_plugin_dirs", lambda: dict(mapping))
    if hasattr(registry_mod, "discover_ui_plugin_dirs"):
        monkeypatch.setattr(registry_mod, "discover_ui_plugin_dirs", lambda: dict(mapping))


# ── ac-006 / ac-007 ────────────────────────────────────────────────────


def test_response_shape_is_id_manifesturl_entryurl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _make_bundle(tmp_path, "alpha")
    _patch_discover(monkeypatch, {"alpha": bundle})

    client = TestClient(create_app(serve_static=False))
    resp = client.get("/api/plugins")
    assert resp.status_code == 200
    data = resp.json()

    assert "plugins" in data
    assert "total" in data
    assert data["total"] == len(data["plugins"]) == 1

    entry = data["plugins"][0]
    assert set(entry.keys()) == {"id", "manifestUrl", "entryUrl"}
    assert entry["id"] == "alpha"
    assert entry["manifestUrl"] == "/api/plugins/alpha/manifest.json"
    assert entry["entryUrl"] == "/api/plugins/alpha/index.js"


def test_builtin_ids_not_in_listing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Built-in plugins are statically imported on the frontend; they
    must NOT appear in ``/api/plugins`` after the 07 split."""
    _patch_discover(monkeypatch, {})
    client = TestClient(create_app(serve_static=False))
    data = client.get("/api/plugins").json()

    ids = {p["id"] for p in data["plugins"]}
    assert "core" not in ids
    assert "metrics" not in ids
    assert "molq" not in ids


def test_multiple_plugins_each_get_their_own_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle_a = _make_bundle(tmp_path, "alpha")
    bundle_b = _make_bundle(tmp_path, "beta")
    _patch_discover(monkeypatch, {"alpha": bundle_a, "beta": bundle_b})

    client = TestClient(create_app(serve_static=False))
    data = client.get("/api/plugins").json()

    by_id = {p["id"]: p for p in data["plugins"]}
    assert by_id["alpha"]["manifestUrl"] == "/api/plugins/alpha/manifest.json"
    assert by_id["alpha"]["entryUrl"] == "/api/plugins/alpha/index.js"
    assert by_id["beta"]["manifestUrl"] == "/api/plugins/beta/manifest.json"
    assert by_id["beta"]["entryUrl"] == "/api/plugins/beta/index.js"


def test_openapi_schema_matches_new_shape() -> None:
    """The OpenAPI surface must declare exactly the three new fields so
    ``npm run generate:api`` produces a matching TypeScript client."""
    schema = create_app(serve_static=False).openapi()
    plugin_resp = schema["components"]["schemas"]["UiPluginResponse"]
    props = plugin_resp["properties"]
    assert set(props.keys()) == {"id", "manifestUrl", "entryUrl"}

    # Legacy fields must be gone.
    for legacy in ("title", "description", "uiModule", "moduleUrl", "capabilities", "metadata"):
        assert legacy not in props
