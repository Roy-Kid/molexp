"""Tests for ``GET /api/assets`` — content-hash lookup (vision-loop-08 asset-query symmetry).

The ``content_hash`` query param is PURE EXPOSURE of the one existing
content-addressed lookup (:func:`molexp.workspace.assets.scan.find_by_content_hash`,
scanning the authoritative per-scope ``assets.json`` manifests — no derived
index): an exact hash answers with a 0/1-element list in the uniform
``AssetResponse`` shape, and an absent param keeps the old listing behavior.

Category map (tester):
    basics       — exact hash -> 1-element list carrying the matched asset
    integration  — param absent keeps the pre-existing full-listing behavior
"""

from __future__ import annotations

import hashlib

import pytest

from molexp.workspace.assets.base import Asset

PAYLOAD_A = b"vl08 payload alpha\n"
PAYLOAD_B = b"vl08 payload beta\n"
HASH_A = "sha256:" + hashlib.sha256(PAYLOAD_A).hexdigest()
HASH_B = "sha256:" + hashlib.sha256(PAYLOAD_B).hexdigest()


@pytest.fixture
def seeded_assets(workspace, tmp_path) -> tuple[Asset, Asset]:
    """Two workspace-scoped ``DataAsset``s with known, distinct content hashes."""
    incoming = tmp_path / "_incoming"
    incoming.mkdir()
    src_a = incoming / "alpha.bin"
    src_a.write_bytes(PAYLOAD_A)
    src_b = incoming / "beta.bin"
    src_b.write_bytes(PAYLOAD_B)
    asset_a = workspace.data_assets.import_asset("alpha-input", src_a)
    asset_b = workspace.data_assets.import_asset("beta-input", src_b)
    return asset_a, asset_b


def test_content_hash_exact_match_returns_single_element_list(client, seeded_assets):
    """basics: an exact content hash answers with exactly the one matching asset."""
    asset_a, _ = seeded_assets

    resp = client.get("/api/assets", params={"content_hash": HASH_A})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1  # 0/1-element list — uniform response shape
    row = body[0]
    assert row["id"] == asset_a.asset_id
    assert row["name"] == "alpha-input"
    assert row["content_hash"] == HASH_A


def test_content_hash_absent_keeps_full_listing_behavior(client, seeded_assets):
    """integration: without the param the route keeps its pre-existing listing."""
    asset_a, asset_b = seeded_assets

    resp = client.get("/api/assets")
    assert resp.status_code == 200
    body = resp.json()
    assert {row["id"] for row in body} == {asset_a.asset_id, asset_b.asset_id}
    assert {row["content_hash"] for row in body} == {HASH_A, HASH_B}
