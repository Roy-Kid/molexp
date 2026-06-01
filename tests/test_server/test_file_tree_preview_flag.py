"""Tests that the workspace file tree surfaces the sidecar-preview flag.

The ``GET /api/workspace/files?include=catalog`` listing is index-driven: it
iterates registered catalog assets and, for each resolved file, sets
``hasPreviewSidecar`` from the existence-only ``resolve_sidecar`` probe. A
plain file with no same-stem ``.py`` sibling never carries a truthy flag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Building the sidecar dataset shares the test_preview helper; molpy is only
# needed for the reader machinery, not for the existence-only flag, but the
# helper imports nothing — guard defensively to match test_preview.
pytest.importorskip("molpy")

from tests.test_server.test_preview import _make_sidecar_dataset


def _node_by_name(children: list[dict], name: str) -> dict | None:
    for node in children:
        if node.get("name") == name:
            return node
    return None


def test_file_tree_flags_registered_sidecar_dataset(client, workspace):
    root = Path(workspace.root)

    # A sidecar-backed dataset registered in place.
    dataset = _make_sidecar_dataset(root, stem="qm9")
    workspace.data_assets.register_in_place(name="qm9", src=dataset)

    # A plain file (no sidecar), also registered so it appears in the index.
    plain = root / "plain.bin"
    plain.write_bytes(b"x")
    workspace.data_assets.register_in_place(name="plain", src=plain)

    resp = client.get("/api/workspace/files", params={"include": "catalog"})
    assert resp.status_code == 200

    children = resp.json()["children"]
    with_node = _node_by_name(children, "qm9.bin")
    without_node = _node_by_name(children, "plain.bin")

    assert with_node is not None
    assert with_node["hasPreviewSidecar"] is True

    assert without_node is not None
    assert not without_node.get("hasPreviewSidecar")
