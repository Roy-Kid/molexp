"""OKF capabilities on ``workspace.Folder`` (wsokf-01).

Additive: every Folder gains a narrative ``index.md`` whose markdown links are
the knowledge graph (``out_edges`` / ``links``). This sits alongside the
authoritative ``metadata.json`` and never replaces it.
"""

from __future__ import annotations

import os
from pathlib import Path

from molexp.workspace import Workspace


def test_index_round_trip(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "lab")
    ws.materialize()
    proj = ws.add_project("alpha")
    assert proj.read_index() == ""  # absent → empty
    proj.write_index("# Alpha\n\nnarrative\n")
    assert proj.read_index() == "# Alpha\n\nnarrative\n"
    # additive: the project's own metadata is untouched (still listed)
    assert [p.name for p in ws.list_projects()] == ["alpha"]


def test_out_edges_resolves_in_tree_folders(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "lab")
    ws.materialize()
    alpha = ws.add_project("alpha")
    beta = ws.add_project("beta")

    alpha.write_index(
        "# Alpha\n\n"
        "- [to-beta](../beta)\n"
        "- [to-beta-index](../beta/index.md)\n"
        "- [ext](https://example.com)\n"
        "- [nowhere](./nope)\n"
    )

    edges = {os.path.normpath(e) for e in alpha.out_edges()}
    assert os.path.normpath(str(beta.resolve())) in edges

    scan = alpha.links()
    assert any("example.com" in e for e in scan.external)
    assert any("nope" in o for o in scan.other)
