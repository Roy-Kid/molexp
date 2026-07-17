"""``GET /api/workspace/files`` honours workspace ``.gitignore`` rules."""

from __future__ import annotations

from pathlib import Path


def test_files_tree_omits_gitignored_entries(client, workspace) -> None:
    root = Path(workspace.root)
    (root / ".gitignore").write_text("scratch/\n*.log\n")
    (root / "scratch").mkdir()
    (root / "scratch" / "secret.txt").write_text("nope")
    (root / "visible.txt").write_text("ok")
    (root / "noise.log").write_text("log")
    (root / "notes").mkdir()
    (root / "notes" / "a.md").write_text("# a")

    # Safety floor: node_modules never listed even without a pattern in .gitignore.
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.js").write_text("x")

    resp = client.get("/api/workspace/files", params={"max_depth": 4})
    assert resp.status_code == 200
    names = {c["name"] for c in resp.json()["children"]}
    assert "visible.txt" in names
    assert "notes" in names
    assert "scratch" not in names
    assert "noise.log" not in names
    assert "node_modules" not in names
