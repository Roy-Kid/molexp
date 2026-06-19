"""Per-run source snapshot — entrypoint + first-party local-import closure."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

from molexp.workspace.source_snapshot import snapshot_sources


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body), encoding="utf-8")


def test_captures_entrypoint_and_transitive_first_party_imports(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write(src / "helper.py", "VALUE = 1\n")
    _write(src / "deep.py", "import helper\nX = helper.VALUE\n")
    _write(src / "entry.py", "import deep\nfrom os import path\nimport numpy as _np\n")
    run_dir = tmp_path / "run"

    manifest = snapshot_sources(src / "entry.py", run_dir, now=datetime(2026, 6, 17, 13, 0, 0))

    # entry + transitive first-party (deep -> helper); stdlib (os) / third-party
    # (numpy) have no sibling .py and are excluded.
    names = {f["name"] for f in manifest["files"]}
    assert names == {"entry.py", "deep.py", "helper.py"}
    assert manifest["entrypoint"] == "entry.py"
    assert manifest["dir"] == "source"
    assert manifest["captured_at"] == "2026-06-17T13:00:00"

    snap = run_dir / "source"
    assert (snap / "entry.py").read_text() == (src / "entry.py").read_text()
    assert (snap / "helper.py").read_text() == "VALUE = 1\n"
    for entry in manifest["files"]:
        assert entry["sha256"].startswith("sha256:")


def test_excludes_nonlocal_and_is_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write(src / "solo.py", "import json\nimport os\n")
    run_dir = tmp_path / "run"

    first = snapshot_sources(src / "solo.py", run_dir)
    second = snapshot_sources(src / "solo.py", run_dir)

    assert {f["name"] for f in first["files"]} == {"solo.py"}  # no stdlib copied
    assert [f["sha256"] for f in first["files"]] == [f["sha256"] for f in second["files"]]
