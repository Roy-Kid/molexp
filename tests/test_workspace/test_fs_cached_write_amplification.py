"""Cache sidecar write-amplification (workflow-workspace-hardening P1-7).

``CachedRemoteFileSystem`` rewrote its whole ``mirror.json`` sidecar on every
``_record`` / ``listdir`` / ``_invalidate``, so a bulk walk
(``prefetch_workspace_indices``) was O(records²). The sidecar is a derived
cache index; ``batched()`` now defers per-op writes and flushes once on exit,
and ``prefetch_workspace_indices`` wraps its walk in it.
"""

from __future__ import annotations

from molexp.workspace import Workspace
from molexp.workspace.fs_cached import CachedRemoteFileSystem, prefetch_workspace_indices
from molexp.workspace.fs_local import LocalFileSystem


def _count_writes(monkeypatch) -> dict[str, int]:
    counter = {"n": 0}
    orig = CachedRemoteFileSystem._write_sidecar

    def counting(self):
        counter["n"] += 1
        return orig(self)

    monkeypatch.setattr(CachedRemoteFileSystem, "_write_sidecar", counting)
    return counter


def test_batched_defers_to_single_write(tmp_path, monkeypatch):
    fs = CachedRemoteFileSystem(LocalFileSystem(), mirror_root=tmp_path / "mirror")
    writes = _count_writes(monkeypatch)

    with fs.batched():
        for i in range(30):
            fs._record(f"/k/{i}", kind="file", size=1, mtime=0.0)
        assert writes["n"] == 0, "no sidecar write should happen inside the batch"

    assert writes["n"] == 1, "exactly one sidecar write on batch exit"


def test_unbatched_still_writes_per_op(tmp_path, monkeypatch):
    fs = CachedRemoteFileSystem(LocalFileSystem(), mirror_root=tmp_path / "mirror")
    writes = _count_writes(monkeypatch)
    fs._record("/k/1", kind="file", size=1, mtime=0.0)
    fs._record("/k/2", kind="file", size=1, mtime=0.0)
    assert writes["n"] == 2  # single ops persist immediately (correctness over speed)


def test_prefetch_writes_sidecar_once(tmp_path, monkeypatch):
    # Build a workspace with several entities on a plain local FS.
    root = tmp_path / "lab"
    ws = Workspace(root=root, name="lab")
    for p in range(3):
        proj = ws.add_project(f"proj{p}")
        for e in range(2):
            exp = proj.add_experiment(f"exp{e}")
            exp.add_run(params={"i": 0})

    # Navigate it through a fresh cached FS and count sidecar writes.
    cached = CachedRemoteFileSystem(LocalFileSystem(), mirror_root=tmp_path / "mirror")
    ws_cached = Workspace(root=root, fs=cached)

    writes = _count_writes(monkeypatch)
    warnings = prefetch_workspace_indices(ws_cached)

    assert warnings == []
    # The whole walk (≥ a dozen reads + listdirs) persists the sidecar once.
    assert writes["n"] == 1, f"prefetch wrote sidecar {writes['n']}x (expected 1, batched)"
