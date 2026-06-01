"""Prefetch walk: partial-failure semantics.

A bad ``experiments.json`` for one project must not abort the walk; it
must surface as a :class:`PrefetchWarning` while other projects still
populate the cache.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

import pytest

from molexp.workspace.fs import StatResult
from molexp.workspace.fs_cached import (
    CachedRemoteFileSystem,
    PrefetchWarning,
    prefetch_workspace_indices,
)


class _ScriptedFS:
    """In-memory FS where individual paths can be primed to raise."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.errors: dict[str, Exception] = {}
        self.dirs: set[str] = set()
        self.calls: Counter[str] = Counter()

    @staticmethod
    def join(*parts: str) -> str:
        # POSIX-style: keep an initial leading slash, strip slashes between segments.
        if not parts:
            return ""
        first = str(parts[0])
        lead = "/" if first.startswith("/") else ""
        cleaned = [str(p).strip("/") for p in parts if p]
        return lead + "/".join(seg for seg in cleaned if seg)

    @staticmethod
    def dirname(path: str) -> str:
        s = str(path)
        return s.rsplit("/", 1)[0] if "/" in s else "."

    @staticmethod
    def basename(path: str) -> str:
        return str(path).rsplit("/", 1)[-1]

    @staticmethod
    def resolve(path: str) -> str:
        return str(path)

    @staticmethod
    def is_absolute(path: str) -> bool:
        return str(path).startswith("/")

    def exists(self, path: str) -> bool:
        self.calls["exists"] += 1
        return str(path) in self.files or str(path) in self.dirs

    def is_dir(self, path: str) -> bool:
        return str(path) in self.dirs

    def is_file(self, path: str) -> bool:
        return str(path) in self.files

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        self.calls["read_text"] += 1
        key = str(path)
        if key in self.errors:
            raise self.errors[key]
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key].decode(encoding)

    def read_bytes(self, path: str) -> bytes:
        key = str(path)
        if key in self.errors:
            raise self.errors[key]
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    def open(self, path: str, mode: str = "r", encoding: str = "utf-8") -> IO[Any]:
        import io as _io

        return _io.StringIO(self.read_text(path, encoding=encoding))

    def write_text(self, path: str, content: str, *, mode: int = 0o600) -> None:
        self.files[str(path)] = content.encode("utf-8")

    def write_bytes(self, path: str, content: bytes, *, mode: int = 0o600) -> None:
        self.files[str(path)] = content

    def rename(self, src: str, dst: str) -> None:
        self.files[str(dst)] = self.files.pop(str(src))

    def remove(self, path: str, *, recursive: bool = False) -> None:
        self.files.pop(str(path), None)

    def copy(self, src: str, dst: str) -> None:
        self.files[str(dst)] = self.files[str(src)]

    def copytree(self, src: str, dst: str, *, dirs_exist_ok: bool = False) -> None:
        pass

    def mkdir(self, path: str, *, parents: bool = True, exist_ok: bool = True) -> None:
        self.dirs.add(str(path))

    def listdir(self, path: str) -> list[str]:
        self.calls["listdir"] += 1
        prefix = str(path).rstrip("/") + "/"
        names: set[str] = set()
        for k in (*self.files.keys(), *self.dirs):
            if k.startswith(prefix):
                tail = k[len(prefix) :]
                names.add(tail.split("/", 1)[0])
        return sorted(names)

    def glob(self, path: str, pattern: str) -> list[str]:
        return []

    def rglob(self, path: str, pattern: str) -> list[str]:
        return []

    def stat(self, path: str) -> StatResult:
        key = str(path)
        if key in self.files:
            return StatResult(size=len(self.files[key]), mtime=42.0, is_dir=False, is_file=True)
        if key in self.dirs:
            return StatResult(size=0, mtime=42.0, is_dir=True, is_file=False)
        raise FileNotFoundError(key)

    def lstat(self, path: str) -> StatResult:
        return self.stat(path)

    def touch(self, path: str) -> None:
        self.files.setdefault(str(path), b"")

    def chmod(self, path: str, mode: int) -> None:
        pass

    def getsize(self, path: str) -> int:
        return len(self.files[str(path)])

    def symlink(self, src: str, dst: str) -> None:
        pass

    def atomic_write_json(self, path: str, data: object) -> None:
        self.files[str(path)] = (json.dumps(data) + "\n").encode("utf-8")

    def atomic_write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        self.files[str(path)] = content.encode("utf-8")


def _seed_workspace(fs: _ScriptedFS, root: str) -> None:
    """Build a 2-project workspace with one project's experiments-index erroring.

    Children-index filenames are auto-derived from ``cls.__name__``
    snake_case (see :meth:`Folder._index_filename`) — singular: workspace's
    projects-index is ``project.json``; project's experiments-index is
    ``experiment.json``; experiment's runs-index is ``run.json``.
    """
    fs.files[f"{root}/workspace.json"] = b'{"id":"ws","name":"ws"}'
    fs.files[f"{root}/project.json"] = json.dumps(
        {"items": [{"path": "alpha"}, {"path": "beta"}]}
    ).encode("utf-8")

    fs.files[f"{root}/projects/alpha/project.json"] = b'{"id":"alpha","name":"alpha"}'
    fs.files[f"{root}/projects/alpha/experiment.json"] = json.dumps(
        {"items": [{"path": "exp1"}]}
    ).encode("utf-8")
    fs.files[f"{root}/projects/alpha/experiments/exp1/experiment.json"] = b'{"id":"exp1"}'
    fs.files[f"{root}/projects/alpha/experiments/exp1/run.json"] = json.dumps(
        {"items": [{"path": "r1"}]}
    ).encode("utf-8")
    fs.files[f"{root}/projects/alpha/experiments/exp1/runs/r1/run.json"] = b'{"id":"r1"}'

    # Beta exists but its experiments-index read raises a transport error.
    fs.files[f"{root}/projects/beta/project.json"] = b'{"id":"beta","name":"beta"}'
    fs.errors[f"{root}/projects/beta/experiment.json"] = ConnectionError("ssh dropped")


@pytest.fixture
def scripted(tmp_path: Path):
    fs = _ScriptedFS()
    root = "/scratch/me/workspace"
    _seed_workspace(fs, root)
    cached = CachedRemoteFileSystem(fs, mirror_root=tmp_path / "mirror", ttl_seconds=300)
    # Synthesize a Workspace-shaped object with just the attributes prefetch needs.
    ws = SimpleNamespace(root=root, _fs=cached)
    return ws, cached, fs


@pytest.mark.unit
def test_partial_failure_yields_warning_and_continues(scripted):
    ws, _cached, _fs = scripted
    warnings = prefetch_workspace_indices(ws)
    bad_paths = [w.path for w in warnings]
    assert "/scratch/me/workspace/projects/beta/experiment.json" in bad_paths
    assert any("ssh dropped" in w.reason for w in warnings), warnings


@pytest.mark.unit
def test_healthy_project_still_hydrated(scripted):
    ws, cached, fs = scripted
    fs.calls.clear()
    prefetch_workspace_indices(ws)

    # Alpha's run.json was read.
    cached_paths = cached.cached_paths()
    assert any("alpha/experiments/exp1/runs/r1/run.json" in k for k in cached_paths), cached_paths


@pytest.mark.unit
def test_walk_uses_listdir_fallback_when_index_missing(tmp_path: Path):
    """A missing children-index file falls back to listdir silently.

    Children-indices are lazily rebuilt by the workspace, so a fresh
    hierarchy commonly lacks them — emitting a warning would create
    noise.  The walk should still hydrate the project's metadata.
    """
    fs = _ScriptedFS()
    root = "/scratch/me/workspace"
    fs.files[f"{root}/workspace.json"] = b"{}"
    # No project.json (children-index of projects) — but the directory has
    # one child whose own project.json exists.
    fs.files[f"{root}/projects/gamma/project.json"] = b'{"id":"gamma"}'
    fs.dirs.add(f"{root}/projects")
    fs.dirs.add(f"{root}/projects/gamma")

    cached = CachedRemoteFileSystem(fs, mirror_root=tmp_path / "mirror", ttl_seconds=300)
    ws = SimpleNamespace(root=root, _fs=cached)

    warnings = prefetch_workspace_indices(ws)
    # No warnings — missing children-index is normal for a fresh hierarchy.
    assert warnings == []
    cached_paths = cached.cached_paths()
    assert any("projects/gamma/project.json" in k for k in cached_paths)


@pytest.mark.unit
def test_warnings_are_immutable_prefetch_warnings(scripted):
    ws, _cached, _fs = scripted
    warnings = prefetch_workspace_indices(ws)
    assert all(isinstance(w, PrefetchWarning) for w in warnings)
    with pytest.raises(Exception):  # noqa: B017 — dataclass frozen
        warnings[0].path = "tampered"  # type: ignore[misc]
