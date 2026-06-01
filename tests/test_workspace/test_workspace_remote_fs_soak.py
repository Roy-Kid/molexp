"""Soak test: Workspace → Project → Experiment → Run over a non-local FileSystem.

The recent ``feat(agent): route Agent/AgentSession/PlanFolder I/O through
self._fs`` work (commit ``4e05199``) proved the agent-layer Folder
subclasses route through ``self._fs``.  This test extends the same
``_SpyFileSystem`` pattern to the *scientific* hierarchy
(:class:`~molexp.workspace.Workspace` / :class:`Project` /
:class:`Experiment` / :class:`Run`) — the hierarchy the
``WorkspaceTarget`` registry endpoints will mount remotely.

Failure of this test would mean the workspace-target endpoints could
register a remote descriptor that the server cannot in fact use as the
active workspace, retiring the medium-confidence librarian risk.
"""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace
from molexp.workspace.fs_local import LocalFileSystem


class _SpyFileSystem:
    """Records every method call but delegates to a real LocalFileSystem."""

    def __init__(self) -> None:
        self._real = LocalFileSystem()
        self.calls: list[tuple[str, str]] = []

    def _record(self, name: str, path: object) -> None:
        try:
            self.calls.append((name, str(path)))
        except Exception:
            self.calls.append((name, "<unrepresentable>"))

    def __getattr__(self, name: str):
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def wrapped(*args: object, **kwargs: object) -> object:
            if args:
                self._record(name, args[0])
            else:
                self._record(name, kwargs.get("path", ""))
            return attr(*args, **kwargs)

        return wrapped


@pytest.fixture
def spy_workspace(tmp_path):
    fs = _SpyFileSystem()
    ws = Workspace(root=tmp_path / "remote_lab", name="Remote Lab", fs=fs)
    ws.materialize()
    return ws, fs


def _ops_touching(calls: list[tuple[str, str]], needle: str) -> set[str]:
    return {op for op, path in calls if needle in path}


@pytest.mark.unit
def test_workspace_construction_routes_through_fs(spy_workspace):
    """``Workspace(root, fs=spy)`` must reach for ``self._fs.join`` / ``exists``."""
    _ws, fs = spy_workspace
    ops = {op for op, _ in fs.calls}
    # At minimum, the constructor + materialize touched join/exists/mkdir.
    assert {"join", "exists"} <= ops, f"missing fs operations; saw {ops}"


@pytest.mark.unit
def test_add_project_writes_metadata_through_fs(spy_workspace):
    ws, fs = spy_workspace
    fs.calls.clear()
    proj = ws.add_project("alpha")
    assert proj is not None
    # Project mkdir + index write must have flowed through fs.
    proj_ops = _ops_touching(fs.calls, "alpha")
    assert "mkdir" in proj_ops, fs.calls
    assert proj.name == "alpha"


@pytest.mark.unit
def test_list_projects_round_trips(spy_workspace):
    ws, _ = spy_workspace
    ws.add_project("alpha")
    ws.add_project("beta")
    names = sorted(p.name for p in ws.list_projects())
    assert names == ["alpha", "beta"]


@pytest.mark.unit
def test_full_hierarchy_round_trips_through_fs(spy_workspace):
    """Workspace → Project → Experiment → Run, every level via fs."""
    ws, fs = spy_workspace
    proj = ws.add_project("alpha")
    exp = proj.add_experiment("first-exp", workflow_source="train.py")
    run = exp.add_run(parameters={"lr": 1e-3})

    # Every level touched a path under remote_lab/.
    for needle in ("alpha", "first-exp", run.metadata.id):
        ops = _ops_touching(fs.calls, needle)
        assert ops, f"no fs ops for {needle}; calls: {fs.calls}"
        assert "mkdir" in ops, f"no mkdir under {needle}; ops: {ops}"


@pytest.mark.unit
def test_non_local_filesystem_branch_skips_path_resolve(tmp_path):
    """The non-LocalFileSystem branch (workspace.py:87-88) must not call
    ``.resolve()`` — the input string is meaningful on the *remote* host,
    not after local pathlib normalisation."""

    class _NonLocalFS(_SpyFileSystem):
        """Subclasses _SpyFileSystem but is NOT a LocalFileSystem."""

    fs = _NonLocalFS()
    # Pass a path with a tilde-style segment that local resolve would mangle.
    raw = str(tmp_path / "remote_root_with_special$chars")
    ws = Workspace(root=raw, name="Remote", fs=fs)

    # ``_root_path`` stores the original string verbatim (modulo Path()).
    assert str(ws.root) == raw

    # No resolve() call should ever have been recorded on the spy.
    resolve_calls = [c for c in fs.calls if c[0] == "resolve"]
    assert not resolve_calls, f"unexpected resolve() on non-LocalFileSystem: {resolve_calls}"


# ── CachedRemoteFileSystem soak ─────────────────────────────────────────


@pytest.mark.unit
def test_workspace_hierarchy_with_cached_wrapper_serves_second_read_from_mirror(
    tmp_path,
):
    """The full Workspace→Project→Experiment→Run dance works through a
    :class:`CachedRemoteFileSystem` wrapper, and a repeat read of any
    persisted JSON file does not call the inner FS again.
    """
    from molexp.workspace.fs_cached import CachedRemoteFileSystem

    inner = _SpyFileSystem()
    cached = CachedRemoteFileSystem(inner, mirror_root=tmp_path / "mirror", ttl_seconds=300)
    ws = Workspace(root=tmp_path / "remote_lab", name="Remote Lab", fs=cached)
    ws.materialize()
    proj = ws.add_project("alpha")
    exp = proj.add_experiment("first-exp", workflow_source="train.py")
    run = exp.add_run(parameters={"lr": 1e-3})

    # Pick a stable file the run created and read it twice through the cache.
    run_meta_path = inner.join(str(run.resolve()), "run.json")
    inner.calls.clear()
    first = cached.read_bytes(run_meta_path)
    second = cached.read_bytes(run_meta_path)
    assert first == second
    read_byte_calls = [c for c in inner.calls if c[0] == "read_bytes"]
    assert len(read_byte_calls) == 1, f"second read must come from mirror; saw {inner.calls}"
