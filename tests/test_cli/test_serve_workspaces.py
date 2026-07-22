"""``molexp serve -ws`` workspace resolution.

``_resolve_served`` — one ``--workspace`` spec → a ``ServedWorkspace``: local
dir (existing/plain/missing), file rejection, SCP inline target, ``@name``
registry lookup, and key uniqueness.
"""

from __future__ import annotations

import pytest
import typer

from molexp.cli.workspace.serve import _resolve_served
from molexp.server.workspace_targets import WorkspaceTarget, WorkspaceTargetRegistry


class TestResolveServed:
    def test_local_dir_with_workspace_json_is_left_untouched(self, tmp_path):
        (tmp_path / "workspace.json").write_text("{}")
        used: set[str] = set()
        sw = _resolve_served(tmp_path, used)
        assert sw.is_remote is False
        assert sw.target_name is None
        assert sw.path == str(tmp_path.resolve())
        assert sw.key.startswith("local-")
        assert (tmp_path / "workspace.json").is_file()

    def test_plain_folder_opens_without_materializing(self, tmp_path):
        ws = tmp_path / "plain"
        ws.mkdir()
        (ws / "notes.txt").write_text("hi\n")
        sw = _resolve_served(ws, set())
        assert sw.path == str(ws.resolve())
        assert sw.is_remote is False
        assert not (ws / "workspace.json").exists()
        assert (ws / "notes.txt").read_text() == "hi\n"

    def test_missing_dir_is_created_empty(self, tmp_path):
        missing = tmp_path / "new-root"
        sw = _resolve_served(missing, set())
        assert sw.path == str(missing.resolve())
        assert missing.is_dir()
        assert not (missing / "workspace.json").exists()

    def test_file_target_is_rejected(self, tmp_path):
        f = tmp_path / "not-a-dir"
        f.write_text("x")
        with pytest.raises(typer.Exit):
            _resolve_served(f, set())

    def test_scp_spec_yields_remote_with_inline_target(self, tmp_path):
        sw = _resolve_served("alice@dardel:/cfs/user/lab", set())
        assert sw.is_remote is True
        assert sw.path is None
        assert sw.remote_target is not None
        assert sw.remote_target.host == "alice@dardel"
        assert sw.remote_target.root_path == "/cfs/user/lab"
        assert sw.target_name is not None
        assert "dardel" in sw.key or "remote" in sw.key
        assert "dardel" in sw.label

    def test_at_name_resolves_via_registry(self, tmp_path, monkeypatch):
        import molexp.server.deps.targets as targets_deps

        registry = WorkspaceTargetRegistry(store_path=tmp_path / "wt.json")
        registry.add(WorkspaceTarget(name="dardel", host="me@dardel.example", root_path="/proj/ws"))
        monkeypatch.setattr(targets_deps, "_workspace_target_registry", registry)

        sw = _resolve_served("@dardel", set())
        assert sw.is_remote is True
        assert sw.target_name == "dardel"
        assert sw.remote_target is not None
        assert sw.remote_target.root_path == "/proj/ws"

    def test_unknown_at_name_exits(self, tmp_path, monkeypatch):
        import molexp.server.deps.targets as targets_deps

        registry = WorkspaceTargetRegistry(store_path=tmp_path / "wt.json")
        monkeypatch.setattr(targets_deps, "_workspace_target_registry", registry)
        with pytest.raises(typer.Exit):
            _resolve_served("@missing", set())

    def test_same_basename_gets_distinct_keys(self, tmp_path):
        a = tmp_path / "x" / "ws"
        b = tmp_path / "y" / "ws"
        for p in (a, b):
            p.mkdir(parents=True)
        used: set[str] = set()
        ka = _resolve_served(a, used).key
        kb = _resolve_served(b, used).key
        assert ka != kb  # same basename "ws" -> disambiguated
