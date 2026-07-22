"""Tests for ``molexp serve -ws`` — local / plain-folder / SCP / @target."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from molexp.cli import app
from molexp.cli.workspace.serve import _resolve_served
from molexp.server.dependencies import (
    set_active_workspace_descriptor,
    set_served_workspaces,
    set_workspace_path_override,
)
from molexp.server.workspace_targets import WorkspaceTarget, WorkspaceTargetRegistry
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_served_state():
    yield
    set_served_workspaces([])
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)


def test_resolve_served_local_with_workspace_json(tmp_path):
    (tmp_path / "workspace.json").write_text("{}")
    used: set[str] = set()
    sw = _resolve_served(tmp_path, used)
    assert sw.is_remote is False
    assert sw.target_name is None
    assert sw.path == str(tmp_path.resolve())
    assert sw.key.startswith("local-")
    # Existing layout is left alone.
    assert (tmp_path / "workspace.json").is_file()


def test_resolve_served_plain_folder_does_not_materialize(tmp_path):
    """Ordinary directories open without writing workspace.json."""
    ws = tmp_path / "plain"
    ws.mkdir()
    (ws / "notes.txt").write_text("hi\n")
    sw = _resolve_served(ws, set())
    assert sw.path == str(ws.resolve())
    assert sw.is_remote is False
    assert not (ws / "workspace.json").exists()
    assert (ws / "notes.txt").read_text() == "hi\n"


def test_resolve_served_creates_missing_dir_without_workspace_json(tmp_path):
    missing = tmp_path / "new-root"
    sw = _resolve_served(missing, set())
    assert sw.path == str(missing.resolve())
    assert missing.is_dir()
    assert not (missing / "workspace.json").exists()


def test_resolve_served_rejects_file(tmp_path):
    f = tmp_path / "not-a-dir"
    f.write_text("x")
    with pytest.raises(typer.Exit):
        _resolve_served(f, set())


def test_resolve_served_scp_remote_inline(tmp_path):
    """SCP specs produce a remote ServedWorkspace with an inline target."""
    sw = _resolve_served("alice@dardel:/cfs/user/lab", set())
    assert sw.is_remote is True
    assert sw.path is None
    assert sw.remote_target is not None
    assert sw.remote_target.host == "alice@dardel"
    assert sw.remote_target.root_path == "/cfs/user/lab"
    assert sw.target_name is not None
    assert "dardel" in sw.key or "remote" in sw.key
    assert "dardel" in sw.label


def test_resolve_served_at_name_uses_registry(tmp_path, monkeypatch):
    import molexp.server.deps.targets as targets_deps

    registry = WorkspaceTargetRegistry(store_path=tmp_path / "wt.json")
    registry.add(WorkspaceTarget(name="dardel", host="me@dardel.example", root_path="/proj/ws"))
    monkeypatch.setattr(targets_deps, "_workspace_target_registry", registry)

    sw = _resolve_served("@dardel", set())
    assert sw.is_remote is True
    assert sw.target_name == "dardel"
    assert sw.remote_target is not None
    assert sw.remote_target.root_path == "/proj/ws"


def test_resolve_served_at_name_unknown(tmp_path, monkeypatch):
    import molexp.server.deps.targets as targets_deps

    registry = WorkspaceTargetRegistry(store_path=tmp_path / "wt.json")
    monkeypatch.setattr(targets_deps, "_workspace_target_registry", registry)
    with pytest.raises(typer.Exit):
        _resolve_served("@missing", set())


def test_resolve_served_distinct_keys_for_same_basename(tmp_path):
    a = tmp_path / "x" / "ws"
    b = tmp_path / "y" / "ws"
    for p in (a, b):
        p.mkdir(parents=True)
    used: set[str] = set()
    ka = _resolve_served(a, used).key
    kb = _resolve_served(b, used).key
    assert ka != kb  # same basename "ws" -> disambiguated


def test_serve_help_only_exposes_workspace_option() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "-ws" in plain
    assert "--workspace" in plain
    assert "user@host" in plain or "user@host:/path" in plain
    assert "--target" not in plain


def test_serve_rejects_removed_target_option() -> None:
    result = runner.invoke(app, ["serve", "--target", "."])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_serve_opens_plain_folder_without_materialize(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "plain"
    target.mkdir()
    original_cwd = Path.cwd()
    called: list[object] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: called.append((args, kwargs)))

    result = runner.invoke(app, ["serve", "-ws", str(target)])

    assert result.exit_code == 0, result.output
    assert not (target / "workspace.json").exists()
    assert Path.cwd() == original_cwd
    assert called


def test_serve_without_ws_does_not_materialize_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "workspace.json").exists()


def test_serve_activates_remote_descriptor(tmp_path: Path, monkeypatch) -> None:
    """Primary SCP -ws activates the remote descriptor (not a local path)."""
    monkeypatch.setattr("uvicorn.run", lambda *_a, **_k: None)
    # Avoid real SSH config resolution noise.
    monkeypatch.setattr(
        "molexp.workspace.target._resolve_ssh_details",
        lambda _host: (None, None),
    )

    result = runner.invoke(app, ["serve", "-ws", "user@host.example:/scratch/ws"])
    assert result.exit_code == 0, result.output
    assert "host.example" in result.output or "remote" in result.output.lower()
