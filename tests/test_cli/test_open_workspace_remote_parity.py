"""CLI remote parity: open_workspace + no remote-only guards for FS CRUD."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from molexp.cli._target import open_workspace, resolve_workspace_target
from molexp.workspace import Workspace
from molexp.workspace.fs_local import LocalFileSystem


def test_open_workspace_local_roundtrip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, name="lab")
    ws.materialize()
    target, _transport, fs, opened = open_workspace(str(tmp_path))
    assert isinstance(fs, LocalFileSystem)
    assert opened.metadata.id == ws.metadata.id
    assert str(target.path) == str(tmp_path.resolve()) or Path(target.path) == tmp_path.resolve()


def test_open_workspace_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="No workspace found"):
        open_workspace(str(missing))


def test_project_list_via_cli_local(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, name="lab")
    ws.materialize()
    ws.add_project("alpha")
    from molexp.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["project", "list", "-ws", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output


def test_context_cli_local(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, name="lab")
    ws.materialize()
    ws.add_project("alpha")
    from molexp.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["context", "-ws", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "projects" in result.output.lower() or "alpha" in result.output or "lab" in result.output


def test_resolve_and_open_share_fs(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, name="lab")
    ws.materialize()
    t1, _tr1, fs1 = resolve_workspace_target(str(tmp_path))
    t2, _tr2, fs2, opened = open_workspace(str(tmp_path))
    assert type(fs1) is type(fs2)
    assert str(t1.path) == str(t2.path)
    assert opened.metadata.name == "lab"
