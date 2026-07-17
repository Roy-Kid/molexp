"""Tests for the pathlib-only ``molexp serve -ws`` surface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from molexp.cli import app
from molexp.cli.workspace.serve import _resolve_served
from molexp.server.dependencies import set_served_workspaces, set_workspace_path_override

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_served_state():
    yield
    set_served_workspaces([])
    set_workspace_path_override(None)


def test_resolve_served_local(tmp_path):
    (tmp_path / "workspace.json").write_text("{}")
    used: set[str] = set()
    sw = _resolve_served(tmp_path, used)
    assert sw.is_remote is False
    assert sw.target_name is None
    assert sw.path == str(tmp_path.resolve())
    assert sw.key.startswith("local-")


def test_resolve_served_creates_workspace_at_exact_path(tmp_path):
    ws = tmp_path / "workspace"
    sw = _resolve_served(ws, set())
    assert sw.path == str(ws.resolve())
    assert (ws / "workspace.json").is_file()


def test_resolve_served_initializes_existing_empty_directory(tmp_path):
    ws = tmp_path / "empty"
    ws.mkdir()
    _resolve_served(ws, set())
    assert (ws / "workspace.json").is_file()


def test_resolve_served_distinct_keys_for_same_basename(tmp_path):
    a = tmp_path / "x" / "ws"
    b = tmp_path / "y" / "ws"
    for p in (a, b):
        p.mkdir(parents=True)
        (p / "workspace.json").write_text("{}")
    used: set[str] = set()
    ka = _resolve_served(a, used).key
    kb = _resolve_served(b, used).key
    assert ka != kb  # same basename "ws" -> disambiguated


def test_serve_help_only_exposes_workspace_option() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    assert "-ws" in result.output
    assert "--workspace" in result.output
    assert "--target" not in result.output
    assert "-t" not in result.output


def test_serve_rejects_removed_target_option() -> None:
    result = runner.invoke(app, ["serve", "--target", "."])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_serve_initializes_missing_workspace_and_does_not_change_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "new-workspace"
    original_cwd = Path.cwd()
    called: list[object] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: called.append((args, kwargs)))

    result = runner.invoke(app, ["serve", "-ws", str(target)])

    assert result.exit_code == 0, result.output
    assert (target / "workspace.json").is_file()
    assert Path.cwd() == original_cwd
    assert called


def test_serve_without_ws_initializes_current_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "workspace.json").is_file()
