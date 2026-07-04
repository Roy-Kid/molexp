"""``molexp context`` — renders the assembled WorkspaceContext (workspace-context-03-cli)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import molexp.workspace as mw
from molexp.cli import app
from molexp.workspace import Workspace


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_context_populated(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(root=tmp_path, name="Lab")
    ws.materialize()
    ws.add_project("p").add_experiment("e").add_run(params={"k": 1})
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["context"])

    assert result.exit_code == 0, result.output
    assert "Lab" in result.output
    assert "recent runs:" in result.output  # the run surfaces


def test_context_uses_shared_backend(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(root=tmp_path, name="Lab")
    ws.materialize()
    ws.add_project("p")
    monkeypatch.chdir(tmp_path)

    calls: list[int] = []
    original = mw.assemble_workspace_context

    def _spy(*args: object, **kwargs: object):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mw, "assemble_workspace_context", _spy)

    result = runner.invoke(app, ["context"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1  # one shared assemble_workspace_context call, no re-walk
