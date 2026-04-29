"""Regression tests for ``molexp init``.

The ``Workspace(...)`` constructor is intentionally side-effect-free
(see CLAUDE.md), so ``init`` MUST call ``materialize()`` to actually
create the directory and ``workspace.json``. A missing materialize
call leaves the user with a "directory does not exist" error from
``molexp serve`` and no diagnostic.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from molexp.cli import app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.integration
def test_init_creates_directory_and_workspace_json(runner, tmp_path):
    target = tmp_path / "fresh-workspace"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.stdout
    assert target.is_dir()
    assert (target / "workspace.json").is_file()


@pytest.mark.integration
def test_init_is_idempotent(runner, tmp_path):
    """Running init twice on the same path must not error or wipe state."""
    target = tmp_path / "ws"
    runner.invoke(app, ["init", str(target)])
    # Drop a sentinel so we can prove a re-init didn't blow it away.
    sentinel = target / "projects" / "_sentinel"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("preserved")

    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.stdout
    assert sentinel.read_text() == "preserved"


@pytest.mark.integration
def test_init_without_argument_uses_cwd(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "workspace.json").is_file()
