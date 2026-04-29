"""Tests for ``molexp target`` add/list/remove/test."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from molexp.cli import app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def initialized_ws(tmp_path, runner):
    """A freshly initialised workspace at ``tmp_path``."""
    runner.invoke(app, ["init", str(tmp_path)])
    return tmp_path


@pytest.mark.integration
def test_list_empty(runner, initialized_ws):
    result = runner.invoke(app, ["target", "list", "--path", str(initialized_ws)])
    assert result.exit_code == 0
    assert "No compute targets" in result.stdout


@pytest.mark.integration
def test_add_local_target(runner, initialized_ws):
    result = runner.invoke(
        app,
        [
            "target", "add", "laptop",
            "--scratch", "/tmp/molexp-scratch",
            "--path", str(initialized_ws),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Added target laptop" in result.stdout
    assert "scheduler=shell" in result.stdout

    listing = runner.invoke(app, ["target", "list", "--path", str(initialized_ws)])
    assert "laptop" in listing.stdout
    assert "shell" in listing.stdout


@pytest.mark.integration
def test_add_remote_target(runner, initialized_ws):
    result = runner.invoke(
        app,
        [
            "target", "add", "hpc",
            "--scratch", "/scratch/me",
            "--scheduler", "slurm",
            "--host", "me@cluster",
            "--port", "2222",
            "--path", str(initialized_ws),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "scheduler=slurm" in result.stdout

    listing = runner.invoke(app, ["target", "list", "--path", str(initialized_ws)])
    assert "me@cluster" in listing.stdout


@pytest.mark.integration
def test_add_invalid_scheduler_rejected(runner, initialized_ws):
    result = runner.invoke(
        app,
        [
            "target", "add", "x",
            "--scratch", "/tmp",
            "--scheduler", "invalid",
            "--path", str(initialized_ws),
        ],
    )
    assert result.exit_code == 2


@pytest.mark.integration
def test_add_duplicate_rejected(runner, initialized_ws):
    runner.invoke(
        app,
        ["target", "add", "x", "--scratch", "/tmp", "--path", str(initialized_ws)],
    )
    result = runner.invoke(
        app,
        ["target", "add", "x", "--scratch", "/other", "--path", str(initialized_ws)],
    )
    assert result.exit_code == 1
    assert "already exists" in result.stdout


@pytest.mark.integration
def test_remove(runner, initialized_ws):
    runner.invoke(
        app, ["target", "add", "x", "--scratch", "/tmp", "--path", str(initialized_ws)],
    )
    result = runner.invoke(
        app, ["target", "remove", "x", "--path", str(initialized_ws)],
    )
    assert result.exit_code == 0
    assert "Removed target x" in result.stdout

    listing = runner.invoke(app, ["target", "list", "--path", str(initialized_ws)])
    assert "x" not in listing.stdout or "No compute targets" in listing.stdout


@pytest.mark.integration
def test_remove_missing(runner, initialized_ws):
    result = runner.invoke(
        app, ["target", "remove", "ghost", "--path", str(initialized_ws)],
    )
    assert result.exit_code == 1


@pytest.mark.integration
def test_test_local_target_succeeds(runner, initialized_ws, tmp_path):
    scratch = tmp_path / "scratch"
    runner.invoke(
        app,
        [
            "target", "add", "laptop",
            "--scratch", str(scratch),
            "--path", str(initialized_ws),
        ],
    )
    result = runner.invoke(
        app, ["target", "test", "laptop", "--path", str(initialized_ws)],
    )
    assert result.exit_code == 0, result.stdout
    assert "ok command execution" in result.stdout
    assert "ok mkdir" in result.stdout
    assert "ok file round-trip" in result.stdout
    assert scratch.exists()


@pytest.mark.integration
def test_test_missing_target(runner, initialized_ws):
    result = runner.invoke(
        app, ["target", "test", "ghost", "--path", str(initialized_ws)],
    )
    assert result.exit_code == 1
