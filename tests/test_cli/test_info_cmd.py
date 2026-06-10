"""``molexp info`` fails loudly when the path holds no workspace.

A default-constructed ``Workspace`` used to make ``info`` print a
healthy-looking empty workspace for any directory; now a missing
``workspace.json`` is a hard error pointing at ``molexp init``.
"""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from molexp.cli import app

runner = CliRunner()


def test_info_errors_on_non_workspace_path(tmp_path: Path):
    result = runner.invoke(app, ["info", "-t", str(tmp_path)])
    assert result.exit_code == 1
    assert "No workspace found at" in result.output
    assert "molexp init" in result.output


def test_info_errors_in_non_workspace_cwd(tmp_path: Path):
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["info"])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 1
    assert "No workspace found at" in result.output


def test_info_succeeds_on_initialized_workspace(tmp_path: Path):
    init_result = runner.invoke(app, ["init", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.output
    result = runner.invoke(app, ["info", "-t", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Workspace:" in result.output
