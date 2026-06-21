"""``molexp info`` fails loudly when the path holds no workspace.

A default-constructed ``Workspace`` used to make ``info`` print a
healthy-looking empty workspace for any directory; now a missing
``workspace.json`` is a hard error pointing at ``molexp init``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from typer.testing import CliRunner

from molexp.cli import app

runner = CliRunner()


def _plain(output: str) -> str:
    """Strip ANSI codes and collapse whitespace.

    Rich wraps long lines at the (narrow) CliRunner terminal width, so a
    phrase like ``molexp init`` can land split across a newline; normalizing
    whitespace makes the substring assertions wrap-insensitive.
    """
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", output))


def test_info_errors_on_non_workspace_path(tmp_path: Path):
    result = runner.invoke(app, ["info", "-t", str(tmp_path)])
    assert result.exit_code == 1
    plain = _plain(result.output)
    assert "No workspace found at" in plain
    assert "molexp init" in plain


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
