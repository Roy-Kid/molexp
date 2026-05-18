"""Tests for internal helpers in ``molexp.cli.run_cmd``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from molexp.cli.run_cmd import _watch_path_for


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_run(workspace_root: Path) -> SimpleNamespace:
    """Build the minimal ``run.experiment.project.workspace.root`` chain."""
    return SimpleNamespace(
        experiment=SimpleNamespace(
            project=SimpleNamespace(workspace=SimpleNamespace(root=workspace_root))
        )
    )


def test_no_workspace_and_no_submitted_returns_dot(chdir_tmp):
    assert _watch_path_for(None, []) == "."


def test_explicit_workspace_equal_to_cwd_returns_dot(chdir_tmp):
    assert _watch_path_for(chdir_tmp, []) == "."


def test_explicit_subdir_is_relative(chdir_tmp):
    sub = chdir_tmp / "lab"
    sub.mkdir()
    assert _watch_path_for(sub, []) == "lab"


def test_unrelated_absolute_stays_absolute(chdir_tmp, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    assert _watch_path_for(other, []) == str(other)


def test_derives_from_submitted_when_workspace_is_none(chdir_tmp):
    sub = chdir_tmp / "lab"
    sub.mkdir()
    run = _fake_run(sub)
    assert _watch_path_for(None, [run]) == "lab"


def test_explicit_workspace_wins_over_submitted(chdir_tmp):
    explicit = chdir_tmp / "explicit"
    explicit.mkdir()
    other = chdir_tmp / "other"
    other.mkdir()
    run = _fake_run(other)
    assert _watch_path_for(explicit, [run]) == "explicit"


def test_submitted_without_expected_attrs_falls_back(chdir_tmp):
    bogus = SimpleNamespace()
    assert _watch_path_for(None, [bogus]) == "."
