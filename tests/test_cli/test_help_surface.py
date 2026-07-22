"""Command-tree surface lock (cli-redesign phase 02).

Locks the redesign's user-visible shape at the registration level: every noun
group lives at the top level and the ``workspace`` god-group is gone. This is a
surface guard — the CLI registration contract, not run/execution behavior
(root-inference / ``--local`` semantics are owned by
``tests/test_cli/test_run.py::TestRootInferencePrecedence``).
"""

from __future__ import annotations

from molexp.cli import app

NOUN_GROUPS = ["project", "experiment", "runs", "asset", "target", "session", "config", "mcp"]


def test_all_noun_groups_registered_top_level() -> None:
    groups = {g.name for g in app.registered_groups}
    for noun in NOUN_GROUPS:
        assert noun in groups, f"noun group {noun!r} not at top level"


def test_no_workspace_god_group() -> None:
    groups = {g.name for g in app.registered_groups}
    assert "workspace" not in groups
