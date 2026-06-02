"""Flat command-tree help surface (cli-redesign phase 02).

Asserts the redesign's user-visible payoff: every verb + noun group lives at the
top level, noun groups expose their CRUD at two levels, the ``workspace``
god-group is gone, and a local ``run`` works with no ``-t`` (defaults to cwd).
"""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from molexp.cli import app

runner = CliRunner()

VERBS = [
    "run",
    "serve",
    "monitor",
    "explore",
    "info",
    "exec",
    "shell",
    "sync",
    "push",
    "pull",
    "init",
    "agent",
]
NOUN_GROUPS = ["project", "experiment", "runs", "asset", "target", "session", "config", "mcp"]


def test_all_verbs_registered_top_level():
    for verb in VERBS:
        result = runner.invoke(app, [verb, "--help"])
        assert result.exit_code == 0, f"verb {verb!r} not at top level"


def test_all_noun_groups_registered_top_level():
    groups = {g.name for g in app.registered_groups}
    for noun in NOUN_GROUPS:
        assert noun in groups, f"noun group {noun!r} not at top level"


def test_no_workspace_god_group():
    groups = {g.name for g in app.registered_groups}
    assert "workspace" not in groups
    # legacy `workspace <target> <cmd>` path no longer parses
    assert runner.invoke(app, ["workspace", "info"]).exit_code != 0


def test_noun_groups_show_crud_subcommands():
    expected = {
        "project": ["list", "create", "info"],
        "runs": ["list", "info", "prune"],
        "target": ["list", "add", "remove", "test"],
        "experiment": ["list", "create"],
    }
    for group, subs in expected.items():
        out = runner.invoke(app, [group, "--help"]).output
        for sub in subs:
            assert sub in out, f"{group} --help missing {sub!r}"


def test_local_run_with_no_target(tmp_path: Path):
    script = tmp_path / "entry.py"
    script.write_text(
        "import molexp as me\n"
        "from molexp.workflow import WorkflowCompiler, default_binding_registry\n"
        "ws = me.Workspace('./_ws')\n"
        "e = ws.add_project('p').add_experiment('e')\n"
        "wf = WorkflowCompiler(name='w')\n"
        "@wf.task\n"
        "async def t(ctx):\n    return 1\n"
        "default_binding_registry.bind(e, wf.compile())\n"
        "me.entry(ws)\n"
    )
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["run", str(script), "--local"])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.output
