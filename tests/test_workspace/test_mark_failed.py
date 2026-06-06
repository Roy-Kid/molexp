"""Typed RunContext.mark_failed (workflow-workspace-hardening P2-1).

The workflow runtime used to record a task failure by reaching into the
workspace-private ``run_context._context.status`` dict with a string
convention. RunContext now exposes a typed ``mark_failed(error)`` on its public
surface (and the ``RunContextLike`` protocol), so the cross-layer contract is
explicit instead of a private-attribute poke.
"""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace
from molexp.workspace.run import RunStatus


@pytest.fixture
def run(tmp_path):
    ws = Workspace(root=tmp_path / "lab", name="lab")
    exp = ws.add_project("p").add_experiment("e")
    return exp.add_run(parameters={"i": 0})


def test_mark_failed_resolves_failed_on_clean_exit(run):
    """Calling mark_failed (no exception propagating) still resolves the run
    status to FAILED on context exit, and records the error message."""
    with run.start() as ctx:
        ctx.mark_failed("task X blew up")
        assert ctx.context.status["run"] == RunStatus.FAILED
        assert ctx.context.errors["run"]["message"] == "task X blew up"

    assert run.status == RunStatus.FAILED


def test_mark_failed_is_in_runcontextlike_protocol():
    """The workflow layer's duck-typed view declares mark_failed."""
    from molexp.workflow.protocols import RunContextLike

    assert hasattr(RunContextLike, "mark_failed")


def test_clean_run_without_mark_failed_succeeds(run):
    with run.start():
        pass
    assert run.status == RunStatus.SUCCEEDED
