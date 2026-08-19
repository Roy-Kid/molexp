"""Tests for the dataflow-by-name :class:`molexp.workflow.context.TaskContext`.

``ctx.inputs`` / ``ctx.config`` / ``ctx.state`` and the old workspace-plumbing
surface are gone: a task body receives its inputs as typed parameters bound
by name, and the only data surface left on ``ctx`` is ``workdir``.
"""

from __future__ import annotations

from molexp.workflow.context import TaskContext


class TestTaskContext:
    def test_workspace_plumbing_surface_is_absent(self):
        """The pre-refactor data/capability surface is gone from ``ctx``."""
        ctx = TaskContext(inputs=None)
        for name in (
            "inputs",
            "config",
            "state",
            "artifact",
            "log",
            "find_asset",
            "checkpoint",
            "set_result",
            "get_result",
        ):
            assert not hasattr(ctx, name), (
                f"TaskContext.{name} must be absent; inputs bind to parameters and "
                f"capabilities flow via the engine's materialization layer."
            )
