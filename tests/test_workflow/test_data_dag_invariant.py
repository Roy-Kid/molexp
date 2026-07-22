"""Data-graph DAG invariant — ``depends_on`` must always be acyclic.

Spec: .claude/specs/03-molexp-workflow-cycles.md §1. A ``depends_on`` cycle is
rejected at compile even when control edges (``wf.control`` / ``wf.branch``)
legitimately form a loop — the data graph and the control graph are distinct.
"""

from __future__ import annotations

import pytest

from molexp.workflow import CycleError, WorkflowCompiler


class TestWorkflowCompiler:
    def test_depends_on_cycle_rejected_even_with_control_loop(self) -> None:
        """A ``depends_on`` cycle raises ``CycleError`` naming the *data graph*
        (and hinting at control edges) even when control edges form a valid
        loop — the control loop must not save the build."""
        wf = WorkflowCompiler(name="bad-data-cycle", entry="a")

        @wf.task
        async def a(ctx) -> int:
            return 1

        @wf.task(depends_on=["a", "b"])  # data dep on b
        async def x(ctx) -> int:
            return 2

        @wf.task(depends_on=["x"])  # data dep on x — closes a cycle (a → x → b → x …)
        async def b(ctx) -> int:
            return 3

        # A legitimate control loop must NOT save the build.
        wf.control("a", "x")
        wf.branch("x", "loop", "a")

        with pytest.raises(CycleError) as exc_info:
            wf.compile()
        msg = str(exc_info.value).lower()
        assert "data graph" in msg
        assert "control" in msg, "the error must hint at using control edges"
