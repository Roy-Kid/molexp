"""Data-graph DAG invariant tests.

Spec: .claude/specs/03-molexp-workflow-cycles.md §1
`depends_on` must always be a DAG. Cycles in `depends_on` are rejected at compile,
even when control edges legitimately form loops.
"""

import pytest

from molexp.workflow import Workflow


def test_depends_on_cycle_rejected_when_control_loops_exist():
    """`depends_on` cycle is rejected with a 'data graph' error message even if control
    edges form a legitimate loop.
    """
    from molexp.workflow import CycleError

    wf = Workflow(name="bad-data-cycle", entry="a")

    @wf.task
    async def a(ctx) -> int:
        return 1

    @wf.task(depends_on=["a", "b"])  # data dep on b
    async def x(ctx) -> int:
        return 2

    @wf.task(depends_on=["x"])  # data dep on x — closes a cycle (a → x → b → x …)
    async def b(ctx) -> int:
        return 3

    # Add a legitimate control loop too — must NOT save the build.
    wf.control("a", "x")
    wf.branch("x", "loop", "a")

    with pytest.raises(CycleError) as exc_info:
        wf.build()
    msg = str(exc_info.value)
    assert "data graph" in msg.lower()
    # Hint to use control edges should be present.
    assert "control" in msg.lower()
