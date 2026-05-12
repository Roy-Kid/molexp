"""Compile-time validation tests for control edges, entry, reachability, deadlock.

Spec: .claude/specs/03-molexp-workflow-cycles.md §3, §4, §8, §9
"""

import pytest

from molexp.workflow import WorkflowBuilder


def test_mixed_route_shape_rejected():
    """Mixing unconditional + branch out-edges on the same node fails at compile."""
    from molexp.workflow import EdgeShapeError

    wf = WorkflowBuilder(name="mixed", entry="src")

    @wf.task
    async def src(ctx) -> int:
        return 1

    @wf.task
    async def a(ctx) -> int:
        return 2

    @wf.task
    async def b(ctx) -> int:
        return 3

    wf.control("src", "a")  # unconditional
    wf.branch("src", "x", "b")  # branch on the same node

    with pytest.raises(EdgeShapeError) as exc_info:
        wf.build()
    assert "src" in str(exc_info.value)


def test_control_workflow_without_entry_rejected():
    """A workflow with control edges but no `wf.entry(...)` declaration fails at compile."""
    from molexp.workflow import EntryAmbiguousError

    wf = WorkflowBuilder(name="no-entry")

    @wf.task
    async def a(ctx) -> int:
        return 1

    @wf.task
    async def b(ctx) -> int:
        return 2

    wf.control("a", "b")
    # Note: no wf.entry(...) call.

    with pytest.raises(EntryAmbiguousError) as exc_info:
        wf.build()
    msg = str(exc_info.value)
    assert "wf.entry" in msg
    # Candidate list should be reported.
    assert "a" in msg


def test_entry_unknown_task_rejected():
    """`wf.entry("missing")` referencing an unknown task fails at compile."""
    from molexp.workflow import UnknownTaskError

    wf = WorkflowBuilder(name="bad-entry")

    @wf.task
    async def real_task(ctx) -> int:
        return 1

    wf.entry("ghost")  # ghost not registered

    with pytest.raises(UnknownTaskError) as exc_info:
        wf.build()
    assert "ghost" in str(exc_info.value)


def test_unreachable_task_rejected():
    """A task unreachable from any entry through control edges fails at compile."""
    from molexp.workflow import UnreachableTaskError

    wf = WorkflowBuilder(name="unreachable", entry="a")

    @wf.task
    async def a(ctx) -> int:
        return 1

    @wf.task
    async def b(ctx) -> int:
        return 2

    @wf.task
    async def orphan(ctx) -> int:  # no incoming edge from any entry
        return 99

    wf.control("a", "b")
    # `orphan` is registered but never reachable.

    with pytest.raises(UnreachableTaskError) as exc_info:
        wf.build()
    assert "orphan" in str(exc_info.value)
