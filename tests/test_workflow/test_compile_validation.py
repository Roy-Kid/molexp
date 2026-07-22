"""Compile-time validation: control-edge shape, entry, and reachability.

Each test locks one distinct ``WorkflowCompiler.compile()`` rejection rule
(a distinct error type + trigger); structural deadlock lives in
``test_deadlock_guard``. Spec: .claude/specs/03-molexp-workflow-cycles.md.
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    EdgeShapeError,
    EntryAmbiguousError,
    UnknownTaskError,
    UnreachableTaskError,
    WorkflowCompiler,
)


class TestCompileValidation:
    def test_mixed_unconditional_and_branch_out_edges_rejected(self) -> None:
        """Mixing an unconditional control edge and a branch out-edge on the same
        node fails with ``EdgeShapeError`` naming the node."""
        wf = WorkflowCompiler(name="mixed", entry="src")

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
            wf.compile()
        assert "src" in str(exc_info.value)

    def test_control_edges_without_entry_declaration_rejected(self) -> None:
        """Control edges but no ``wf.entry(...)`` ⇒ ``EntryAmbiguousError`` that
        names the remedy and the candidate entries."""
        wf = WorkflowCompiler(name="no-entry")

        @wf.task
        async def a(ctx) -> int:
            return 1

        @wf.task
        async def b(ctx) -> int:
            return 2

        wf.control("a", "b")

        with pytest.raises(EntryAmbiguousError) as exc_info:
            wf.compile()
        msg = str(exc_info.value)
        assert "wf.entry" in msg
        assert "a" in msg  # candidate list reported

    def test_entry_referencing_unknown_task_rejected(self) -> None:
        """``wf.entry("ghost")`` referencing an unregistered task ⇒ ``UnknownTaskError``."""
        wf = WorkflowCompiler(name="bad-entry")

        @wf.task
        async def real_task(ctx) -> int:
            return 1

        wf.entry("ghost")

        with pytest.raises(UnknownTaskError) as exc_info:
            wf.compile()
        assert "ghost" in str(exc_info.value)

    def test_task_unreachable_from_entry_rejected(self) -> None:
        """A registered task with no control path from any entry ⇒ ``UnreachableTaskError``."""
        wf = WorkflowCompiler(name="unreachable", entry="a")

        @wf.task
        async def a(ctx) -> int:
            return 1

        @wf.task
        async def b(ctx) -> int:
            return 2

        @wf.task
        async def orphan(ctx) -> int:
            return 99

        wf.control("a", "b")

        with pytest.raises(UnreachableTaskError) as exc_info:
            wf.compile()
        assert "orphan" in str(exc_info.value)
