"""RED tests for the harness->agent tool adapter (plan-emergent-03, ac-008).

Pins ``molexp.harness.plan_tools.as_loop_tool(tool, *, ctx, approve=None)`` — the
adapter that turns a :class:`PlanTool` into an opaque async callable an agent
loop can accept (``__name__`` / ``__doc__`` sourced from the PlanTool so
pydantic-ai can derive a schema), attaches a before-tool side-effect gate
(``enforce_side_effect_approvals`` on ``tool.side_effects``; empty => bypass, no
gate, no approval artifact), and an after-tool hook recording the outcome on
``ctx.event_log``.

Function-level only: the PlanTools here carry trivial in-memory ``fn`` bodies;
no board, no subprocess, no server.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.plan_tools import PlanTool, PlanToolResult, as_loop_tool
from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

# ──────────────────────────────────────────────────────────── fixtures / helpers


def _make_ctx(root: Path) -> HarnessRunContext:
    """Build a ``HarnessRunContext`` backed by isolated on-disk stores."""
    db_path = root / "events.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    events = SQLiteEventLog(path=db_path)
    lineage = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-plan-tools-adapter",
        workspace_root=root,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
    )


async def denying_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Reject every request — if reached on a read-only tool, the bypass failed."""
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="test",
        decided_at=datetime.now(tz=UTC),
        reason="denied",
    )


class TestAsLoopToolMetadata:
    """The adapter yields an opaque async callable carrying PlanTool metadata."""

    def test_returns_async_callable_with_planTool_name_and_doc(self, tmp_path: Path) -> None:
        async def fn(**kwargs: object) -> PlanToolResult:
            return PlanToolResult(ok=True, summary="", data={}, artifact_ref=None)

        tool = PlanTool(
            name="list_tasks",
            description="List all tasks on the board.",
            input_schema={"type": "object"},
            side_effects=[],
            fn=fn,
        )
        ctx = _make_ctx(tmp_path)

        adapted = as_loop_tool(tool, ctx=ctx)

        assert inspect.iscoroutinefunction(adapted)
        assert adapted.__name__ == "list_tasks"
        assert adapted.__doc__ == "List all tasks on the board."


class TestAsLoopToolReadOnlyBypass:
    """A read-only PlanTool runs its body with no gate and no approval artifact."""

    async def test_read_only_tool_runs_body_without_gating(self, tmp_path: Path) -> None:
        ran: list[bool] = []

        async def fn(**kwargs: object) -> PlanToolResult:
            ran.append(True)
            return PlanToolResult(ok=True, summary="done", data={}, artifact_ref=None)

        tool = PlanTool(
            name="inspect_task",
            description="Inspect one task.",
            input_schema={"type": "object"},
            side_effects=[],
            fn=fn,
        )
        ctx = _make_ctx(tmp_path)
        # A denying approver would RAISE if any gate ran; reaching the body and
        # writing no approval artifact proves the read-only bypass.
        adapted = as_loop_tool(tool, ctx=ctx, approve=denying_approver)

        result = await adapted()

        assert ran == [True]
        assert isinstance(result, PlanToolResult)
        assert ctx.artifact_store.list_by_kind("side_effect_approval") == []

    async def test_read_only_after_hook_records_tool_outcome_event(self, tmp_path: Path) -> None:
        async def fn(**kwargs: object) -> PlanToolResult:
            return PlanToolResult(ok=True, summary="done", data={}, artifact_ref=None)

        tool = PlanTool(
            name="list_tasks",
            description="List tasks.",
            input_schema={"type": "object"},
            side_effects=[],
            fn=fn,
        )
        ctx = _make_ctx(tmp_path)
        adapted = as_loop_tool(tool, ctx=ctx)

        await adapted()

        events = ctx.event_log.list_events(ctx.run_id)
        # The read-only path runs no gate, so any recorded event is the after-hook.
        assert any(event.type.startswith("tool_") for event in events)


class TestAsLoopToolSideEffectGate:
    """A side-effect PlanTool copies side_effects for the host pipeline."""

    async def test_side_effect_tool_copies_effects_and_runs_body(self, tmp_path: Path) -> None:
        async def fn(**kwargs: object) -> PlanToolResult:
            return PlanToolResult(ok=True, summary="did it", data={}, artifact_ref=None)

        tool = PlanTool(
            name="run_delete",
            description="Delete something.",
            input_schema={"type": "object"},
            side_effects=["delete"],
            fn=fn,
        )
        ctx = _make_ctx(tmp_path)
        adapted = as_loop_tool(tool, ctx=ctx)

        result = await adapted()

        assert adapted.side_effects == ["delete"]
        assert isinstance(result, PlanToolResult)
        assert ctx.artifact_store.list_by_kind("side_effect_approval") == []
