"""The seven board-state plan tools + the ``BOARD_TOOLS`` descriptor tuple.

Three read tools (``list_tasks`` / ``inspect_task`` / ``inspect_artifact``) never
touch the board's write surface; four write tools (``update_task`` /
``complete_task`` / ``block_task`` / ``propose_plan_patch``) each route through
EXACTLY ONE :class:`TaskBoardHandle` immutable-write and return the outcome as a
:class:`PlanToolResult` — the input board is never mutated.

Every tool takes the uniform keyword-only ``(*, ctx, board, …)`` signature so the
adapter can wrap them uniformly. ``inspect_artifact`` reads through
``ctx.artifact_store`` rather than the board, but keeps the same shape. Missing
ids surface as typed errors (``inspect_task`` lets the board's lookup error
propagate; ``inspect_artifact`` lets :class:`ArtifactNotFoundError` propagate) —
never a silent ``None``.

``BOARD_TOOLS`` exposes the seven functions as :class:`PlanTool` descriptors,
each declaring ``side_effects == []`` (read/write-through-immutable-copy is not a
destructive side effect on the harness store, so the approval gate is bypassed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.harness.plan_tools.tool import PlanTool, PlanToolResult

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.plan_tools.board import TaskBoardHandle

__all__ = [
    "BOARD_TOOLS",
    "block_task",
    "complete_task",
    "inspect_artifact",
    "inspect_task",
    "list_tasks",
    "propose_plan_patch",
    "update_task",
]


async def list_tasks(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001 — uniform plan-tool signature; reads the board only
    board: TaskBoardHandle,
) -> PlanToolResult:
    """List every task on the board (read-only).

    Args:
        ctx: The harness run context (unused; kept for the uniform signature).
        board: The task board to read.

    Returns:
        A successful :class:`PlanToolResult` carrying the task count.
    """
    tasks = board.list_tasks()
    return PlanToolResult(
        ok=True, summary=f"{len(tasks)} task(s) on the board", data={"count": len(tasks)}
    )


async def inspect_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001 — uniform plan-tool signature; reads the board only
    board: TaskBoardHandle,
    task_id: str,
) -> PlanToolResult:
    """Inspect one task by id (read-only).

    Args:
        ctx: The harness run context (unused; kept for the uniform signature).
        board: The task board to read.
        task_id: The id of the task to inspect.

    Returns:
        A successful :class:`PlanToolResult` describing the task.

    Raises:
        Exception: Whatever the board's ``get_task`` raises for an unknown id
            (typically ``KeyError``) — propagated, never swallowed to ``None``.
    """
    task = board.get_task(task_id)
    return PlanToolResult(
        ok=True,
        summary=f"inspected task {task_id!r}",
        data={"task_id": task_id, "task": repr(task)},
    )


async def inspect_artifact(
    *,
    ctx: HarnessRunContext,
    board: TaskBoardHandle,  # noqa: ARG001 — uniform plan-tool signature; reads the artifact store
    artifact_id: str,
) -> PlanToolResult:
    """Inspect one persisted artifact by id (read-only, via the artifact store).

    Args:
        ctx: The harness run context whose ``artifact_store`` is read.
        board: The task board (unused; kept for the uniform signature).
        artifact_id: The id of the artifact to read.

    Returns:
        A successful :class:`PlanToolResult` carrying the artifact byte size.

    Raises:
        ArtifactNotFoundError: If ``artifact_id`` is unknown — propagated, never
            swallowed to ``None``.
    """
    raw = ctx.artifact_store.get(artifact_id)
    return PlanToolResult(
        ok=True,
        summary=f"inspected artifact {artifact_id!r}",
        data={"artifact_id": artifact_id, "bytes": len(raw)},
    )


async def update_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001 — uniform plan-tool signature; writes through the board
    board: TaskBoardHandle,
    task_id: str,
    changes: dict[str, object],
) -> PlanToolResult:
    """Apply ``changes`` to one task via the board's immutable write.

    Calls exactly one ``board.with_task_updated(...)`` and returns success; the
    input board is never mutated.

    Args:
        ctx: The harness run context (unused; kept for the uniform signature).
        board: The task board to write through.
        task_id: The id of the task to update.
        changes: The field changes to apply.

    Returns:
        A successful :class:`PlanToolResult`.
    """
    board.with_task_updated(task_id, changes)
    return PlanToolResult(ok=True, summary=f"updated task {task_id!r}", data={"task_id": task_id})


async def complete_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001 — uniform plan-tool signature; writes through the board
    board: TaskBoardHandle,
    task_id: str,
) -> PlanToolResult:
    """Mark one task complete via the board's immutable write.

    Calls exactly one ``board.mark_complete(...)`` and returns success; the input
    board is never mutated.

    Args:
        ctx: The harness run context (unused; kept for the uniform signature).
        board: The task board to write through.
        task_id: The id of the task to complete.

    Returns:
        A successful :class:`PlanToolResult`.
    """
    board.mark_complete(task_id)
    return PlanToolResult(ok=True, summary=f"completed task {task_id!r}", data={"task_id": task_id})


async def block_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001 — uniform plan-tool signature; writes through the board
    board: TaskBoardHandle,
    task_id: str,
    reason: str,
) -> PlanToolResult:
    """Mark one task blocked via the board's immutable write.

    Calls exactly one ``board.mark_blocked(...)`` and returns success; the input
    board is never mutated.

    Args:
        ctx: The harness run context (unused; kept for the uniform signature).
        board: The task board to write through.
        task_id: The id of the task to block.
        reason: The reason the task is blocked.

    Returns:
        A successful :class:`PlanToolResult`.
    """
    board.mark_blocked(task_id, reason)
    return PlanToolResult(
        ok=True, summary=f"blocked task {task_id!r}", data={"task_id": task_id, "reason": reason}
    )


async def propose_plan_patch(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001 — uniform plan-tool signature; writes through the board
    board: TaskBoardHandle,
    patch: dict[str, object],
) -> PlanToolResult:
    """Apply a plan-level ``patch`` via the board's immutable write.

    Calls exactly one ``board.apply_patch(...)`` and returns success; the input
    board is never mutated.

    Args:
        ctx: The harness run context (unused; kept for the uniform signature).
        board: The task board to write through.
        patch: The plan patch to apply.

    Returns:
        A successful :class:`PlanToolResult`.
    """
    board.apply_patch(patch)
    return PlanToolResult(ok=True, summary="applied plan patch", data={"patch": patch})


# The seven board tools as PlanTool descriptors. Each declares no side effects,
# so wrapping any of them via ``as_loop_tool`` bypasses the approval gate.
BOARD_TOOLS: tuple[PlanTool, ...] = (
    PlanTool(
        name="list_tasks",
        description="List every task on the board.",
        fn=list_tasks,
        input_schema={"type": "object", "properties": {}},
        side_effects=[],
    ),
    PlanTool(
        name="inspect_task",
        description="Inspect one task by id.",
        fn=inspect_task,
        input_schema={"type": "object", "properties": {"task_id": {"type": "string"}}},
        side_effects=[],
    ),
    PlanTool(
        name="inspect_artifact",
        description="Inspect one persisted artifact by id.",
        fn=inspect_artifact,
        input_schema={"type": "object", "properties": {"artifact_id": {"type": "string"}}},
        side_effects=[],
    ),
    PlanTool(
        name="update_task",
        description="Apply field changes to one task.",
        fn=update_task,
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}, "changes": {"type": "object"}},
        },
        side_effects=[],
    ),
    PlanTool(
        name="complete_task",
        description="Mark one task complete.",
        fn=complete_task,
        input_schema={"type": "object", "properties": {"task_id": {"type": "string"}}},
        side_effects=[],
    ),
    PlanTool(
        name="block_task",
        description="Mark one task blocked with a reason.",
        fn=block_task,
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}, "reason": {"type": "string"}},
        },
        side_effects=[],
    ),
    PlanTool(
        name="propose_plan_patch",
        description="Apply a plan-level patch.",
        fn=propose_plan_patch,
        input_schema={"type": "object", "properties": {"patch": {"type": "object"}}},
        side_effects=[],
    ),
)
