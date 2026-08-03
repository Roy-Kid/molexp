"""Board-state plan tools + the ``BOARD_TOOLS`` descriptor tuple.

Read tools never write; write tools each route through exactly one
:class:`TaskBoardHandle` immutable-write and return a :class:`PlanToolResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    "place_task",
    "propose_plan_patch",
    "update_task",
]


async def list_tasks(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001
    board: TaskBoardHandle,
) -> PlanToolResult:
    """List every task on the board with structured fields."""
    tasks = board.list_tasks()
    return PlanToolResult(
        ok=True,
        summary=f"{len(tasks)} task(s) on the board",
        data={"count": len(tasks), "tasks": list(tasks)},
    )


async def inspect_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001
    board: TaskBoardHandle,
    task_id: str,
) -> PlanToolResult:
    """Inspect one task by id (structured payload)."""
    task = board.get_task(task_id)
    payload: dict[str, Any]
    if isinstance(task, dict):
        payload = {str(key): value for key, value in task.items()}
    else:
        payload = {"task": task}
    return PlanToolResult(
        ok=True,
        summary=f"inspected task {task_id!r}",
        data={"task_id": task_id, "task": payload},
    )


async def inspect_artifact(
    *,
    ctx: HarnessRunContext,
    board: TaskBoardHandle,  # noqa: ARG001
    artifact_id: str,
) -> PlanToolResult:
    """Inspect one persisted artifact by id via the artifact store."""
    raw = ctx.artifact_store.get(artifact_id)
    return PlanToolResult(
        ok=True,
        summary=f"inspected artifact {artifact_id!r}",
        data={"artifact_id": artifact_id, "bytes": len(raw)},
    )


async def place_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001
    board: TaskBoardHandle,
    task_id: str,
    name: str,
    acceptance: list[str] | None = None,
) -> PlanToolResult:
    """Upsert a task onto the board (create or replace by id)."""
    board.place(task_id, name, acceptance=acceptance)
    return PlanToolResult(
        ok=True,
        summary=f"placed task {task_id!r}",
        data={
            "task_id": task_id,
            "name": name,
            "acceptance": list(acceptance or ()),
        },
    )


async def update_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001
    board: TaskBoardHandle,
    task_id: str,
    changes: dict[str, object],
) -> PlanToolResult:
    """Apply ``changes`` to one task via the board's immutable write."""
    board.with_task_updated(task_id, changes)
    return PlanToolResult(ok=True, summary=f"updated task {task_id!r}", data={"task_id": task_id})


async def complete_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001
    board: TaskBoardHandle,
    task_id: str,
) -> PlanToolResult:
    """Mark one task complete."""
    board.mark_complete(task_id)
    return PlanToolResult(ok=True, summary=f"completed task {task_id!r}", data={"task_id": task_id})


async def block_task(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001
    board: TaskBoardHandle,
    task_id: str,
    reason: str,
) -> PlanToolResult:
    """Mark one task blocked with a reason."""
    board.mark_blocked(task_id, reason)
    return PlanToolResult(
        ok=True, summary=f"blocked task {task_id!r}", data={"task_id": task_id, "reason": reason}
    )


async def propose_plan_patch(
    *,
    ctx: HarnessRunContext,  # noqa: ARG001
    board: TaskBoardHandle,
    patch: dict[str, object],
) -> PlanToolResult:
    """Apply a plan-level patch (place/update/remove tasks)."""
    board.apply_patch(patch)
    return PlanToolResult(ok=True, summary="applied plan patch", data={"patch": patch})


BOARD_TOOLS: tuple[PlanTool, ...] = (
    PlanTool(
        name="list_tasks",
        description="List every task on the plan board with id, name, status, and acceptance.",
        fn=list_tasks,
        input_schema={"type": "object", "properties": {}},
        side_effects=[],
    ),
    PlanTool(
        name="inspect_task",
        description="Inspect one task by id (structured fields).",
        fn=inspect_task,
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        side_effects=[],
    ),
    PlanTool(
        name="inspect_artifact",
        description="Inspect one persisted harness artifact by id.",
        fn=inspect_artifact,
        input_schema={
            "type": "object",
            "properties": {"artifact_id": {"type": "string"}},
            "required": ["artifact_id"],
        },
        side_effects=[],
    ),
    PlanTool(
        name="place_task",
        description=(
            "Create or replace a task on the board. "
            "Provide task_id, a human name, and acceptance criteria strings."
        ),
        fn=place_task,
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "name": {"type": "string"},
                "acceptance": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["task_id", "name"],
        },
        side_effects=[],
    ),
    PlanTool(
        name="update_task",
        description="Apply field changes to one task (name, acceptance, status).",
        fn=update_task,
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "changes": {"type": "object"},
            },
            "required": ["task_id", "changes"],
        },
        side_effects=[],
    ),
    PlanTool(
        name="complete_task",
        description="Mark one task complete.",
        fn=complete_task,
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        side_effects=[],
    ),
    PlanTool(
        name="block_task",
        description="Mark one task blocked with a reason.",
        fn=block_task,
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["task_id", "reason"],
        },
        side_effects=[],
    ),
    PlanTool(
        name="propose_plan_patch",
        description=(
            "Apply a plan-level patch. Keys: tasks (list of {id,name,acceptance}), "
            "update (map task_id→changes), remove (list of task ids)."
        ),
        fn=propose_plan_patch,
        input_schema={
            "type": "object",
            "properties": {"patch": {"type": "object"}},
            "required": ["patch"],
        },
        side_effects=[],
    ),
)
