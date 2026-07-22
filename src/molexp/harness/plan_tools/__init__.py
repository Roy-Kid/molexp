"""``molexp.harness.plan_tools`` — the process-internal plan-tool surface.

A stable import target (``molexp.harness.plan_tools``) that is deliberately NOT
re-exported from ``molexp.harness.__all__``. It bundles:

* :class:`PlanTool` / :class:`PlanToolResult` — the tool descriptor + payload.
* :class:`TaskBoardHandle` — the immutable task-board Protocol.
* the seven board-state tools + :data:`BOARD_TOOLS` descriptors.
* :func:`run_capability` / :func:`run_acceptance_test` — side-effecting tools that
  route through ``dispatch_capability``.
* :func:`as_loop_tool` — the :class:`PlanTool` → agent-loop-callable adapter.
"""

from __future__ import annotations

from molexp.harness.plan_tools.adapter import as_loop_tool
from molexp.harness.plan_tools.board import TaskBoardHandle
from molexp.harness.plan_tools.board_tools import (
    BOARD_TOOLS,
    block_task,
    complete_task,
    inspect_artifact,
    inspect_task,
    list_tasks,
    propose_plan_patch,
    update_task,
)
from molexp.harness.plan_tools.capability_tools import run_acceptance_test, run_capability
from molexp.harness.plan_tools.tool import PlanTool, PlanToolResult

__all__ = [
    "BOARD_TOOLS",
    "PlanTool",
    "PlanToolResult",
    "TaskBoardHandle",
    "as_loop_tool",
    "block_task",
    "complete_task",
    "inspect_artifact",
    "inspect_task",
    "list_tasks",
    "propose_plan_patch",
    "run_acceptance_test",
    "run_capability",
    "update_task",
]
