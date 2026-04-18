"""WorkflowStep: trampoline BaseNode for level-based parallel step execution.

A single node class handles all steps by tracking `level_index`.  Each level
contains one or more steps that share no mutual dependencies and can execute
concurrently via asyncio.gather.  After each level it returns
``WorkflowStep(level_index + 1)`` until all levels are done, then returns
``End(state)``.

The graph is:
    WorkflowStep(0) → WorkflowStep(1) → … → End(state)
where each node executes all steps in its level in parallel.
"""

from __future__ import annotations

import asyncio
from mollog import get_logger
from dataclasses import dataclass
from typing import Any

from pydantic_graph import BaseNode, End, GraphRunContext

from .state import WorkflowDeps, WorkflowState

logger = get_logger(__name__)


@dataclass
class WorkflowStep(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    """Trampoline node that executes one **level** of workflow steps per invocation.

    level_index: index into the level list computed by the compiler.
    Each level contains steps that can run in parallel.
    """

    level_index: int = 0

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> "WorkflowStep | End[WorkflowState]":
        levels: list[list[_StepEntry]] = ctx.deps.levels  # type: ignore[attr-defined]

        if self.level_index >= len(levels):
            return End(ctx.state)

        level = levels[self.level_index]

        if len(level) == 1:
            # Fast path: single step, no gather overhead
            new_state = await self._execute_single(level[0], ctx)
        else:
            # Parallel path: run all steps in this level concurrently
            new_state = await self._execute_parallel(level, ctx)

        if new_state.failed:
            ctx.state._sync_from(new_state)
            return End(ctx.state)

        # Sync the updated state into ctx.state so pydantic-graph snapshots it.
        ctx.state._sync_from(new_state)

        return WorkflowStep(level_index=self.level_index + 1)

    async def _execute_single(
        self, entry: _StepEntry, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> WorkflowState:
        inputs = _gather_inputs(entry, ctx.state)

        from ..context import TaskContext

        step_ctx = TaskContext(
            state=ctx.state,
            deps=ctx.deps.user_deps,
            inputs=inputs,
            config=ctx.deps.config,
            run_context=ctx.deps.run_context,
        )

        logger.debug(f"Executing step {entry.name!r}")

        try:
            output = await _call_task(entry, step_ctx, deps=ctx.deps)
            return ctx.state.record(entry.name, output)
        except Exception as exc:
            logger.exception(f"Step {entry.name!r} failed")
            return ctx.state.fail(entry.name, exc)

    async def _execute_parallel(
        self,
        entries: list[_StepEntry],
        ctx: GraphRunContext[WorkflowState, WorkflowDeps],
    ) -> WorkflowState:
        from ..context import TaskContext

        async def _run_one(entry: _StepEntry) -> tuple[str, Any, Exception | None]:
            inputs = _gather_inputs(entry, ctx.state)
            step_ctx = TaskContext(
                state=ctx.state,
                deps=ctx.deps.user_deps,
                inputs=inputs,
                config=ctx.deps.config,
                run_context=ctx.deps.run_context,
            )
            try:
                output = await _call_task(entry, step_ctx, deps=ctx.deps)
                return (entry.name, output, None)
            except Exception as exc:
                logger.exception(f"Step {entry.name!r} failed")
                return (entry.name, None, exc)

        logger.debug(
            f"Executing level {self.level_index} in parallel: "
            f"{[e.name for e in entries]}"
        )

        results = await asyncio.gather(*[_run_one(e) for e in entries])

        new_state = ctx.state
        for step_name, output, exc in results:
            if exc is not None:
                return new_state.fail(step_name, exc)
            new_state = new_state.record(step_name, output)

        return new_state


def _gather_inputs(entry: _StepEntry, state: WorkflowState) -> Any:
    """Collect inputs from all upstream dependencies.

    Returns:
        - ``None`` if the step has no dependencies.
        - The single upstream output if there is exactly one dependency.
        - A ``dict[dep_name → output]`` if there are multiple dependencies.
    """
    if not entry.depends_on:
        return None

    if len(entry.depends_on) == 1:
        return state.step_outputs.get(entry.depends_on[0])

    return {
        dep: state.step_outputs.get(dep)
        for dep in entry.depends_on
    }


async def _call_task(
    entry: "_StepEntry", task_ctx: Any, deps: WorkflowDeps | None = None
) -> Any:
    """Dispatch to local execution or remote submission.

    Resolution order:
    1. Remote execution (if ``entry.remote`` is set and executor available)
    2. :class:`~Runnable` protocol (has ``.execute()``)
    3. Bare callable (function decorated with ``@wf.task``)
    """
    # 1. Remote gate
    if entry.remote is not None and deps is not None:
        remote_executor = getattr(deps, "remote_executor", None)
        if remote_executor is not None:
            run_dir = getattr(deps, "run_dir", None)
            return await remote_executor.execute_remote(
                entry=entry,
                inputs=task_ctx.inputs,
                run_dir=run_dir,
            )

    # 2. Protocol-based dispatch (Runnable — any object with .execute())
    from ..protocols import Runnable

    if isinstance(entry.fn_or_class, Runnable):
        return await entry.fn_or_class.execute(task_ctx)

    # 3. Bare callable (function)
    if callable(entry.fn_or_class):
        return await entry.fn_or_class(task_ctx)

    raise TypeError(
        f"Task '{entry.name}' is neither Runnable nor callable: "
        f"{type(entry.fn_or_class)}"
    )


@dataclass
class _StepEntry:
    """Internal descriptor for a single step held in WorkflowDeps."""

    name: str
    fn_or_class: Any
    depends_on: list[str]
    is_actor: bool = False
    remote: Any = None
