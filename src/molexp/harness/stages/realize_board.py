"""``RealizeBoard`` — deterministic map -> reduce -> compile of a task board.

Plan-emergent-06's realization stage. Given a *frozen* :class:`BoundWorkflow`
(the board), it:

1. **Maps** :func:`~molexp.harness.stages.realize_task.realize_one_task` over
   EVERY :class:`BoundTask` concurrently (``asyncio.gather``) — full coverage is
   guaranteed by structure, not chosen by a model. Each worker self-repairs its
   task against its own isolated source tree.
2. **Reduces** the greens into ONE ``workflow_source`` (the deterministic
   ``build_workflow`` assembly + one per-task module file) and ONE
   ``test_source``.
3. **Blocks** — if >= 1 task exhausts its budget, it persists a durable
   ``intervention_request`` artifact and raises
   :class:`TaskRealizationBlockedError` BEFORE any compile. It never
   assembles, compiles, or falls back on a block: an incomplete board waits for
   a human decision.
4. **Compiles** (all-green only) via ``MaterializeExecution`` then
   ``CompileWorkflow`` (``--compile-only``), returning the compile
   ``execution_result`` ref.

Like ``CompileWorkflow`` / ``MaterializeExecution`` this stage never imports
``molexp.workflow`` — the engine loads only inside the executor subprocess.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, TaskRealizationBlockedError
from molexp.harness.schemas import (
    BlockedTask,
    BoundWorkflow,
    GeneratedFile,
    InterventionRequest,
    PlanArtifactRef,
    TestSource,
    TestSpec,
    TestSpecBundle,
    WorkflowSource,
)
from molexp.harness.stages._resolve import require_latest
from molexp.harness.stages.compile_workflow import CompileWorkflow
from molexp.harness.stages.generate_workflow_source import _render_assembly, _slug
from molexp.harness.stages.materialize_execution import MaterializeExecution
from molexp.harness.stages.realize_task import TaskRealization, realize_one_task
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = ["RealizeBoard"]


class RealizeBoard(Stage):
    """Map -> reduce -> compile realizer of a frozen task board."""

    name: ClassVar[str] = "realize_board"

    def __init__(self, executor: Executor, *, attempts: int = 3, timeout_s: int = 300) -> None:
        if attempts < 1:
            raise ValueError("RealizeBoard attempts must be >= 1")
        self._executor = executor
        self._attempts = attempts
        self._timeout_s = timeout_s

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("RealizeBoard requires ctx.agent_gateway")
        bound_ref = require_latest(ctx, "bound_workflow", stage=self.name)
        experiment_spec = require_latest(ctx, "experiment_spec", stage=self.name)
        bound = BoundWorkflow.model_validate_json(ctx.artifact_store.get(bound_ref.id))
        if not bound.tasks:
            raise StageExecutionError("RealizeBoard: bound_workflow has no tasks")

        # ── Map: one self-repairing worker per task, in parallel ───────────
        results = await asyncio.gather(
            *[
                realize_one_task(
                    ctx,
                    task=task,
                    bound=bound,
                    bound_ref=bound_ref,
                    experiment_spec_id=experiment_spec.id,
                    executor=self._executor,
                    attempts=self._attempts,
                    timeout_s=self._timeout_s,
                )
                for task in bound.tasks
            ]
        )

        # ── Block branch: persist-then-raise, never compile on a block ─────
        blocked = [r.blocked for r in results if r.blocked is not None]
        if blocked:
            request_ref = self._persist_intervention(
                ctx, bound=bound, bound_ref=bound_ref, blocked=blocked
            )
            raise TaskRealizationBlockedError(request_ref, [bt.task_id for bt in blocked])

        # ── Reduce: assemble the greens into workflow_source + test_source ─
        slugs = {task.id: _slug(task.ir_task_id or task.id) for task in bound.tasks}
        self._publish_reduced(ctx, bound=bound, bound_ref=bound_ref, slugs=slugs, results=results)

        # ── Compile: materialize the tree + --compile-only dry run ─────────
        await MaterializeExecution().run(ctx)
        return await CompileWorkflow(self._executor).run(ctx)

    def _persist_intervention(
        self,
        ctx: HarnessRunContext,
        *,
        bound: BoundWorkflow,
        bound_ref: PlanArtifactRef,
        blocked: list[BlockedTask],
    ) -> PlanArtifactRef:
        """Write the durable ``intervention_request`` before the raise."""
        request = InterventionRequest(
            id=f"ivr-{generate_id()}",
            bound_workflow_id=bound.id,
            blocked_tasks=tuple(blocked),
            reason=f"{len(blocked)} task(s) could not be realized",
        )
        return ctx.artifact_store.put_json(
            kind="intervention_request",
            obj=request.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[bound_ref.id],
        )

    def _publish_reduced(
        self,
        ctx: HarnessRunContext,
        *,
        bound: BoundWorkflow,
        bound_ref: PlanArtifactRef,
        slugs: dict[str, str],
        results: list[TaskRealization],
    ) -> None:
        """Reduce the per-task greens into one ``workflow_source`` + ``test_source``.

        Mirrors ``SequentialTaskBuild._publish_cumulative``'s output shape: the
        deterministic ``build_workflow`` assembly as ``source``, one per-task
        module in ``files``, and a matching ``TestSpecBundle`` + ``TestSource``.
        """
        module_by_task = {
            task.id: (r.module_src or "") for task, r in zip(bound.tasks, results, strict=True)
        }
        test_by_task = {
            task.id: (r.test_src or "") for task, r in zip(bound.tasks, results, strict=True)
        }
        wf_files = [
            GeneratedFile(path=f"workflow/{slugs[task.id]}.py", source=module_by_task[task.id])
            for task in bound.tasks
        ]
        wf = WorkflowSource(
            source=_render_assembly(bound, slugs),
            module_name="workflow",
            bound_workflow_id=bound.id,
            symbols=("build_workflow",),
            files=wf_files,
        )
        ctx.artifact_store.put_json(
            kind="workflow_source",
            obj=wf.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[bound_ref.id],
        )

        specs = [
            TestSpec(
                id=f"ts-{slugs[task.id]}",
                name=f"test_{slugs[task.id]}",
                kind="unit_test",
                target_task_id=task.ir_task_id or task.id,
                description=f"unit tests for {slugs[task.id]}",
            )
            for task in bound.tasks
        ]
        bundle = TestSpecBundle(id=f"tsb-{bound.id}", bound_workflow_id=bound.id, specs=specs)
        ts_ref = ctx.artifact_store.put_json(
            kind="test_spec",
            obj=bundle.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[bound_ref.id],
        )
        test_files = [
            GeneratedFile(path=f"tests/test_{slugs[task.id]}.py", source=test_by_task[task.id])
            for task in bound.tasks
        ]
        test_src = TestSource(
            source="",
            module_name="test_suite",
            test_spec_id=bundle.id,
            bound_workflow_id=bound.id,
            files=test_files,
        )
        ctx.artifact_store.put_json(
            kind="test_source",
            obj=test_src.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[ts_ref.id, bound_ref.id],
        )
