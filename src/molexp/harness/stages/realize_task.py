"""``realize_one_task`` — the per-task self-repair worker (plan-emergent-06).

The map-body of the realization phase: for ONE frozen :class:`BoundTask`, loop
up to ``attempts`` times — generate the task module + its unit test through the
injected ``AgentGateway``, run that one test through the injected
:class:`Executor`, and on red feed the failure back and regenerate
(``use_mcp=True`` on repair turns). It is a plain coroutine (NOT a ``Stage``) so
:class:`~molexp.harness.stages.realize_board.RealizeBoard` can ``asyncio.gather``
one worker per board task.

The load-bearing behavior is the *return-on-block* contract: at the attempt
ceiling the worker RETURNS a blocked :class:`TaskRealization` (carrying a
:class:`BlockedTask` payload) rather than raising, so the board's map continues
the other tasks and later collects every block into one intervention request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import BlockedTask, BoundWorkflow, PlanArtifactRef
from molexp.harness.stages.generate_workflow_source import _rename_top_function, _slug
from molexp.harness.stages.task_codegen import (
    failure_text,
    gen_task_module,
    gen_task_test,
    run_isolated_task_pytest,
)

if TYPE_CHECKING:
    from molexp.harness.executors import Executor
    from molexp.harness.schemas import BoundTask

__all__ = ["TaskRealization", "realize_one_task"]

_CREATED_BY = "stage:realize_task"


@dataclass(frozen=True, slots=True)
class TaskRealization:
    """Outcome of realizing one task.

    ``status == "green"`` carries the final ``module_src`` + ``test_src`` (both
    ready to reduce into the board's workflow/test source); ``status ==
    "blocked"`` carries a :class:`BlockedTask` and leaves the sources ``None``.
    """

    status: Literal["green", "blocked"]
    module_src: str | None = None
    test_src: str | None = None
    blocked: BlockedTask | None = None


async def realize_one_task(
    ctx: HarnessRunContext,
    *,
    task: BoundTask,
    bound: BoundWorkflow,
    bound_ref: PlanArtifactRef,
    experiment_spec_id: str,
    executor: Executor,
    attempts: int = 3,
    timeout_s: int = 300,
) -> TaskRealization:
    """Codegen + self-repair one bound task to a green unit test, or block it.

    Args:
        ctx: The harness run context (artifact store + agent gateway).
        task: The single :class:`BoundTask` to realize.
        bound: The full :class:`BoundWorkflow` (for inter-task wiring context).
        bound_ref: The ``bound_workflow`` artifact ref (lineage parent).
        experiment_spec_id: The ``experiment_spec`` artifact id fed to codegen.
        executor: Runs the per-task pytest (isolated tree; skipped under dry-run).
        attempts: Max codegen+pytest attempts before blocking (>= 1).
        timeout_s: Per-pytest timeout.

    Returns:
        A green :class:`TaskRealization` with both sources, or — at the ceiling
        — a blocked one carrying a :class:`BlockedTask`. Never raises on a
        persistently-red task.
    """
    ceiling = max(1, attempts)
    slug = _slug(getattr(task, "ir_task_id", None) or task.id)
    slugs = {t.id: _slug(t.ir_task_id or t.id) for t in bound.tasks}
    feedback_ids: list[str] = []
    last_error = ""
    last_error_ref: str | None = None

    for attempt in range(1, ceiling + 1):
        is_repair = attempt > 1
        module_src = _rename_top_function(
            await gen_task_module(
                ctx,
                bound=bound,
                bound_ref=bound_ref,
                task=task,
                experiment_spec_id=experiment_spec_id,
                created_by=_CREATED_BY,
                feedback_ids=feedback_ids,
                is_repair=is_repair,
                slugs=slugs,
            ),
            slug,
        )
        test_src = await gen_task_test(
            ctx,
            bound_ref=bound_ref,
            task=task,
            slug=slug,
            module_src=module_src,
            created_by=_CREATED_BY,
            feedback_ids=feedback_ids,
            is_repair=is_repair,
            bound=bound,
            slugs=slugs,
        )
        try:
            await run_isolated_task_pytest(
                ctx,
                executor=executor,
                slug=slug,
                module_src=module_src,
                test_src=test_src,
                timeout_s=timeout_s,
                created_by=_CREATED_BY,
            )
            return TaskRealization(status="green", module_src=module_src, test_src=test_src)
        except StageExecutionError as exc:
            last_error = failure_text(ctx, exc)
            last_error_ref = getattr(getattr(exc, "persisted_ref", None), "id", None)
            fb = ctx.artifact_store.put_text(
                kind="task_build_feedback",
                text=(
                    f"## task {task.id} attempt {attempt}/{ceiling} failed\n\n{last_error}\n\n"
                    "Rewrite the guilty artifact (task module or its pytest). molmcp covers "
                    "domain APIs only; prefer live source signatures over inventing APIs."
                ),
                created_by=_CREATED_BY,
                parent_ids=[],
            )
            feedback_ids = [fb.id]

    blocked = BlockedTask(
        task_id=task.id,
        ir_task_id=task.ir_task_id,
        slug=slug,
        attempts=ceiling,
        error=last_error,
        error_artifact_id=last_error_ref,
        question=(
            f"task {task.ir_task_id} failed after {ceiling} attempts; how should it be written?"
        ),
    )
    return TaskRealization(status="blocked", blocked=blocked)
