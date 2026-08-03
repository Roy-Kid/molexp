"""Deterministic ``TaskBoard`` → ``BoundWorkflow`` + ``experiment_spec`` materialization.

Phase-2 realization expects a ``bound_workflow`` and ``experiment_spec``
artifact. The planning loop produces an :class:`ExperimentPlan` (opaque
spec + board); this module is the single conversion seam so realization
never re-implements board→bound mapping.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from molexp.harness.plan.experiment_plan import ExperimentPlan
from molexp.harness.schemas.bound_workflow import (
    BoundTask,
    BoundWorkflow,
    ExecutionEnvironment,
    ResourcePolicy,
)
from molexp.harness.schemas.workflow_ir import DependencyEdge
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.harness.schemas import PlanArtifactRef
    from molexp.harness.store.file_artifact_store import FileArtifactStore

__all__ = ["board_plan_to_bound_workflow", "materialize_plan_for_realization"]


def board_plan_to_bound_workflow(
    plan: ExperimentPlan,
    *,
    workflow_ir_id: str | None = None,
    bound_id: str | None = None,
) -> BoundWorkflow:
    """Project a frozen experiment plan into a minimal :class:`BoundWorkflow`.

    Each board task becomes a :class:`BoundTask`. Capability identity prefers
    the first feasibility ``probed_refs`` entry when present; otherwise a
    stable placeholder ``board.<task_id>`` is used so realization can still
    attempt codegen (and block with an intervention if it cannot green).

    Edges are sequential in board order (t0 → t1 → …) when there are 2+
    tasks — a conservative default until the planner records explicit deps.
    """
    tasks: list[BoundTask] = []
    for task in plan.board.tasks:
        refs = ()
        if task.feasibility is not None:
            refs = task.feasibility.probed_refs
        cap = refs[0] if refs else f"board.{task.id}"
        package, _, callable_name = cap.partition(":")
        if not callable_name:
            package, callable_name = "molexp", cap.replace(".", "_")
        tasks.append(
            BoundTask(
                id=task.id,
                ir_task_id=task.id,
                capability_id=cap,
                package=package or "molexp",
                callable=callable_name or task.id,
                parameters={},
                inputs={},
                outputs={"result": f"{task.id}.result"},
                side_effects=[],
                tests=list(task.acceptance),
                provenance={
                    "source": "plan_board",
                    "task_name": task.name,
                },
            )
        )

    edges: list[DependencyEdge] = [
        DependencyEdge(source_task_id=left.id, target_task_id=right.id)
        for left, right in pairwise(tasks)
    ]

    return BoundWorkflow(
        id=bound_id or f"bw-{generate_id()}",
        workflow_ir_id=workflow_ir_id or f"wir-{generate_id()}",
        tasks=tasks,
        edges=edges,
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend="local",
            max_runtime_s=3600,
            denied_paths=["/", "~/.ssh"],
        ),
        review_flags=[],
    )


def materialize_plan_for_realization(
    plan: ExperimentPlan,
    store: FileArtifactStore,
    *,
    created_by: str,
    parent_ids: tuple[str, ...] = (),
) -> tuple[PlanArtifactRef, PlanArtifactRef]:
    """Persist ``experiment_spec`` + ``bound_workflow`` for :class:`RealizeBoard`.

    Returns:
        ``(experiment_spec_ref, bound_workflow_ref)``.
    """
    parents = list(parent_ids)
    spec_obj = dict(plan.spec)
    if "id" not in spec_obj:
        spec_obj["id"] = str(spec_obj.get("title") or "experiment")
    spec_ref = store.put_json(
        kind="experiment_spec",
        obj=spec_obj,
        created_by=created_by,
        parent_ids=parents,
    )
    bound = board_plan_to_bound_workflow(plan)
    bound_ref = store.put_json(
        kind="bound_workflow",
        obj=bound.model_dump(mode="json"),
        created_by=created_by,
        parent_ids=[spec_ref.id, *parents],
    )
    return spec_ref, bound_ref
