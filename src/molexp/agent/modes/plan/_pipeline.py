"""PlanMode pipeline: 5 stages × small subgraphs, inside a wf.loop.

Five cognitive stages, each driving its own small sub-workflow:

* :data:`INTAKE_SUBGRAPH` — IngestReport → DraftReportDigest → ClarifyMissingInformation.
* :data:`DISCOVERY_SUBGRAPH` — DraftCapabilityNeeds → DiscoverCapabilities,
  fed by a seeded :class:`_BoundaryStub` for the upstream plan brief.
* :data:`MATERIALIZE_SUBGRAPH` — CompileWorkflowIR → CompileTaskIR →
  GenerateWorkflowSkeleton → (GenerateTaskTests ∥ GenerateTaskImplementations),
  fed by seeded boundaries for plan brief + evidence.
* :data:`VERIFY_SUBGRAPH` — ValidateWorkspace → HumanReview →
  FinalHandoffCheck, fed by seeded boundaries for the upstream artefacts.

Strategize is a single leaf (``DraftImplementationPlan``); its stage
task invokes the leaf directly without wrapping it in a 1-node sub-workflow.

The outer workflow (:data:`PLAN_WORKFLOW`) is an 8-task chain wrapped
in :meth:`WorkflowBuilder.loop`::

    PrepareIteration ──▶ UnderstandStage ──▶ StrategizeStage
                                                  │
                              ┌───────────────────┘
                              ▼
                         BindStage ──▶ MaterializeStage ──▶ VerifyStage
                                                                 │
                                                                 ▼
                                                           RepairDecide
                                                                 │
                                              Next("continue") ──┤
                                              ▼                  │
                                       (back to PrepareIteration)
                                                                 │
                                                            Next("exit") ──▶ End

Every stage is a single-input single-output Task, so the outer body is
a strict chain — ``wf.loop``'s data-dep ready check is safe even
across iterations. Internal parallel forks live inside sub-workflows,
which run fresh each iteration and never carry state between rounds.
"""

from __future__ import annotations

from typing import Any

from mollog import get_logger

from molexp.agent.modes.plan._bookends import (
    BindStage,
    MaterializeStage,
    PrepareIteration,
    RepairDecide,
    StrategizeStage,
    UnderstandStage,
    VerifyStage,
)
from molexp.agent.modes.plan.tasks import (
    ClarifyMissingInformation,
    CompileTaskIR,
    CompileWorkflowIR,
    DraftReportDigest,
    FinalHandoffCheck,
    GenerateTaskImplementations,
    GenerateTaskTests,
    GenerateWorkflowSkeleton,
    HumanReview,
    IngestReport,
    ValidateWorkspace,
)
from molexp.agent.modes.plan.tasks_capability import (
    DiscoverCapabilities,
    DraftCapabilityNeeds,
)
from molexp.workflow import Task, TaskContext, Workflow, WorkflowBuilder

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DISCOVERY_SUBGRAPH",
    "INTAKE_SUBGRAPH",
    "MATERIALIZE_SUBGRAPH",
    "PLAN_WORKFLOW",
    "VERIFY_SUBGRAPH",
    "build_discovery_subgraph",
    "build_intake_subgraph",
    "build_materialize_subgraph",
    "build_plan_workflow",
    "build_verify_subgraph",
]


_LOG = get_logger(__name__)


DEFAULT_MAX_ITERATIONS = 8
"""Default repair-loop budget; matches :attr:`PlanModeConfig.max_iterations`."""


# ── Boundary stub ──────────────────────────────────────────────────────────


class _BoundaryStub(Task):
    """No-op placeholder task that gets seeded via ``seed_outputs=``.

    A sub-workflow that needs cross-stage data from an upstream stage
    registers a :class:`_BoundaryStub` under the upstream task's name.
    The stage task then passes the actual value via ``seed_outputs``;
    the workflow runtime treats the stub as already-completed and
    threads the seeded value into downstream tasks' ``ctx.inputs``
    without ever invoking the stub's body.
    """

    async def execute(self, ctx: TaskContext[Any, Any, None]) -> None:
        del ctx
        raise RuntimeError(
            "_BoundaryStub.execute called — the stub must be seeded via "
            "Workflow.execute(seed_outputs={<name>: value}). A missing seed "
            "is a wiring bug in the enclosing stage task."
        )


# ── Sub-workflow builders ──────────────────────────────────────────────────


def build_intake_subgraph() -> Workflow:
    """Stage 1/5 — receive + summarize + clarify the user's request.

    Three leaves, no boundaries: ``user_input`` flows in via
    ``ctx.config``, terminal output is :class:`ClarificationResult`.
    """
    builder = WorkflowBuilder(name="plan_intake", entry="IngestReport")
    builder.add(IngestReport(), name="IngestReport")
    builder.add(DraftReportDigest(), name="DraftReportDigest", depends_on=["IngestReport"])
    builder.add(
        ClarifyMissingInformation(),
        name="ClarifyMissingInformation",
        depends_on=["DraftReportDigest"],
    )
    return builder.build()


def build_discovery_subgraph() -> Workflow:
    """Stage 3/5 — draft capability needs and resolve them to evidence.

    Two leaves + one boundary: ``ClarifyMissingInformation`` seeded by
    UnderstandStage's output. Terminal: :class:`CapabilityEvidenceBatch`.
    """
    builder = WorkflowBuilder(name="plan_discovery", entry=["ClarifyMissingInformation"])
    builder.add(_BoundaryStub(), name="ClarifyMissingInformation")
    builder.add(
        DraftCapabilityNeeds(),
        name="DraftCapabilityNeeds",
        depends_on=["ClarifyMissingInformation"],
    )
    builder.add(
        DiscoverCapabilities(),
        name="DiscoverCapabilities",
        depends_on=["DraftCapabilityNeeds"],
    )
    return builder.build()


def build_materialize_subgraph() -> Workflow:
    """Stage 4/5 — IR compilation + per-task code generation.

    Five leaves + two boundaries (``DraftImplementationPlan``,
    ``DiscoverCapabilities``). Tests + Impls fan out from
    GenerateWorkflowSkeleton; the sub-workflow runs fresh each outer
    iteration so the parallel branches are safe.
    """
    builder = WorkflowBuilder(
        name="plan_materialize",
        entry=["DraftImplementationPlan", "DiscoverCapabilities"],
    )
    builder.add(_BoundaryStub(), name="DraftImplementationPlan")
    builder.add(_BoundaryStub(), name="DiscoverCapabilities")
    builder.add(
        CompileWorkflowIR(),
        name="CompileWorkflowIR",
        depends_on=["DraftImplementationPlan", "DiscoverCapabilities"],
    )
    builder.add(
        CompileTaskIR(),
        name="CompileTaskIR",
        depends_on=["CompileWorkflowIR", "DiscoverCapabilities"],
    )
    builder.add(
        GenerateWorkflowSkeleton(),
        name="GenerateWorkflowSkeleton",
        depends_on=["CompileWorkflowIR", "CompileTaskIR", "DiscoverCapabilities"],
    )
    builder.add(
        GenerateTaskTests(),
        name="GenerateTaskTests",
        depends_on=["CompileTaskIR", "GenerateWorkflowSkeleton", "DiscoverCapabilities"],
    )
    builder.add(
        GenerateTaskImplementations(),
        name="GenerateTaskImplementations",
        depends_on=["CompileTaskIR", "GenerateWorkflowSkeleton", "DiscoverCapabilities"],
    )
    return builder.build()


def build_verify_subgraph() -> Workflow:
    """Stage 5/5 — validate, human review, final handoff.

    Three leaves + five boundaries. The boundaries capture every
    cross-stage dependency the verify leaves still consume by name.
    Terminal: :class:`HandoffResult` from FinalHandoffCheck.
    """
    builder = WorkflowBuilder(
        name="plan_verify",
        entry=[
            "DraftReportDigest",
            "DraftImplementationPlan",
            "CompileTaskIR",
            "GenerateTaskTests",
            "GenerateTaskImplementations",
        ],
    )
    # Boundaries seeded by VerifyStage from runtime.last_inner_outputs.
    builder.add(_BoundaryStub(), name="DraftReportDigest")
    builder.add(_BoundaryStub(), name="DraftImplementationPlan")
    builder.add(_BoundaryStub(), name="CompileTaskIR")
    builder.add(_BoundaryStub(), name="GenerateTaskTests")
    builder.add(_BoundaryStub(), name="GenerateTaskImplementations")
    builder.add(
        ValidateWorkspace(),
        name="ValidateWorkspace",
        depends_on=["CompileTaskIR", "GenerateTaskTests", "GenerateTaskImplementations"],
    )
    builder.add(
        HumanReview(),
        name="HumanReview",
        depends_on=["DraftReportDigest", "DraftImplementationPlan", "ValidateWorkspace"],
    )
    builder.add(
        FinalHandoffCheck(),
        name="FinalHandoffCheck",
        depends_on=["HumanReview"],
    )
    return builder.build()


INTAKE_SUBGRAPH: Workflow = build_intake_subgraph()
"""Frozen subgraph for stage 1 (Understand). 3 leaves."""

DISCOVERY_SUBGRAPH: Workflow = build_discovery_subgraph()
"""Frozen subgraph for stage 3 (Bind). 2 leaves + 1 boundary."""

MATERIALIZE_SUBGRAPH: Workflow = build_materialize_subgraph()
"""Frozen subgraph for stage 4 (Materialize). 5 leaves + 2 boundaries."""

VERIFY_SUBGRAPH: Workflow = build_verify_subgraph()
"""Frozen subgraph for stage 5 (Verify). 3 leaves + 5 boundaries."""


# ── Outer workflow ─────────────────────────────────────────────────────────


def build_plan_workflow(*, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> Workflow:
    """Assemble the outer review→repair workflow (8 tasks).

    Shape::

        PrepareIteration ──control──▶ UnderstandStage ──▶ StrategizeStage
            ──▶ BindStage ──▶ MaterializeStage ──▶ VerifyStage ──▶ RepairDecide

        wf.loop(body=[PrepareIteration, …5 stages], until=RepairDecide)

    Stages are wired with ``depends_on`` so each stage receives the
    previous stage's typed output as ``ctx.inputs``. PrepareIteration
    is wired via ``wf.control`` (no payload) so UnderstandStage starts
    with ``ctx.inputs=None``.
    """
    builder = WorkflowBuilder(name="plan_mode", entry="PrepareIteration")
    builder.add(PrepareIteration(), name="PrepareIteration")
    builder.control("PrepareIteration", "UnderstandStage")
    builder.add(UnderstandStage(), name="UnderstandStage")
    builder.add(StrategizeStage(), name="StrategizeStage", depends_on=["UnderstandStage"])
    builder.add(BindStage(), name="BindStage", depends_on=["StrategizeStage"])
    builder.add(MaterializeStage(), name="MaterializeStage", depends_on=["BindStage"])
    builder.add(VerifyStage(), name="VerifyStage", depends_on=["MaterializeStage"])
    builder.add(RepairDecide(), name="RepairDecide", depends_on=["VerifyStage"])

    builder.loop(
        body=[
            "PrepareIteration",
            "UnderstandStage",
            "StrategizeStage",
            "BindStage",
            "MaterializeStage",
            "VerifyStage",
        ],
        until="RepairDecide",
        max_iters=max_iterations,
        on_exit="_end",
    )

    spec = builder.build()
    _LOG.debug(
        f"[plan-pipeline] built outer workflow_id={spec.workflow_id} "
        f"max_iters={max_iterations}"
    )
    return spec


PLAN_WORKFLOW: Workflow = build_plan_workflow()
"""Outer plan-mode workflow with the default repair budget.
Use :func:`build_plan_workflow` for a custom ``max_iterations``."""
