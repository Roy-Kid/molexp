"""Builder for the materialize-to-workspace PlanMode pipeline.

Six nodes — ``IngestReport → DraftReportDigest → DraftImplementationPlan
→ CompileWorkflowIR → CompileTaskIR → GenerateWorkflowSkeleton``.
Sub-spec 06 extends this pipeline with the remaining four nodes
(``GenerateTaskTests``, ``GenerateTaskImplementations``,
``ValidateWorkspace``, ``HumanReview``); ``build_plan_workflow`` is
factored so 06 can either rebuild from primitives or call
:func:`extend_plan_pipeline` (added in that sub-spec) on the
:data:`PLAN_WORKFLOW` constant.

The pipeline is a pure data-edge DAG — no control edges, no
``wf.loop`` primitive. ``GenerateWorkflowSkeleton`` reads two
upstreams (the workflow contract + the per-task IR set) so its
``ctx.inputs`` is a ``dict[str, *Result]`` keyed by upstream node name.
"""

from __future__ import annotations

from molexp.agent.modes.plan.tasks import (
    CompileTaskIR,
    CompileWorkflowIR,
    DraftImplementationPlan,
    DraftReportDigest,
    GenerateTaskImplementations,
    GenerateTaskTests,
    GenerateWorkflowSkeleton,
    HumanReview,
    IngestReport,
    ValidateWorkspace,
)
from molexp.workflow import Workflow, WorkflowBuilder

__all__ = [
    "PLAN_WORKFLOW",
    "build_plan_workflow",
]


def build_plan_workflow() -> Workflow:
    """Assemble the 10-node materialize-to-workspace pipeline.

    Pipeline shape::

        IngestReport → DraftReportDigest → DraftImplementationPlan
            → CompileWorkflowIR → CompileTaskIR → GenerateWorkflowSkeleton
            → GenerateTaskTests / GenerateTaskImplementations
            → ValidateWorkspace → HumanReview

    Step names are the Task class ``__name__`` so
    :class:`~molexp.agent.modes.plan.policy.PlanModelPolicy.tier_for`
    finds them by their canonical id without any per-pipeline mapping
    table. ``GenerateTaskTests`` and ``GenerateTaskImplementations``
    fan out from ``GenerateWorkflowSkeleton`` (data-graph siblings);
    ``ValidateWorkspace`` joins them and ``HumanReview`` is the
    terminal node.
    """
    builder = WorkflowBuilder(name="plan_mode", entry="IngestReport")
    builder.add(IngestReport(), name="IngestReport", next_="DraftReportDigest")
    builder.add(
        DraftReportDigest(),
        name="DraftReportDigest",
        depends_on=["IngestReport"],
        next_="DraftImplementationPlan",
    )
    builder.add(
        DraftImplementationPlan(),
        name="DraftImplementationPlan",
        depends_on=["DraftReportDigest"],
        next_="CompileWorkflowIR",
    )
    builder.add(
        CompileWorkflowIR(),
        name="CompileWorkflowIR",
        depends_on=["DraftImplementationPlan"],
        next_="CompileTaskIR",
    )
    builder.add(
        CompileTaskIR(),
        name="CompileTaskIR",
        depends_on=["CompileWorkflowIR"],
        next_="GenerateWorkflowSkeleton",
    )
    builder.add(
        GenerateWorkflowSkeleton(),
        name="GenerateWorkflowSkeleton",
        depends_on=["CompileWorkflowIR", "CompileTaskIR"],
    )
    builder.add(
        GenerateTaskTests(),
        name="GenerateTaskTests",
        depends_on=["CompileTaskIR", "GenerateWorkflowSkeleton"],
    )
    builder.add(
        GenerateTaskImplementations(),
        name="GenerateTaskImplementations",
        depends_on=["CompileTaskIR", "GenerateWorkflowSkeleton"],
    )
    builder.add(
        ValidateWorkspace(),
        name="ValidateWorkspace",
        depends_on=[
            "CompileTaskIR",
            "GenerateTaskTests",
            "GenerateTaskImplementations",
        ],
        next_="HumanReview",
    )
    builder.add(
        HumanReview(),
        name="HumanReview",
        depends_on=["ValidateWorkspace"],
    )
    return builder.build()


PLAN_WORKFLOW: Workflow = build_plan_workflow()
"""Module-level frozen :class:`Workflow` for the v1 PlanMode pipeline."""
