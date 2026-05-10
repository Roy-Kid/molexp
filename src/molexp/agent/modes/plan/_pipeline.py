"""Builder for the materialize-to-workspace PlanMode pipeline.

Thirteen nodes::

    IngestReport → DraftReportDigest → DraftImplementationPlan
        → CompileWorkflowIR → CompileTaskIR
        → DraftCapabilityNeeds → DiscoverCapabilities          [Phase 4]
        → GenerateWorkflowSkeleton
        → GenerateTaskTests / GenerateTaskImplementations  (parallel)
        → ValidateWorkspace → HumanReview → FinalHandoffCheck

The pipeline is a pure data-edge DAG — no control edges, no
``wf.loop`` primitive. ``GenerateWorkflowSkeleton`` (and the two
parallel codegen siblings) reads three upstreams (its two original
inputs plus the new ``DiscoverCapabilities`` output), so its
``ctx.inputs`` is a ``dict[str, *Result]`` keyed by upstream node
name; the rest take a single bare upstream value.
"""

from __future__ import annotations

from mollog import get_logger

from molexp.agent.modes.plan.tasks import (
    CompileTaskIR,
    CompileWorkflowIR,
    DraftImplementationPlan,
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
from molexp.workflow import Workflow, WorkflowBuilder

__all__ = [
    "PLAN_WORKFLOW",
    "build_plan_workflow",
]


_LOG = get_logger(__name__)


def build_plan_workflow() -> Workflow:
    """Assemble the 13-node materialize-to-workspace pipeline.

    Pipeline shape::

        IngestReport → DraftReportDigest → DraftImplementationPlan
            → CompileWorkflowIR → CompileTaskIR
            → DraftCapabilityNeeds → DiscoverCapabilities
            → GenerateWorkflowSkeleton
            → GenerateTaskTests / GenerateTaskImplementations
            → ValidateWorkspace → HumanReview → FinalHandoffCheck

    Step names are the Task class ``__name__`` so
    :class:`~molexp.agent.modes.plan.policy.PlanModelPolicy.tier_for`
    finds them by their canonical id without any per-pipeline mapping
    table. ``GenerateTaskTests`` and ``GenerateTaskImplementations``
    fan out from ``GenerateWorkflowSkeleton`` (data-graph siblings);
    ``ValidateWorkspace`` joins them and ``HumanReview`` is the
    review node, then ``FinalHandoffCheck`` verifies the RunMode
    entrypoint before the workflow terminates.

    The capability-discovery pair (``DraftCapabilityNeeds`` →
    ``DiscoverCapabilities``) sits between ``CompileTaskIR`` and the
    codegen fan-out so every codegen node receives the
    :class:`~molexp.agent.modes.plan.capability.CapabilityEvidenceBatch`
    needed for the AST evidence gate (Phase 5).
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
        next_="DraftCapabilityNeeds",
    )
    builder.add(
        DraftCapabilityNeeds(),
        name="DraftCapabilityNeeds",
        depends_on=["DraftImplementationPlan", "CompileWorkflowIR", "CompileTaskIR"],
        next_="DiscoverCapabilities",
    )
    builder.add(
        DiscoverCapabilities(),
        name="DiscoverCapabilities",
        depends_on=["DraftCapabilityNeeds"],
        next_="GenerateWorkflowSkeleton",
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
        next_="FinalHandoffCheck",
    )
    builder.add(
        FinalHandoffCheck(),
        name="FinalHandoffCheck",
        depends_on=["HumanReview"],
    )
    spec = builder.build()
    _LOG.debug(f"[plan-pipeline] built workflow_id={spec.workflow_id} tasks={len(spec._tasks)}")
    return spec


PLAN_WORKFLOW: Workflow = build_plan_workflow()
"""Module-level frozen :class:`Workflow` for the v1 PlanMode pipeline."""
