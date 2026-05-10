"""Structured payload types for the rewritten PlanMode pipeline.

Replaces the old in-memory PlanSpec / ApprovedPlan family with the
**materialize-to-workspace** pipeline's contracts:

- :class:`ReportDigest` / :class:`PlanBrief` — natural-language
  digests of the user-supplied report and the proposed implementation
  plan; rendered to disk and consumed by downstream nodes.
- :class:`TaskIRBrief` — per-task IR companion to
  :class:`molexp.workflow.WorkflowContract.TaskIO` carrying the
  natural-language responsibility / success-criteria fields a code
  generator needs.
- :class:`ApprovalDecision` — :class:`HumanReview` node's verdict;
  carries the structured repair targets consumed by the
  ``planmode-review-repair-loop`` driver.
- :class:`RepairIterationRecord` — per-iteration audit record stored
  on the manifest's ``repair_history`` tuple.
- ``*Result`` types (one per workflow node) — frozen, *path-bearing*
  return values. Downstream nodes consume :class:`Path` references,
  never embedded blobs, so the workspace is the single source of
  truth for materialized content.

:class:`molexp.workflow.WorkflowContract` is re-exported here so
PlanMode tasks can compose the typed-IR contract without importing
from the workflow module directly in every file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes.plan.capability import (
    CapabilityEvidence,
    CapabilityEvidenceBatch,
    CapabilityNeed,
    CapabilityNeedReport,
    MissingCapability,
)
from molexp.agent.modes.plan.workspace_layout import (
    CheckResult,
    RepairIterationRecord,
)
from molexp.workflow import WorkflowContract

__all__ = [
    "ApprovalDecision",
    "CapabilityEvidence",
    "CapabilityEvidenceBatch",
    "CapabilityNeed",
    "CapabilityNeedReport",
    "DigestResult",
    "HandoffResult",
    "IngestReportResult",
    "MissingCapability",
    "PlanBrief",
    "PlanBriefResult",
    "PlanReviewView",
    "RepairIterationRecord",
    "ReportDigest",
    "SkeletonResult",
    "TaskIRBrief",
    "TaskIRResult",
    "TaskImplementationModule",
    "TaskImplementationsResult",
    "TaskTestModule",
    "TaskTestsResult",
    "ValidationResult",
    "WorkflowContract",
    "WorkflowIRResult",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")


# ── Approval ───────────────────────────────────────────────────────────────


class ApprovalDecision(BaseModel):
    """Human-review verdict consumed by the ``HumanReview`` node and the
    review→repair loop driver.

    The first three fields encode the basic approve / reject signal (kept
    backward-compatible with serialized payloads predating the repair-loop
    extension). The four ``target_*`` / ``cascade_downstream`` / ``feedback``
    fields encode a *structured rejection*: which plan-level nodes to
    re-run, which experiment-task ids to regenerate, whether downstream
    nodes should cascade, and an optional natural-language hint for the
    next LLM round.

    Attributes:
        approved: ``True`` to accept the materialized plan, ``False`` to
            request a repair iteration.
        reason: Free-form short justification (one sentence).
        override_validation: When ``True``, accepts a plan whose
            :class:`ValidationResult.passed` was ``False``; the manifest
            is then marked ``approved_with_override``.
        target_node_ids: Plan-level node names (e.g. ``"DraftImplementationPlan"``,
            ``"CompileWorkflowIR"``) that the repair loop should re-execute.
            Empty tuple when only ``target_task_ids`` rework is requested
            or when ``approved`` is ``True``.
        target_task_ids: Experiment-task ids (e.g. ``"prepare"``) whose
            generated test / implementation modules need a fresh LLM pass
            via ``GenerateTaskTests`` / ``GenerateTaskImplementations``.
        cascade_downstream: When ``True``, every downstream node of the
            ``target_node_ids`` selection is added to the partial-rerun
            subgraph (``Workflow.subgraph(..., include_downstream=True)``).
        feedback: Optional natural-language note persisted to
            ``repairs/iter-<n>/decision.yaml`` and threaded into the next
            iteration's LLM prompt under ``repair_feedback``.
    """

    model_config = _FROZEN

    approved: bool
    reason: str = ""
    override_validation: bool = False
    target_node_ids: tuple[str, ...] = ()
    target_task_ids: tuple[str, ...] = ()
    cascade_downstream: bool = False
    feedback: str = ""


# ``RepairIterationRecord`` lives in ``workspace_layout.py`` to break the
# circular import chain (workspace_layout exports CheckResult here, and
# PlanManifest needs to embed RepairIterationRecord). It is re-exported
# from this module so downstream callers can import it from the same
# place as :class:`ApprovalDecision`.


# ── Natural-language digests ───────────────────────────────────────────────


class ReportDigest(BaseModel):
    """Structured summary of a user-supplied experimental report.

    Rendered to ``report/digest.md`` by ``DraftReportDigest``; the
    structured form is preserved alongside the markdown so downstream
    nodes can read field values without re-parsing prose.

    Attributes:
        summary: One-paragraph plain-language overview.
        experimental_goal: The single principal objective.
        scientific_assumptions: Tuple of explicit assumptions / priors.
        systems_and_variables: Tuple of subjects-of-study (e.g. one
            chemical system + observable variables per entry).
        expected_outputs: Tuple of natural-language descriptions of the
            outputs the experiment is supposed to produce.
        missing_information: Tuple of gaps the report did not specify;
            these surface as warnings to a human reviewer.
    """

    model_config = _FROZEN

    summary: str
    experimental_goal: str
    scientific_assumptions: tuple[str, ...] = ()
    systems_and_variables: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()


class PlanBrief(BaseModel):
    """Natural-language implementation plan for the experiment.

    Bridges the gap between the (terse) :class:`ReportDigest` and the
    (machine-readable) :class:`WorkflowContract`. The workflow IR
    compiler reads this to ground its own structured output.

    Attributes:
        overview: One-paragraph plain-language plan.
        chosen_method: Concrete method name + brief rationale.
        stages: Ordered list of stages (each a short noun phrase).
        rationale: Why this decomposition; surfaced to human review.
    """

    model_config = _FROZEN

    overview: str
    chosen_method: str
    stages: tuple[str, ...] = ()
    rationale: str = ""


class TaskIRBrief(BaseModel):
    """Per-task natural-language brief paired with the typed IR.

    Sub-spec 06's code-generation passes consume both the structured
    :class:`molexp.workflow.TaskIO` (typed inputs / outputs / artifacts)
    AND this brief (responsibility / success criteria / failure
    conditions / minimal test expectations).

    Attributes:
        task_id: Matches the task id in the workflow contract.
        responsibility: One-sentence statement of what the task does.
        success_criteria: Tuple of natural-language post-conditions.
        failure_conditions: Tuple of natural-language failure modes.
        test_expectations: Tuple of test-shaped sentences (sub-spec 06
            translates these into ``pytest.skip`` markers when the
            implementation is a stub).
        is_stub: True if the implementation may legitimately be
            ``raise NotImplementedError`` (v1 of sub-spec 06 emits
            stubs for tasks the LLM cannot write end-to-end).
    """

    model_config = _FROZEN

    task_id: str
    responsibility: str
    success_criteria: tuple[str, ...] = ()
    failure_conditions: tuple[str, ...] = ()
    test_expectations: tuple[str, ...] = ()
    is_stub: bool = False


# ── Per-node Result types ──────────────────────────────────────────────────


class IngestReportResult(BaseModel):
    """``IngestReport`` output — original report path + content hash."""

    model_config = _FROZEN

    report_path: Path
    report_hash: str


class DigestResult(BaseModel):
    """``DraftReportDigest`` output — digest path + structured payload."""

    model_config = _FROZEN

    digest_path: Path
    digest: ReportDigest


class PlanBriefResult(BaseModel):
    """``DraftImplementationPlan`` output — plan path + structured brief."""

    model_config = _FROZEN

    plan_path: Path
    plan_brief: PlanBrief


class WorkflowIRResult(BaseModel):
    """``CompileWorkflowIR`` output — YAML path + parsed contract."""

    model_config = _FROZEN

    workflow_yaml_path: Path
    contract: WorkflowContract


class TaskIRResult(BaseModel):
    """``CompileTaskIR`` output — per-task YAML paths + parsed briefs."""

    model_config = _FROZEN

    task_ir_paths: tuple[Path, ...]
    briefs: tuple[TaskIRBrief, ...]


class SkeletonResult(BaseModel):
    """``GenerateWorkflowSkeleton`` output — generated package paths.

    The skeleton is validated via :func:`compile` (syntax-only — no
    import, no execution); a failure raises
    :class:`~molexp.agent.modes.plan.errors.SkeletonCompileError`.
    """

    model_config = _FROZEN

    workflow_py_path: Path
    package_path: Path


# ── Sub-spec 06: codegen / validation / handoff schemas ───────────────────


class TaskTestModule(BaseModel):
    """Generated pytest module source for one task.

    The provider returns one of these per task; the generated text
    becomes ``tests/test_<task_id>.py``.

    Attributes:
        task_id: Matches the contract entry the test exercises.
        source: Full pytest module source (header + imports + tests).
        imports: Optional descriptive list of imports the source uses;
            informational, not consumed by the writer.
        fixtures: Optional descriptive list of pytest fixtures the
            source defines.
        evidence_refs: Tuple of ``api_ref`` values (``f"{module}.{symbol}"``)
            the LLM declares it used. Phase 5 codegen nodes assert
            this set equals the source's module-level
            ``__capability_evidence__: tuple[str, ...] = (...)`` literal
            and forwards it to
            :func:`~molexp.agent.modes.plan.capability.validate_codegen_evidence`
            for the AST diff. Defaults to ``()`` so older callers /
            ``discovery_skipped=True`` paths stay valid.
    """

    model_config = _FROZEN

    task_id: str
    source: str
    imports: tuple[str, ...] = ()
    fixtures: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class TaskTestsResult(BaseModel):
    """``GenerateTaskTests`` output — paths to every generated test file."""

    model_config = _FROZEN

    test_paths: tuple[Path, ...]


class TaskImplementationModule(BaseModel):
    """Generated module source for one task's runnable implementation.

    Attributes:
        task_id: Matches the contract entry this module implements.
        source: Full module source (imports + class definition +
            ``async def execute(ctx)``).
        is_stub: When True, the source is a placeholder that raises
            :class:`NotImplementedError`. The validator's pytest
            invocation respects this flag via ``pytest.skip("stub")``
            so v1 stubs do not fail CI.
        evidence_refs: Tuple of ``api_ref`` values the LLM declares it
            used; Phase 5 codegen asserts this matches the source's
            ``__capability_evidence__`` literal. ``is_stub=True`` is
            permitted to ship an empty tuple (no Molcrafts API calls
            in stub bodies).
    """

    model_config = _FROZEN

    task_id: str
    source: str
    is_stub: bool = False
    evidence_refs: tuple[str, ...] = ()


class TaskImplementationsResult(BaseModel):
    """``GenerateTaskImplementations`` output — paths to every impl file."""

    model_config = _FROZEN

    impl_paths: tuple[Path, ...]


class ValidationResult(BaseModel):
    """``ValidateWorkspace`` output — report path + pass/fail summary."""

    model_config = _FROZEN

    report_path: Path
    passed: bool
    summary: str
    checks: tuple[CheckResult, ...] = ()
    status: str = ""


class PlanReviewView(BaseModel):
    """Snapshot of materialized PlanMode artifacts surfaced to a human reviewer.

    Composed by ``HumanReviewTask`` from the per-node ``*Result``
    objects, plus the materialized workspace path. The view is the
    minimal payload an interactive reviewer / UI needs to decide
    approve / reject; it does not embed the full markdown / IR text
    (those live on disk and are linked by path).

    Attributes:
        plan_id: Identifier for the plan being reviewed.
        experiment_workspace_path: Root directory of the materialized workspace.
        digest: Structured report digest.
        plan_brief: Natural-language implementation plan.
        contract: Typed workflow contract.
        validation_passed: Echo of :class:`ValidationResult.passed`.
        validation_summary: Echo of :class:`ValidationResult.summary`.
        previous_validation_failures: Identifiers of validation checks that
            failed in the prior iteration (empty on the very first round).
            Surfaced so a human reviewer can see what triggered the most
            recent rejection.
        repair_iteration: Zero on the first round; increments by one each
            time the review→repair loop re-enters with a rejection.
    """

    model_config = _FROZEN

    plan_id: str
    experiment_workspace_path: Path
    digest: ReportDigest
    plan_brief: PlanBrief
    contract: WorkflowContract
    validation_passed: bool
    validation_summary: str
    previous_validation_failures: tuple[str, ...] = ()
    repair_iteration: int = 0


class HandoffResult(BaseModel):
    """``HumanReview`` output — wraps the :class:`PlanRunHandoff`."""

    model_config = _FROZEN

    handoff: PlanRunHandoff
    decision: ApprovalDecision
    validation_passed: bool
    ready_for_run: bool = False
    status: str = "ready_for_review"
    report_path: Path | None = None
    validation_checks: tuple[CheckResult, ...] = ()


# Avoid circular import at type-check time; the runtime resolution
# happens via :meth:`HandoffResult.model_rebuild` once both classes
# are defined (see ``__init__.py``).
from molexp.agent.modes.plan.handoff import PlanRunHandoff  # noqa: E402

HandoffResult.model_rebuild()
