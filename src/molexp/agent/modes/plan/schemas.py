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
- :class:`ApprovalDecision` — kept for sub-spec 06's ``HumanReview``
  node.
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

from molexp.workflow import WorkflowContract

__all__ = [
    "ApprovalDecision",
    "DigestResult",
    "IngestReportResult",
    "PlanBrief",
    "PlanBriefResult",
    "ReportDigest",
    "SkeletonResult",
    "TaskIRBrief",
    "TaskIRResult",
    "WorkflowContract",
    "WorkflowIRResult",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")


# ── Approval (kept for sub-spec 06) ────────────────────────────────────────


class ApprovalDecision(BaseModel):
    """Human-review verdict.

    Used by sub-spec 06's ``HumanReview`` node; kept here so
    intermediate sub-specs that don't yet wire human review have a
    stable place to import the type from.
    """

    model_config = _FROZEN

    approved: bool
    reason: str = ""


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
