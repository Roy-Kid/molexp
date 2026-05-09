"""Typed sidecar contract layer for the workflow IR.

The contract is a *declarative* companion to a :class:`Workflow` /
:class:`WorkflowSpec` — it does not change runtime semantics.
Generated code (e.g. PlanMode-emitted task modules) compiles
against the contract; the runtime keeps using ``TaskContext.inputs``
/ outputs as today.

The shapes here are pure data: every type is a frozen
:class:`pydantic.BaseModel` with ``extra="forbid"`` so an unknown
field at the boundary surfaces a :class:`pydantic.ValidationError`
rather than silently sliding through. Collection fields default to
``()`` so absent sections in IR JSON parse to empty tuples
(back-compat).

Validation runs through :func:`validate_workflow_contract`. Each
member of :class:`ValidationCheckId` is paired with a runner; adding
a new check is one enum entry plus one runner function. Some checks
are intrinsic to the contract (no spec needed); the spec-aware ones
take an optional ``spec`` argument so the same entry point covers
both phases.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .spec import Workflow

__all__ = [
    "ArtifactDecl",
    "Severity",
    "TaskIO",
    "TaskInputSpec",
    "TaskOutputSpec",
    "ValidationCheck",
    "ValidationCheckId",
    "ValidationIssue",
    "ValidationReport",
    "WorkflowContract",
    "default_validation_checks",
    "validate_workflow_contract",
]

Severity = Literal["error", "warning"]
"""Two-level issue severity. ``"error"`` flips ``ValidationReport.ok`` to
``False``; ``"warning"`` is informational and does not fail the
report."""

_FROZEN = ConfigDict(frozen=True, extra="forbid")


# ── Per-task IO declaration ────────────────────────────────────────────────


class TaskInputSpec(BaseModel):
    """One declared input for a task in the workflow contract.

    Attributes:
        name: Identifier of this input within the task body.
        type: JSON-Schema-style type string. Free-form
            (``"string"`` / ``"array<float>"`` /
            ``"object[FetchResult]"``) — the contract layer does not
            interpret it; downstream code generators do.
        required: Whether the task body must receive this input. False
            inputs may be omitted by upstream callers.
        source: Upstream task name producing the value, or ``None`` for
            entry-task inputs (caller-supplied).
        description: Human-readable hint; consumed by docs / planners.
    """

    model_config = _FROZEN

    name: str
    type: str
    required: bool = True
    source: str | None = None
    description: str = ""


class TaskOutputSpec(BaseModel):
    """One declared output for a task in the workflow contract."""

    model_config = _FROZEN

    name: str
    type: str
    description: str = ""


class ArtifactDecl(BaseModel):
    """One declared on-disk artifact a task produces.

    Attributes:
        path: POSIX-style path relative to the run dir (e.g.
            ``"artifacts/foo.json"``). Forward-slash separator only.
        mime: MIME type or extension hint (e.g. ``"application/json"``,
            ``"text/csv"``). Free-form.
        description: Human-readable description.
        produced_by: ``task_id`` of the task that writes this artifact.
            Must reference a ``TaskIO`` entry in the same contract for
            the ``produced_by_resolves`` validation check to pass.
    """

    model_config = _FROZEN

    path: str
    mime: str = ""
    description: str = ""
    produced_by: str


class TaskIO(BaseModel):
    """One task's declared inputs / outputs / artifacts.

    All three collections default to empty tuples so a contract whose
    tasks only declare some axes is well-formed.
    """

    model_config = _FROZEN

    task_id: str
    inputs: tuple[TaskInputSpec, ...] = ()
    outputs: tuple[TaskOutputSpec, ...] = ()
    artifacts: tuple[ArtifactDecl, ...] = ()


# ── Validation checks ──────────────────────────────────────────────────────


class ValidationCheckId(StrEnum):
    """Identifiers for the static checks implemented in this module.

    Adding a new check is one enum entry plus one runner function in
    the ``_CHECK_RUNNERS`` table at the bottom of this module — by
    design, so the catalog is grep-discoverable.
    """

    no_orphan_tasks = "no_orphan_tasks"
    """Every task in the spec has a matching ``TaskIO`` entry, and
    every ``TaskIO.task_id`` references a task in the spec.
    Spec-dependent — emits no issues when ``spec`` is omitted."""

    unique_artifact_paths = "unique_artifact_paths"
    """No two artifacts across the whole contract declare the same
    ``path``."""

    acyclic_data_edges = "acyclic_data_edges"
    """The data-dependency graph induced by ``inputs[].source`` is
    acyclic."""

    every_input_has_source = "every_input_has_source"
    """Every input declares a non-``None`` ``source``. Spec-aware
    variant: when ``spec`` is provided, entry tasks (no incoming data
    edge) are exempt; without ``spec`` every ``source=None`` is
    flagged regardless of position."""

    produced_by_resolves = "produced_by_resolves"
    """Every artifact's ``produced_by`` references a ``task_id``
    declared in the contract's ``task_io`` list."""

    outputs_match_downstream_inputs = "outputs_match_downstream_inputs"
    """Every input's ``(source, name)`` matches some output declared
    by the named upstream task. Warning-level by default — runtime
    inputs may legitimately come from configuration, not from
    upstream output declarations."""


class ValidationCheck(BaseModel):
    """A check selection — the id of the runner plus its severity.

    The contract may override the default severity of a check (e.g.
    promote ``outputs_match_downstream_inputs`` from warning to error
    in a strict consumer).
    """

    model_config = _FROZEN

    id: ValidationCheckId
    severity: Severity = "error"


class ValidationIssue(BaseModel):
    """One finding produced by a check runner."""

    model_config = _FROZEN

    check_id: ValidationCheckId
    severity: Severity
    target: str
    """Either an offending ``task_id`` or an ``artifact_path`` —
    whichever the runner identifies as the locus of the failure."""
    message: str
    hint: str = ""


class ValidationReport(BaseModel):
    """Aggregate of all issues a validation pass produced.

    ``ok`` is computed at construction from ``issues``: True when no
    issue carries ``severity == "error"``, regardless of whether
    warning-level issues exist.
    """

    model_config = _FROZEN

    ok: bool
    issues: tuple[ValidationIssue, ...] = ()


# Default validation suite — applied when the contract omits its own
# ``validation_checks`` tuple.
def default_validation_checks() -> tuple[ValidationCheck, ...]:
    """Return the baseline tuple of checks the contract layer applies.

    All checks default to ``severity="error"`` except
    ``outputs_match_downstream_inputs``, which is warning-level so the
    common case of "input came from config, not upstream" does not
    trip the report.
    """
    return (
        ValidationCheck(id=ValidationCheckId.no_orphan_tasks, severity="error"),
        ValidationCheck(id=ValidationCheckId.unique_artifact_paths, severity="error"),
        ValidationCheck(id=ValidationCheckId.acyclic_data_edges, severity="error"),
        ValidationCheck(id=ValidationCheckId.every_input_has_source, severity="error"),
        ValidationCheck(id=ValidationCheckId.produced_by_resolves, severity="error"),
        ValidationCheck(
            id=ValidationCheckId.outputs_match_downstream_inputs,
            severity="warning",
        ),
    )


# ── Workflow-level contract ────────────────────────────────────────────────


class WorkflowContract(BaseModel):
    """Sidecar contract wrapping a :class:`Workflow`'s IR.

    Attributes:
        workflow_id: Must match the wrapped spec's ``workflow_id`` so a
            contract can never drift onto a different spec. Format
            mirrors the spec IR's ``workflow_id`` (``workflow_<8 hex>``
            in canonical form, but the contract itself does not enforce
            the regex — that's the spec layer's job).
        task_io: One :class:`TaskIO` per task, in any order. Empty for
            workflows that haven't declared their I/O surface yet.
        validation_checks: Selected check + severity tuple. When empty,
            :func:`validate_workflow_contract` substitutes
            :func:`default_validation_checks`.
    """

    model_config = _FROZEN

    workflow_id: str
    task_io: tuple[TaskIO, ...] = ()
    validation_checks: tuple[ValidationCheck, ...] = ()


# ── Validation entry point ─────────────────────────────────────────────────


def validate_workflow_contract(
    contract: WorkflowContract,
    *,
    spec: Workflow | None = None,
) -> ValidationReport:
    """Run every selected check against the contract.

    When ``spec`` is supplied the spec-aware checks (currently
    ``no_orphan_tasks`` and ``every_input_has_source``) cross-check
    against the spec's task set; when omitted those checks fall back
    to contract-internal logic only.

    Args:
        contract: The contract under test.
        spec: Optional :class:`Workflow` providing the authoritative
            task list for cross-checks.

    Returns:
        A :class:`ValidationReport` whose ``ok`` field is ``True`` iff
        no error-severity issue was emitted.
    """
    checks: Iterable[ValidationCheck] = (
        contract.validation_checks if contract.validation_checks else default_validation_checks()
    )
    issues: list[ValidationIssue] = []
    for check in checks:
        runner = _CHECK_RUNNERS[check.id]
        for issue_kwargs in runner(contract, spec):
            issues.append(
                ValidationIssue(
                    check_id=check.id,
                    severity=check.severity,
                    **issue_kwargs,
                )
            )
    ok = not any(i.severity == "error" for i in issues)
    return ValidationReport(ok=ok, issues=tuple(issues))


# ── Per-check runners ──────────────────────────────────────────────────────
#
# Each runner returns an iterable of partial ``ValidationIssue`` kwarg
# dicts — the dispatcher fills in ``check_id`` and ``severity``. This
# keeps every runner focused on the *finding*, not on the bookkeeping
# of severity selection.


_IssueKwargs = dict[str, str]


def _check_no_orphan_tasks(
    contract: WorkflowContract, spec: Workflow | None
) -> Sequence[_IssueKwargs]:
    """Cross-check spec task set against contract.task_io.

    Spec-dependent — without a spec, this check is a no-op (the
    contract has no notion of "the canonical set of tasks").
    """
    if spec is None:
        return ()
    spec_task_names: set[str] = set(_spec_task_names(spec))
    contract_task_ids: set[str] = {tio.task_id for tio in contract.task_io}
    issues: list[_IssueKwargs] = []
    for tid in spec_task_names - contract_task_ids:
        issues.append(
            {
                "target": tid,
                "message": f"task {tid!r} declared in spec has no TaskIO entry in the contract",
                "hint": "Add a TaskIO entry to contract.task_io for this task.",
            }
        )
    for tid in contract_task_ids - spec_task_names:
        issues.append(
            {
                "target": tid,
                "message": f"contract.task_io references {tid!r}, which is not a task in the spec",
                "hint": "Remove the stray TaskIO entry, or add the task to the spec.",
            }
        )
    return issues


def _check_unique_artifact_paths(
    contract: WorkflowContract,
    spec: Workflow | None,  # noqa: ARG001 — dispatcher-uniform signature; this check ignores spec.
) -> Sequence[_IssueKwargs]:
    """No two artifacts may declare the same path."""
    seen: dict[str, str] = {}  # path → first-declaring task_id
    issues: list[_IssueKwargs] = []
    for tio in contract.task_io:
        for art in tio.artifacts:
            prev = seen.get(art.path)
            if prev is None:
                seen[art.path] = tio.task_id
                continue
            issues.append(
                {
                    "target": art.path,
                    "message": (
                        f"artifact path {art.path!r} declared by both {prev!r} and {tio.task_id!r}"
                    ),
                    "hint": "Pick distinct paths or merge the declarations.",
                }
            )
    return issues


def _check_acyclic_data_edges(
    contract: WorkflowContract,
    spec: Workflow | None,  # noqa: ARG001 — dispatcher-uniform signature; this check ignores spec.
) -> Sequence[_IssueKwargs]:
    """The data dep graph induced by inputs[].source must be acyclic."""
    # Build adjacency: task_id -> {sources}
    deps: dict[str, set[str]] = {}
    for tio in contract.task_io:
        sources: set[str] = set()
        for inp in tio.inputs:
            if inp.source is not None:
                sources.add(inp.source)
        deps[tio.task_id] = sources

    visiting: set[str] = set()
    visited: set[str] = set()
    issues: list[_IssueKwargs] = []

    def dfs(node: str, path: list[str]) -> None:
        if node in visiting:
            cycle = " → ".join([*path[path.index(node) :], node])
            issues.append(
                {
                    "target": node,
                    "message": f"cycle detected in data edges: {cycle}",
                    "hint": "Break the cycle or model the back-edge as a control edge.",
                }
            )
            return
        if node in visited:
            return
        visiting.add(node)
        for src in deps.get(node, ()):
            dfs(src, [*path, node])
        visiting.discard(node)
        visited.add(node)

    for task_id in deps:
        if task_id not in visited:
            dfs(task_id, [])
    return issues


def _check_every_input_has_source(
    contract: WorkflowContract, spec: Workflow | None
) -> Sequence[_IssueKwargs]:
    """Every input declares a non-None source.

    Spec-aware: with ``spec``, tasks the spec marks as entry (no
    incoming data edge) are exempt — entry-task inputs are caller-
    supplied so ``source=None`` is correct there. Without ``spec``,
    every ``source=None`` is flagged.
    """
    entry_ids: set[str] = set()
    if spec is not None:
        entry_ids = set(_spec_entry_task_names(spec))

    issues: list[_IssueKwargs] = []
    for tio in contract.task_io:
        if tio.task_id in entry_ids:
            continue
        for inp in tio.inputs:
            if inp.source is None:
                issues.append(
                    {
                        "target": tio.task_id,
                        "message": (f"input {inp.name!r} on task {tio.task_id!r} has no source"),
                        "hint": (
                            "Set source=<upstream_task_id> or move the task to the workflow entry."
                        ),
                    }
                )
    return issues


def _check_produced_by_resolves(
    contract: WorkflowContract,
    spec: Workflow | None,  # noqa: ARG001 — dispatcher-uniform signature; this check ignores spec.
) -> Sequence[_IssueKwargs]:
    """Every artifact's produced_by must reference a contract task_id."""
    known: set[str] = {tio.task_id for tio in contract.task_io}
    issues: list[_IssueKwargs] = []
    for tio in contract.task_io:
        for art in tio.artifacts:
            if art.produced_by not in known:
                issues.append(
                    {
                        "target": art.path,
                        "message": (
                            f"artifact {art.path!r} declares produced_by="
                            f"{art.produced_by!r}, which is not a known task"
                        ),
                        "hint": (
                            "Set produced_by to a task_id that has a TaskIO entry in this contract."
                        ),
                    }
                )
    return issues


def _check_outputs_match_downstream_inputs(
    contract: WorkflowContract,
    spec: Workflow | None,  # noqa: ARG001 — dispatcher-uniform signature; this check ignores spec.
) -> Sequence[_IssueKwargs]:
    """Each input's (source, name) must appear in the source task's outputs."""
    outputs_by_task: dict[str, set[str]] = {
        tio.task_id: {out.name for out in tio.outputs} for tio in contract.task_io
    }
    issues: list[_IssueKwargs] = []
    for tio in contract.task_io:
        for inp in tio.inputs:
            if inp.source is None:
                continue
            upstream_outputs = outputs_by_task.get(inp.source)
            if upstream_outputs is None:
                # The produced_by_resolves / no_orphan_tasks checks own
                # the "source unknown" case; don't double-report here.
                continue
            if inp.name not in upstream_outputs:
                issues.append(
                    {
                        "target": tio.task_id,
                        "message": (
                            f"input {inp.name!r} on task {tio.task_id!r} sources from "
                            f"{inp.source!r}, which does not declare an output named "
                            f"{inp.name!r}"
                        ),
                        "hint": (
                            "Add a matching TaskOutputSpec on the upstream task, or rename "
                            "this input to match an existing output."
                        ),
                    }
                )
    return issues


_CheckRunner = Callable[[WorkflowContract, "Workflow | None"], Sequence["_IssueKwargs"]]


_CHECK_RUNNERS: Mapping[ValidationCheckId, _CheckRunner] = {
    ValidationCheckId.no_orphan_tasks: _check_no_orphan_tasks,
    ValidationCheckId.unique_artifact_paths: _check_unique_artifact_paths,
    ValidationCheckId.acyclic_data_edges: _check_acyclic_data_edges,
    ValidationCheckId.every_input_has_source: _check_every_input_has_source,
    ValidationCheckId.produced_by_resolves: _check_produced_by_resolves,
    ValidationCheckId.outputs_match_downstream_inputs: _check_outputs_match_downstream_inputs,
}


# ── Spec accessors (private helpers) ───────────────────────────────────────


def _spec_task_names(spec: Workflow) -> Sequence[str]:
    """Return the task ids declared in a Workflow.

    Workflow stores its registered tasks under ``_tasks`` (a list of
    ``TaskRegistration``). This helper isolates that knowledge so the
    rest of ``contract.py`` does not import spec internals directly.
    """
    return tuple(t.name for t in spec._tasks)


def _spec_entry_task_names(spec: Workflow) -> Sequence[str]:
    """Return the task ids that have no incoming data edge.

    These are the workflow's data-graph entry points; their inputs are
    legitimately caller-supplied (``source=None`` is correct).
    """
    return tuple(t.name for t in spec._tasks if not t.depends_on)
