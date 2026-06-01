"""Typed sidecar contract layer for the workflow IR.

The contract is a declarative companion to a Workflow — it does not change
runtime semantics. Validation runs through :func:`validate_workflow_contract`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class ValidationCheckId(StrEnum):
    """Known validation check identifiers."""

    no_orphan_tasks = "no_orphan_tasks"
    unique_artifact_paths = "unique_artifact_paths"
    acyclic_data_edges = "acyclic_data_edges"
    every_input_has_source = "every_input_has_source"
    produced_by_resolves = "produced_by_resolves"
    outputs_match_downstream_inputs = "outputs_match_downstream_inputs"


_FROZEN = ConfigDict(frozen=True, extra="forbid")


# ── Per-task IO declaration ──────────────────────────────────────────────


class TaskInputSpec(BaseModel):
    """One declared input for a task in the workflow contract."""

    model_config = _FROZEN

    name: str
    type: str
    required: bool = True
    source: str | None = Field(
        default=None,
        description=(
            "The bare task_id of the upstream task that produces this "
            "input (e.g. 'parse_monomer'), exactly as that task's "
            "'task_id' field reads. Never 'task_id.output_name'. "
            "None for a workflow-level input no upstream task produces."
        ),
    )
    description: str = ""


class TaskOutputSpec(BaseModel):
    """One declared output for a task in the workflow contract."""

    model_config = _FROZEN

    name: str
    type: str
    description: str = ""


class ArtifactDecl(BaseModel):
    """One declared on-disk artifact a task produces."""

    model_config = _FROZEN

    path: str
    mime: str = ""
    description: str = ""
    produced_by: str


class TaskIO(BaseModel):
    """One task's declared inputs / outputs / artifacts."""

    model_config = _FROZEN

    task_id: str
    inputs: tuple[TaskInputSpec, ...] = ()
    outputs: tuple[TaskOutputSpec, ...] = ()
    artifacts: tuple[ArtifactDecl, ...] = ()


# ── Validation types ─────────────────────────────────────────────────────


class ValidationCheck(BaseModel):
    """A check selection — id + severity."""

    model_config = _FROZEN

    id: ValidationCheckId
    severity: Severity = "error"


class ValidationIssue(BaseModel):
    """One finding produced by a check runner."""

    model_config = _FROZEN

    check_id: ValidationCheckId
    severity: Severity
    target: str
    message: str
    hint: str = ""


class ValidationReport(BaseModel):
    """Aggregate of all issues a validation pass produced."""

    model_config = _FROZEN

    ok: bool
    issues: tuple[ValidationIssue, ...] = ()


class WorkflowContract(BaseModel):
    """Sidecar contract wrapping a Workflow's IR."""

    model_config = _FROZEN

    workflow_id: str
    task_io: tuple[TaskIO, ...] = ()
    validation_checks: tuple[ValidationCheck, ...] = ()


# ── Default checks ───────────────────────────────────────────────────────


_C = ValidationCheckId
_DEFAULT_CHECKS: tuple[ValidationCheck, ...] = (
    ValidationCheck(id=_C.no_orphan_tasks, severity="error"),
    ValidationCheck(id=_C.unique_artifact_paths, severity="error"),
    ValidationCheck(id=_C.acyclic_data_edges, severity="error"),
    ValidationCheck(id=_C.every_input_has_source, severity="error"),
    ValidationCheck(id=_C.produced_by_resolves, severity="error"),
    ValidationCheck(id=_C.outputs_match_downstream_inputs, severity="warning"),
)


def default_validation_checks() -> tuple[ValidationCheck, ...]:
    """Return the baseline tuple of checks (all error except outputs_match_downstream_inputs)."""
    return _DEFAULT_CHECKS


# ── Check runners ────────────────────────────────────────────────────────


def _check_no_orphan_tasks(
    contract: WorkflowContract, spec: Workflow | None
) -> list[ValidationIssue]:
    """Cross-check spec task set against contract.task_io."""
    if spec is None:
        return []
    spec_names: set[str] = {t.name for t in spec._tasks}
    contract_ids: set[str] = {tio.task_id for tio in contract.task_io}
    issues: list[ValidationIssue] = []
    for tid in spec_names - contract_ids:
        issues.append(
            ValidationIssue(
                check_id=_C.no_orphan_tasks,
                severity="error",
                target=tid,
                message=f"task {tid!r} declared in spec has no TaskIO entry in the contract",
                hint="Add a TaskIO entry to contract.task_io for this task.",
            )
        )
    for tid in contract_ids - spec_names:
        issues.append(
            ValidationIssue(
                check_id=_C.no_orphan_tasks,
                severity="error",
                target=tid,
                message=f"contract.task_io references {tid!r}, which is not a task in the spec",
                hint="Remove the stray TaskIO entry, or add the task to the spec.",
            )
        )
    return issues


def _check_unique_artifact_paths(
    contract: WorkflowContract, _spec: object | None
) -> list[ValidationIssue]:
    """No two artifacts may declare the same path."""
    seen: dict[str, str] = {}
    issues: list[ValidationIssue] = []
    for tio in contract.task_io:
        for art in tio.artifacts:
            prev = seen.get(art.path)
            if prev is None:
                seen[art.path] = tio.task_id
                continue
            issues.append(
                ValidationIssue(
                    check_id=_C.unique_artifact_paths,
                    severity="error",
                    target=art.path,
                    message=f"artifact path {art.path!r} declared by both {prev!r} and {tio.task_id!r}",
                    hint="Pick distinct paths or merge the declarations.",
                )
            )
    return issues


def _check_acyclic_data_edges(
    contract: WorkflowContract, _spec: object | None
) -> list[ValidationIssue]:
    """The data dep graph induced by inputs[].source must be acyclic."""
    deps: dict[str, set[str]] = {}
    for tio in contract.task_io:
        deps[tio.task_id] = {inp.source for inp in tio.inputs if inp.source is not None}

    visiting: set[str] = set()
    visited: set[str] = set()
    issues: list[ValidationIssue] = []

    def dfs(node: str, path: list[str]) -> None:
        if node in visiting:
            cycle = " → ".join([*path[path.index(node) :], node])
            issues.append(
                ValidationIssue(
                    check_id=_C.acyclic_data_edges,
                    severity="error",
                    target=node,
                    message=f"cycle detected in data edges: {cycle}",
                    hint="Break the cycle or model the back-edge as a control edge.",
                )
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
) -> list[ValidationIssue]:
    """Every input declares a non-None source (entry tasks exempt when spec provided)."""
    entry_ids: set[str] = set()
    if spec is not None:
        entry_ids = {t.name for t in spec._tasks if not t.depends_on}
    issues: list[ValidationIssue] = []
    for tio in contract.task_io:
        if tio.task_id in entry_ids:
            continue
        for inp in tio.inputs:
            if inp.source is None:
                issues.append(
                    ValidationIssue(
                        check_id=_C.every_input_has_source,
                        severity="error",
                        target=tio.task_id,
                        message=f"input {inp.name!r} on task {tio.task_id!r} has no source",
                        hint="Set source=<upstream_task_id> or move the task to the workflow entry.",
                    )
                )
    return issues


def _check_produced_by_resolves(
    contract: WorkflowContract, _spec: object | None
) -> list[ValidationIssue]:
    """Every artifact's produced_by must reference a contract task_id."""
    known: set[str] = {tio.task_id for tio in contract.task_io}
    issues: list[ValidationIssue] = []
    for tio in contract.task_io:
        for art in tio.artifacts:
            if art.produced_by not in known:
                issues.append(
                    ValidationIssue(
                        check_id=_C.produced_by_resolves,
                        severity="error",
                        target=art.path,
                        message=f"artifact {art.path!r} declares produced_by={art.produced_by!r}, "
                        f"which is not a known task",
                        hint="Set produced_by to a task_id that has a TaskIO entry in this contract.",
                    )
                )
    return issues


def _check_outputs_match_downstream_inputs(
    contract: WorkflowContract, _spec: object | None
) -> list[ValidationIssue]:
    """Each input's (source, name) must appear in the source task's outputs."""
    outputs_by_task: dict[str, set[str]] = {
        tio.task_id: {out.name for out in tio.outputs} for tio in contract.task_io
    }
    issues: list[ValidationIssue] = []
    for tio in contract.task_io:
        for inp in tio.inputs:
            if inp.source is None:
                continue
            upstream_outputs = outputs_by_task.get(inp.source)
            if upstream_outputs is None or inp.name not in upstream_outputs:
                if upstream_outputs is None:
                    continue  # owned by produced_by_resolves / no_orphan_tasks
                issues.append(
                    ValidationIssue(
                        check_id=_C.outputs_match_downstream_inputs,
                        severity="warning",
                        target=tio.task_id,
                        message=f"input {inp.name!r} on task {tio.task_id!r} sources from "
                        f"{inp.source!r}, which does not declare an output named {inp.name!r}",
                        hint="Add a matching TaskOutputSpec on the upstream task.",
                    )
                )
    return issues


# ── Runner dispatch ──────────────────────────────────────────────────────

_RUNNERS: dict[ValidationCheckId, object] = {
    _C.no_orphan_tasks: _check_no_orphan_tasks,
    _C.unique_artifact_paths: _check_unique_artifact_paths,
    _C.acyclic_data_edges: _check_acyclic_data_edges,
    _C.every_input_has_source: _check_every_input_has_source,
    _C.produced_by_resolves: _check_produced_by_resolves,
    _C.outputs_match_downstream_inputs: _check_outputs_match_downstream_inputs,
}


# ── Validation entry point ───────────────────────────────────────────────


def validate_workflow_contract(
    contract: WorkflowContract,
    *,
    spec: Workflow | None = None,
) -> ValidationReport:
    """Run every selected check against the contract.

    When spec is supplied, spec-aware checks cross-check against the spec's
    task set; when omitted those checks fall back to contract-internal logic.
    """
    checks = contract.validation_checks if contract.validation_checks else _DEFAULT_CHECKS
    issues: list[ValidationIssue] = []
    check_severity: dict[ValidationCheckId, Severity] = {c.id: c.severity for c in checks}
    for check in checks:
        runner = _RUNNERS.get(check.id)
        if runner is None:
            continue
        issues.extend(runner(contract, spec))
    # Apply per-check severity override (runner defaults may differ from config)
    for i, issue in enumerate(issues):
        configured = check_severity.get(issue.check_id)
        if configured is not None and configured != issue.severity:
            issues[i] = issue.model_copy(update={"severity": configured})
    ok = not any(i.severity == "error" for i in issues)
    return ValidationReport(ok=ok, issues=tuple(issues))
