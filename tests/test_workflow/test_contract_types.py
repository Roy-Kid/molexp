"""Tests for the frozen-pydantic types in :mod:`molexp.workflow.contract`.

Coverage focus:

- Empty-tuple defaults on every collection field (so absent IR sections
  parse to empty tuples).
- ``extra="forbid"`` rejection of unknown kwargs.
- Frozen-config rejection of post-construction mutation.
- Severity literal accepts only ``"error"`` / ``"warning"``.
- ``ValidationCheckId`` enum membership matches the documented list.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.workflow.contract import (
    ArtifactDecl,
    TaskInputSpec,
    TaskIO,
    TaskOutputSpec,
    ValidationCheck,
    ValidationCheckId,
    ValidationIssue,
    ValidationReport,
    WorkflowContract,
    default_validation_checks,
)

# ── TaskInputSpec / TaskOutputSpec / ArtifactDecl ──────────────────────────


def test_task_input_spec_minimal_construction() -> None:
    inp = TaskInputSpec(name="x", type="string")
    assert inp.name == "x"
    assert inp.type == "string"
    assert inp.required is True
    assert inp.source is None
    assert inp.description == ""


def test_task_input_spec_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        TaskInputSpec(name="x", type="string", extra_field="oops")  # type: ignore[call-arg]


def test_task_input_spec_is_frozen() -> None:
    inp = TaskInputSpec(name="x", type="string")
    with pytest.raises(ValidationError):
        inp.name = "y"  # type: ignore[misc]


def test_task_output_spec_minimal_construction() -> None:
    out = TaskOutputSpec(name="y", type="array<float>")
    assert out.description == ""


def test_task_output_spec_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        TaskOutputSpec(name="y", type="int", weird=1)  # type: ignore[call-arg]


def test_artifact_decl_requires_path_and_produced_by() -> None:
    art = ArtifactDecl(path="artifacts/foo.json", produced_by="taskA")
    assert art.mime == ""
    assert art.description == ""
    with pytest.raises(ValidationError):
        ArtifactDecl(path="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ArtifactDecl(produced_by="t")  # type: ignore[call-arg]


def test_artifact_decl_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ArtifactDecl(path="x", produced_by="t", unknown="z")  # type: ignore[call-arg]


# ── TaskIO ─────────────────────────────────────────────────────────────────


def test_task_io_defaults_collections_to_empty_tuples() -> None:
    tio = TaskIO(task_id="taskA")
    assert tio.inputs == ()
    assert tio.outputs == ()
    assert tio.artifacts == ()


def test_task_io_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        TaskIO(task_id="taskA", typo=1)  # type: ignore[call-arg]


# ── ValidationCheckId / ValidationCheck ────────────────────────────────────


def test_validation_check_id_members_match_documented_list() -> None:
    expected = {
        "no_orphan_tasks",
        "unique_artifact_paths",
        "acyclic_data_edges",
        "every_input_has_source",
        "produced_by_resolves",
        "outputs_match_downstream_inputs",
    }
    actual = {member.value for member in ValidationCheckId}
    assert actual == expected


def test_validation_check_default_severity_is_error() -> None:
    check = ValidationCheck(id=ValidationCheckId.unique_artifact_paths)
    assert check.severity == "error"


def test_validation_check_accepts_only_known_severity() -> None:
    with pytest.raises(ValidationError):
        ValidationCheck(
            id=ValidationCheckId.unique_artifact_paths,
            severity="critical",  # type: ignore[arg-type]
        )


# ── ValidationIssue / ValidationReport ─────────────────────────────────────


def test_validation_issue_minimal_construction() -> None:
    issue = ValidationIssue(
        check_id=ValidationCheckId.unique_artifact_paths,
        severity="error",
        target="artifacts/dup.json",
        message="dup",
    )
    assert issue.hint == ""


def test_validation_issue_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ValidationIssue(
            check_id=ValidationCheckId.unique_artifact_paths,
            severity="error",
            target="x",
            message="m",
            mystery=1,  # type: ignore[call-arg]
        )


def test_validation_report_defaults_issues_to_empty_tuple() -> None:
    rep = ValidationReport(ok=True)
    assert rep.issues == ()


# ── WorkflowContract ───────────────────────────────────────────────────────


def test_workflow_contract_minimal_construction() -> None:
    c = WorkflowContract(workflow_id="workflow_00000000")
    assert c.task_io == ()
    assert c.validation_checks == ()


def test_workflow_contract_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        WorkflowContract(
            workflow_id="workflow_00000000",
            stray=1,  # type: ignore[call-arg]
        )


def test_workflow_contract_is_frozen() -> None:
    c = WorkflowContract(workflow_id="workflow_00000000")
    with pytest.raises(ValidationError):
        c.workflow_id = "other"  # type: ignore[misc]


# ── default_validation_checks ──────────────────────────────────────────────


def test_default_validation_checks_covers_every_id() -> None:
    defaults = default_validation_checks()
    assert {c.id for c in defaults} == set(ValidationCheckId)


def test_default_validation_checks_warning_for_outputs_match() -> None:
    by_id = {c.id: c for c in default_validation_checks()}
    assert by_id[ValidationCheckId.outputs_match_downstream_inputs].severity == "warning"
    # Every other default check is severity="error".
    for cid, check in by_id.items():
        if cid is ValidationCheckId.outputs_match_downstream_inputs:
            continue
        assert check.severity == "error", cid
