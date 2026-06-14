"""Tests for :func:`molexp.workflow.contract.validate_workflow_contract`.

Each :class:`ValidationCheckId` member gets one positive (passes) and
one negative (fails with the expected ``check_id``) test case. The
aggregation rule ``report.ok = not any(severity == 'error')`` is also
exercised directly with mixed warning / error fixtures.
"""

from __future__ import annotations

from molexp.workflow.contract import (
    ArtifactDecl,
    TaskInputSpec,
    TaskIO,
    TaskOutputSpec,
    ValidationCheck,
    ValidationCheckId,
    WorkflowContract,
    validate_workflow_contract,
)


def _emitted_check_ids(report) -> set[ValidationCheckId]:  # type: ignore[no-untyped-def]
    return {issue.check_id for issue in report.issues}


# ── unique_artifact_paths ──────────────────────────────────────────────────


def test_unique_artifact_paths_passes_when_paths_distinct() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                artifacts=(ArtifactDecl(path="a/x.json", produced_by="A"),),
            ),
            TaskIO(
                task_id="B",
                artifacts=(ArtifactDecl(path="b/y.json", produced_by="B"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.unique_artifact_paths not in _emitted_check_ids(rep)


def test_unique_artifact_paths_fails_on_duplicate_path() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                artifacts=(ArtifactDecl(path="dup.json", produced_by="A"),),
            ),
            TaskIO(
                task_id="B",
                artifacts=(ArtifactDecl(path="dup.json", produced_by="B"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.unique_artifact_paths in _emitted_check_ids(rep)
    issue = next(i for i in rep.issues if i.check_id is ValidationCheckId.unique_artifact_paths)
    assert issue.target == "dup.json"
    assert issue.severity == "error"
    assert rep.ok is False


# ── acyclic_data_edges ─────────────────────────────────────────────────────


def test_acyclic_data_edges_passes_on_dag() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                outputs=(TaskOutputSpec(name="x", type="int"),),
            ),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="x", type="int", source="A"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.acyclic_data_edges not in _emitted_check_ids(rep)


def test_acyclic_data_edges_fails_on_cycle() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                inputs=(TaskInputSpec(name="x", type="int", source="B"),),
            ),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="y", type="int", source="A"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.acyclic_data_edges in _emitted_check_ids(rep)
    assert rep.ok is False


# ── every_input_has_source ─────────────────────────────────────────────────


def test_every_input_has_source_passes_when_all_have_source() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                outputs=(TaskOutputSpec(name="x", type="int"),),
            ),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="x", type="int", source="A"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.every_input_has_source not in _emitted_check_ids(rep)


def test_every_input_has_source_fails_without_spec_when_source_none() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                inputs=(TaskInputSpec(name="x", type="int"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.every_input_has_source in _emitted_check_ids(rep)
    issue = next(i for i in rep.issues if i.check_id is ValidationCheckId.every_input_has_source)
    assert issue.target == "A"
    assert rep.ok is False


# ── produced_by_resolves ───────────────────────────────────────────────────


def test_produced_by_resolves_passes_when_known_task() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                artifacts=(ArtifactDecl(path="a.json", produced_by="A"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.produced_by_resolves not in _emitted_check_ids(rep)


def test_produced_by_resolves_fails_on_unknown_task() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                artifacts=(ArtifactDecl(path="a.json", produced_by="ghost"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.produced_by_resolves in _emitted_check_ids(rep)
    issue = next(i for i in rep.issues if i.check_id is ValidationCheckId.produced_by_resolves)
    assert issue.target == "a.json"
    assert rep.ok is False


# ── outputs_match_downstream_inputs ────────────────────────────────────────


def test_outputs_match_downstream_inputs_passes_when_names_align() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                outputs=(TaskOutputSpec(name="x", type="int"),),
            ),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="x", type="int", source="A"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.outputs_match_downstream_inputs not in _emitted_check_ids(rep)


def test_outputs_match_downstream_inputs_warns_on_name_mismatch() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                outputs=(TaskOutputSpec(name="x", type="int"),),
            ),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="y", type="int", source="A"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert ValidationCheckId.outputs_match_downstream_inputs in _emitted_check_ids(rep)
    issue = next(
        i for i in rep.issues if i.check_id is ValidationCheckId.outputs_match_downstream_inputs
    )
    assert issue.severity == "warning"
    # Warning-only ⇒ report still ok.
    assert rep.ok is True


# ── no_orphan_tasks (spec-aware) ───────────────────────────────────────────


def test_no_orphan_tasks_no_op_without_spec() -> None:
    """Without a spec the check has nothing to cross-reference."""
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(TaskIO(task_id="phantom"),),
    )
    rep = validate_workflow_contract(contract, spec=None)
    assert ValidationCheckId.no_orphan_tasks not in _emitted_check_ids(rep)


def test_no_orphan_tasks_passes_when_spec_set_matches_contract() -> None:
    from molexp.workflow.compiler import WorkflowCompiler
    from molexp.workflow.task import Task

    class Inert(Task):
        async def execute(self, ctx):  # type: ignore[no-untyped-def, override]
            return None

    spec = (
        WorkflowCompiler(name="wf")
        .add(Inert(), name="A")
        .add(Inert(), name="B", depends_on=["A"])
        .compile()
    )
    contract = WorkflowContract(
        workflow_id=spec.workflow_id,
        task_io=(
            TaskIO(task_id="A", outputs=(TaskOutputSpec(name="x", type="int"),)),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="x", type="int", source="A"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract, spec=spec)
    assert ValidationCheckId.no_orphan_tasks not in _emitted_check_ids(rep)


def test_no_orphan_tasks_fails_when_spec_has_extra_task() -> None:
    from molexp.workflow.compiler import WorkflowCompiler
    from molexp.workflow.task import Task

    class Inert(Task):
        async def execute(self, ctx):  # type: ignore[no-untyped-def, override]
            return None

    spec = (
        WorkflowCompiler(name="wf")
        .add(Inert(), name="A")
        .add(Inert(), name="B", depends_on=["A"])
        .compile()
    )
    # Contract is missing TaskIO for "B".
    contract = WorkflowContract(
        workflow_id=spec.workflow_id,
        task_io=(TaskIO(task_id="A"),),
    )
    rep = validate_workflow_contract(contract, spec=spec)
    assert ValidationCheckId.no_orphan_tasks in _emitted_check_ids(rep)
    bad = [i for i in rep.issues if i.check_id is ValidationCheckId.no_orphan_tasks]
    assert any(i.target == "B" for i in bad)
    assert rep.ok is False


# ── ValidationReport.ok aggregation ────────────────────────────────────────


def test_report_ok_true_when_only_warning_issues_present() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                outputs=(TaskOutputSpec(name="x", type="int"),),
            ),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="y", type="int", source="A"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    # The mismatch is warning-level only → no error issues → ok stays True.
    assert all(i.severity == "warning" for i in rep.issues)
    assert rep.ok is True


def test_report_ok_false_when_any_error_issue_present() -> None:
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                artifacts=(ArtifactDecl(path="dup", produced_by="A"),),
            ),
            TaskIO(
                task_id="B",
                artifacts=(ArtifactDecl(path="dup", produced_by="B"),),
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert any(i.severity == "error" for i in rep.issues)
    assert rep.ok is False


def test_severity_override_promotes_warning_to_error() -> None:
    """A consumer who wants strict ``outputs_match_downstream_inputs``
    can promote it to error-severity; ``ok`` flips accordingly."""
    contract = WorkflowContract(
        workflow_id="workflow_00000000",
        task_io=(
            TaskIO(
                task_id="A",
                outputs=(TaskOutputSpec(name="x", type="int"),),
            ),
            TaskIO(
                task_id="B",
                inputs=(TaskInputSpec(name="y", type="int", source="A"),),
            ),
        ),
        validation_checks=(
            ValidationCheck(
                id=ValidationCheckId.outputs_match_downstream_inputs,
                severity="error",
            ),
        ),
    )
    rep = validate_workflow_contract(contract)
    assert rep.ok is False
