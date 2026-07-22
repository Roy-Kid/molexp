"""Tests for :func:`molexp.workflow.contract.validate_workflow_contract`.

Each :class:`ValidationCheckId` member owns a positive (passes) and a negative
(fails with the expected ``check_id`` / target / severity) case — one class per
check. ``ValidationReport.ok = not any(severity == 'error')`` is exercised via
the per-check severity override.
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


class TestUniqueArtifactPaths:
    def test_passes_when_paths_distinct(self) -> None:
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

    def test_fails_on_duplicate_path(self) -> None:
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


class TestAcyclicDataEdges:
    def test_passes_on_dag(self) -> None:
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

    def test_fails_on_cycle(self) -> None:
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


class TestEveryInputHasSource:
    def test_passes_when_all_inputs_have_source(self) -> None:
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

    def test_fails_without_spec_when_source_none(self) -> None:
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
        issue = next(
            i for i in rep.issues if i.check_id is ValidationCheckId.every_input_has_source
        )
        assert issue.target == "A"
        assert rep.ok is False


class TestProducedByResolves:
    def test_passes_when_produced_by_known_task(self) -> None:
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

    def test_fails_on_unknown_task(self) -> None:
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


class TestOutputsMatchDownstreamInputs:
    def test_passes_when_names_align(self) -> None:
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

    def test_warns_on_name_mismatch_but_report_stays_ok(self) -> None:
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


class TestNoOrphanTasks:
    """Spec-aware check: inert without a spec, cross-checks the spec task set
    against ``contract.task_io`` when a spec is supplied."""

    def test_no_op_without_spec(self) -> None:
        contract = WorkflowContract(
            workflow_id="workflow_00000000",
            task_io=(TaskIO(task_id="phantom"),),
        )
        rep = validate_workflow_contract(contract, spec=None)
        assert ValidationCheckId.no_orphan_tasks not in _emitted_check_ids(rep)

    def test_passes_when_spec_set_matches_contract(self) -> None:
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

    def test_fails_when_spec_has_task_absent_from_contract(self) -> None:
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


class TestValidationReportAggregation:
    def test_severity_override_promotes_warning_to_error(self) -> None:
        """A consumer promoting ``outputs_match_downstream_inputs`` to error
        severity flips ``report.ok`` to False."""
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
