"""Tests for validate_bound_workflow (Phase 3 §11.3, structural-only).

One unit test per failure code. Baseline pair (ir, bw) maps `t1`→`b1` and
`t2`→`b2` with a single edge `b1`→`b2`.

Codes:
- unknown_ir_task (error)
- duplicate_ir_task_binding (error)
- input_key_mismatch (error)
- output_key_mismatch (error)
- allowed_path_outside_workspace (error)
- missing_baseline_deny (error)
- edge_topology_mismatch (error)
"""

from __future__ import annotations

from pathlib import Path


def _baseline():
    from molexp.harness.schemas.bound_workflow import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import (
        DependencyEdge,
        ExpectedOutput,
        TaskIR,
        WorkflowIR,
    )

    t1 = TaskIR(
        id="t1",
        name="Build",
        purpose="x",
        task_type="builder",
        inputs={"n_chains": ParameterValue(value=100, source="user_provided")},
        outputs={"structure": "structure.pdb"},
    )
    t2 = TaskIR(
        id="t2",
        name="Run",
        purpose="x",
        task_type="md_runner",
        inputs={"structure": ParameterValue(value="structure.pdb", source="user_provided")},
        outputs={"trajectory": "traj.dcd"},
    )
    ir = WorkflowIR(
        id="wf-001",
        name="wf",
        objective="x",
        inputs={"n_chains": ParameterValue(value=100, source="user_provided")},
        tasks=[t1, t2],
        edges=[DependencyEdge(source_task_id="t1", target_task_id="t2")],
        expected_outputs=[
            ExpectedOutput(name="trajectory", kind="dataset", description="x"),
        ],
    )

    b1 = BoundTask(
        id="b1",
        ir_task_id="t1",
        capability_id="molpy.builder.X",
        package="molpy",
        callable="molpy.builder.X.run",
        parameters={"n_chains": ParameterValue(value=100, source="user_provided")},
        inputs={"n_chains": "wf-input-ref"},
        outputs={"structure": "structure.pdb"},
    )
    b2 = BoundTask(
        id="b2",
        ir_task_id="t2",
        capability_id="molpy.md.Y",
        package="molpy",
        callable="molpy.md.Y.run",
        parameters={},
        inputs={"structure": "b1-structure-ref"},
        outputs={"trajectory": "traj.dcd"},
    )
    bw = BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[b1, b2],
        edges=[DependencyEdge(source_task_id="b1", target_task_id="b2")],
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend="local",
            max_runtime_s=3600,
            allowed_paths=[],  # populated in tests as needed
            denied_paths=["/", "~/.ssh"],
        ),
    )
    return ir, bw


def _codes(report) -> list[str]:
    return [v.code for v in report.violations]


class TestBoundWorkflowValidator:
    # ---------------------------------------------------------------- baseline

    def test_baseline_pair_is_clean(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert report.passed is True
        assert report.violations == []
        assert report.target_kind == "bound_workflow"
        assert report.target_id == "bw-001"

    def test_validate_bound_workflow_signature_and_import(self, tmp_path: Path) -> None:
        from molexp.harness import BoundWorkflowValidator as top
        from molexp.harness.schemas.validation import ValidationReport
        from molexp.harness.validators import BoundWorkflowValidator as via_pkg
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator as via_mod

        assert top is via_pkg is via_mod

        ir, bw = _baseline()
        report = top.validate(bw, ir, workspace_root=tmp_path)
        assert isinstance(report, ValidationReport)

    def test_phase_4_placeholder_comment_present(self) -> None:
        """ac-006: a Phase-4 placeholder comment block marks where
        capability-aware checks will land."""
        import inspect

        from molexp.harness.validators import bound_workflow as mod

        src = inspect.getsource(mod)
        assert "Phase 4" in src
        assert "CapabilityRegistry" in src

    # --------------------------------------------------------------- violations

    def test_unknown_ir_task(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        bad_task = bw.tasks[0].model_copy(update={"ir_task_id": "ghost"})
        bw = bw.model_copy(update={"tasks": [bad_task, bw.tasks[1]]})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "unknown_ir_task" in _codes(report)
        assert report.passed is False

    def test_duplicate_ir_task_binding(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        dup = bw.tasks[1].model_copy(update={"id": "b3", "ir_task_id": "t1"})
        bw = bw.model_copy(update={"tasks": [*bw.tasks, dup]})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "duplicate_ir_task_binding" in _codes(report)

    def test_input_key_mismatch(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        bad = bw.tasks[0].model_copy(update={"inputs": {"wrong_key": "ref"}})
        bw = bw.model_copy(update={"tasks": [bad, bw.tasks[1]]})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "input_key_mismatch" in _codes(report)

    def test_output_key_mismatch(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        bad = bw.tasks[0].model_copy(update={"outputs": {"wrong_out": "structure.pdb"}})
        bw = bw.model_copy(update={"tasks": [bad, bw.tasks[1]]})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "output_key_mismatch" in _codes(report)

    def test_allowed_path_outside_workspace_absolute(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        bad_policy = bw.resource_policy.model_copy(update={"allowed_paths": ["/etc/passwd"]})
        bw = bw.model_copy(update={"resource_policy": bad_policy})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "allowed_path_outside_workspace" in _codes(report)

    def test_allowed_path_outside_workspace_relative_escapes(self, tmp_path: Path) -> None:
        """A relative path that resolves outside workspace must fail."""
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        bad_policy = bw.resource_policy.model_copy(update={"allowed_paths": ["../escape"]})
        bw = bw.model_copy(update={"resource_policy": bad_policy})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "allowed_path_outside_workspace" in _codes(report)

    def test_allowed_path_inside_workspace_is_clean(self, tmp_path: Path) -> None:
        """A relative path inside workspace must NOT trigger the violation."""
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        ok_policy = bw.resource_policy.model_copy(update={"allowed_paths": ["./inside"]})
        bw = bw.model_copy(update={"resource_policy": ok_policy})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "allowed_path_outside_workspace" not in _codes(report)

    def test_missing_baseline_deny_root(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        bad_policy = bw.resource_policy.model_copy(update={"denied_paths": ["~/.ssh"]})  # missing /
        bw = bw.model_copy(update={"resource_policy": bad_policy})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "missing_baseline_deny" in _codes(report)

    def test_missing_baseline_deny_ssh(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        bad_policy = bw.resource_policy.model_copy(update={"denied_paths": ["/"]})  # missing ~/.ssh
        bw = bw.model_copy(update={"resource_policy": bad_policy})
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "missing_baseline_deny" in _codes(report)

    def test_edge_topology_mismatch(self, tmp_path: Path) -> None:
        """Bound edges, after id-translation back to ir_task_id, must equal ir.edges."""
        from molexp.harness.schemas.workflow_ir import DependencyEdge
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        # Reverse the bound edge so the translated topology no longer matches.
        bw = bw.model_copy(
            update={"edges": [DependencyEdge(source_task_id="b2", target_task_id="b1")]}
        )
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "edge_topology_mismatch" in _codes(report)

    def test_edge_topology_match_with_id_translation(self, tmp_path: Path) -> None:
        """Even when BoundTask.id ≠ TaskIR.id, topology must agree after translation."""
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert "edge_topology_mismatch" not in _codes(report)

    def test_no_violations_when_clean(self, tmp_path: Path) -> None:
        """The full clean baseline must produce ValidationReport.passed=True."""
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw = _baseline()
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path)
        assert report.passed is True
