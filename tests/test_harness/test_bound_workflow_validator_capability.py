"""Capability-aware (Phase 4) checks of ``BoundWorkflowValidator``.

These fire only when a ``CapabilityRegistry`` is supplied. Codes:
``unknown_capability`` / ``capability_call_invalid`` / ``backend_not_supported``
/ ``undeclared_side_effect``. Also locks the regression that ``registry=None``
stays byte-identical to Phase 3, and the skip semantics for an unknown cap.
"""

from __future__ import annotations

from pathlib import Path


def _baseline_with_registry():
    """Build a matched ir/bw pair AND a registry that satisfies it."""
    from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry
    from molexp.harness.schemas.bound_workflow import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )
    from molexp.harness.schemas.capability import ToolCapability
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import (
        ExpectedOutput,
        PlanTaskIR,
        PlanWorkflowIR,
    )

    t1 = PlanTaskIR(
        id="t1",
        name="Build",
        purpose="x",
        task_type="builder",
        inputs={"n_chains": ParameterValue(value=100, source="user_provided")},
        outputs={"structure": "structure.pdb"},
    )
    ir = PlanWorkflowIR(
        id="wf-001",
        name="wf",
        objective="x",
        inputs={"n_chains": ParameterValue(value=100, source="user_provided")},
        tasks=[t1],
        edges=[],
        expected_outputs=[
            ExpectedOutput(name="structure", kind="dataset", description="x"),
        ],
    )

    b1 = BoundTask(
        id="b1",
        ir_task_id="t1",
        capability_id="molpy.builder.X",
        package="molpy",
        callable="molpy.builder.X.run",
        parameters={"n_chains": ParameterValue(value=100, source="user_provided")},
        inputs={"n_chains": "wf:n_chains"},
        outputs={"structure": "structure.pdb"},
        side_effects=["fs_write"],
    )
    bw = BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[b1],
        edges=[],
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend="local",
            max_runtime_s=3600,
            denied_paths=["/", "~/.ssh"],
        ),
    )

    cap = ToolCapability(
        id="molpy.builder.X",
        package="molpy",
        name="X",
        description="builder",
        input_schema={
            "type": "object",
            "properties": {"n_chains": {"type": "integer"}},
            "required": ["n_chains"],
        },
        output_schema={"type": "object", "properties": {"structure": {"type": "string"}}},
        supported_backends=["local"],
        side_effects=["fs_write", "network"],
    )
    registry = InMemoryCapabilityRegistry(capabilities=[cap])
    return ir, bw, registry, t1


def _codes(report) -> list[str]:
    return [v.code for v in report.violations]


class TestBoundWorkflowValidatorCapability:
    def test_baseline_with_registry_passes_clean(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw, registry, _ = _baseline_with_registry()
        report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path, registry=registry)
        assert report.passed is True
        assert report.violations == []

    def test_registry_none_yields_phase3_behavior(self, tmp_path: Path) -> None:
        """With registry=None no capability-aware code fires, even for a bogus id."""
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw, _, _ = _baseline_with_registry()
        nonsense = bw.tasks[0].model_copy(update={"capability_id": "ghost"})
        bw_bad = bw.model_copy(update={"tasks": [nonsense]})
        report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path)
        new_codes = {
            "unknown_capability",
            "capability_call_invalid",
            "backend_not_supported",
            "undeclared_side_effect",
        }
        assert not (set(_codes(report)) & new_codes)

    def test_unknown_capability_suppresses_other_checks_for_that_task(
        self,
        tmp_path: Path,
    ) -> None:
        """An unknown capability_id yields ``unknown_capability`` and skips the
        other three checks for that task (we can't reason about a ghost)."""
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw, registry, _ = _baseline_with_registry()
        nonsense = bw.tasks[0].model_copy(update={"capability_id": "ghost.capability"})
        bw_bad = bw.model_copy(update={"tasks": [nonsense]})
        report = BoundWorkflowValidator.validate(
            bw_bad, ir, workspace_root=tmp_path, registry=registry
        )
        codes = _codes(report)
        assert "unknown_capability" in codes
        assert "capability_call_invalid" not in codes
        assert "backend_not_supported" not in codes
        assert "undeclared_side_effect" not in codes

    def test_capability_call_invalid(self, tmp_path: Path) -> None:
        """Capability requires n_chains; dropping it from BoundTask.parameters
        makes ``registry.validate_call`` reject → ``capability_call_invalid``."""
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw, registry, _ = _baseline_with_registry()
        bad_task = bw.tasks[0].model_copy(update={"parameters": {}})
        bw_bad = bw.model_copy(update={"tasks": [bad_task]})
        report = BoundWorkflowValidator.validate(
            bw_bad, ir, workspace_root=tmp_path, registry=registry
        )
        assert "capability_call_invalid" in _codes(report)

    def test_backend_not_supported(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw, registry, _ = _baseline_with_registry()
        bw_bad = bw.model_copy(update={"execution_backend": "slurm"})  # cap only supports "local"
        report = BoundWorkflowValidator.validate(
            bw_bad, ir, workspace_root=tmp_path, registry=registry
        )
        assert "backend_not_supported" in _codes(report)

    def test_undeclared_side_effect(self, tmp_path: Path) -> None:
        from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

        ir, bw, registry, _ = _baseline_with_registry()
        # cap.side_effects = ["fs_write", "network"]; task claims "gpu" which is undeclared.
        bad_task = bw.tasks[0].model_copy(update={"side_effects": ["fs_write", "gpu"]})
        bw_bad = bw.model_copy(update={"tasks": [bad_task]})
        report = BoundWorkflowValidator.validate(
            bw_bad, ir, workspace_root=tmp_path, registry=registry
        )
        assert "undeclared_side_effect" in _codes(report)
