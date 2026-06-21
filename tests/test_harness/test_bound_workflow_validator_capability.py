"""Tests for capability-aware extension of validate_bound_workflow (Phase 4).

Codes added by Phase 4:
- unknown_capability (error)
- capability_call_invalid (error)
- backend_not_supported (error)
- undeclared_side_effect (error)

Plus skip semantics: unknown_capability for one task does NOT suppress
capability-aware checks for other tasks.

Plus regression: registry=None → behavior identical to Phase 3.
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
    ir = WorkflowIR(
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


# ----------------------------------------------------------------- baseline


def test_baseline_with_registry_clean(tmp_path: Path) -> None:
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, registry, _ = _baseline_with_registry()
    report = BoundWorkflowValidator.validate(bw, ir, workspace_root=tmp_path, registry=registry)
    assert report.passed is True
    assert report.violations == []


# --------------------------------------------- regression: registry=None


def test_registry_none_yields_phase3_behavior(tmp_path: Path) -> None:
    """When registry=None, no capability-aware checks fire even if the
    capability_id is nonsense."""
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, _, _ = _baseline_with_registry()
    nonsense = bw.tasks[0].model_copy(update={"capability_id": "ghost"})
    bw_bad = bw.model_copy(update={"tasks": [nonsense]})
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path)
    # The four new codes must NOT appear.
    new_codes = {
        "unknown_capability",
        "capability_call_invalid",
        "backend_not_supported",
        "undeclared_side_effect",
    }
    assert not (set(_codes(report)) & new_codes)


# --------------------------------------------- new violation codes


def test_unknown_capability(tmp_path: Path) -> None:
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, registry, _ = _baseline_with_registry()
    nonsense = bw.tasks[0].model_copy(update={"capability_id": "ghost.capability"})
    bw_bad = bw.model_copy(update={"tasks": [nonsense]})
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path, registry=registry)
    assert "unknown_capability" in _codes(report)
    assert report.passed is False


def test_unknown_capability_suppresses_other_capability_checks_for_that_task(
    tmp_path: Path,
) -> None:
    """When a task's capability is unknown, the other three checks for
    THAT task are skipped (we can't check what we don't have)."""
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, registry, _ = _baseline_with_registry()
    nonsense = bw.tasks[0].model_copy(update={"capability_id": "ghost.capability"})
    bw_bad = bw.model_copy(update={"tasks": [nonsense]})
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path, registry=registry)
    codes = _codes(report)
    assert "unknown_capability" in codes
    # The other three must NOT fire (we can't reason about a ghost).
    assert "capability_call_invalid" not in codes
    assert "backend_not_supported" not in codes
    assert "undeclared_side_effect" not in codes


def test_capability_call_invalid_extra_key(tmp_path: Path) -> None:
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, registry, _ = _baseline_with_registry()
    # IR + bound input keys must match (Phase-3 input_key_mismatch check).
    # We add stray PARAMETER, not an input — parameters are validated by
    # validate_call against input_schema.
    bad_params = {
        "n_chains": ParameterValue(value=100, source="user_provided"),
        "stray_param": ParameterValue(value=1, source="agent_inferred"),
    }
    bad_task = bw.tasks[0].model_copy(update={"parameters": bad_params})
    bw_bad = bw.model_copy(update={"tasks": [bad_task]})
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path, registry=registry)
    assert "capability_call_invalid" in _codes(report)


def test_capability_call_invalid_missing_required(tmp_path: Path) -> None:
    """Capability requires n_chains; BoundTask.parameters dropped it.
    The IR task input key 'n_chains' still resolves so input_key_mismatch
    doesn't fire — we drop only from BoundTask.parameters.
    """
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, registry, _ = _baseline_with_registry()
    bad_task = bw.tasks[0].model_copy(update={"parameters": {}})
    bw_bad = bw.model_copy(update={"tasks": [bad_task]})
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path, registry=registry)
    assert "capability_call_invalid" in _codes(report)


def test_backend_not_supported(tmp_path: Path) -> None:
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, registry, _ = _baseline_with_registry()
    bw_bad = bw.model_copy(update={"execution_backend": "slurm"})  # cap only supports "local"
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path, registry=registry)
    assert "backend_not_supported" in _codes(report)


def test_undeclared_side_effect(tmp_path: Path) -> None:
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, registry, _ = _baseline_with_registry()
    # cap.side_effects = ["fs_write", "network"]; task claims "gpu" which is undeclared.
    bad_task = bw.tasks[0].model_copy(update={"side_effects": ["fs_write", "gpu"]})
    bw_bad = bw.model_copy(update={"tasks": [bad_task]})
    report = BoundWorkflowValidator.validate(bw_bad, ir, workspace_root=tmp_path, registry=registry)
    assert "undeclared_side_effect" in _codes(report)


def test_task_side_effects_subset_of_capability_is_clean(tmp_path: Path) -> None:
    """BoundTask declares a subset of capability.side_effects → no violation."""
    from molexp.harness.validators.bound_workflow import BoundWorkflowValidator

    ir, bw, registry, _ = _baseline_with_registry()
    # cap declares ["fs_write", "network"]; task declares ["fs_write"] only.
    bad_task = bw.tasks[0].model_copy(update={"side_effects": ["fs_write"]})
    bw_ok = bw.model_copy(update={"tasks": [bad_task]})
    report = BoundWorkflowValidator.validate(bw_ok, ir, workspace_root=tmp_path, registry=registry)
    assert "undeclared_side_effect" not in _codes(report)


def test_phase4_placeholder_comment_present() -> None:
    """ac-009: the Phase-3 placeholder is gone; a Phase-5+ marker is in."""
    import inspect

    from molexp.harness.validators import bound_workflow as mod

    src = inspect.getsource(mod)
    assert "Phase 4: capability" not in src
    assert "Phase 5+: ExecutionEnvironment cross-check" in src
