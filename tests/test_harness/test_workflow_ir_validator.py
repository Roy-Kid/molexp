"""Tests for validate_workflow_ir (Phase 3 §11.2).

One unit test per failure code. Each test starts from a clean baseline IR
and mutates ONE field to trigger ONE violation, asserting the report
surfaces exactly that code.

Codes:
- duplicate_task_id (error)
- unknown_edge_source (error)
- unknown_edge_target (error)
- cyclic_dependency (error)
- missing_producer (error, only required=True)
- unresolved_input (error)
- agent_inferred_not_flagged (warning — does NOT flip passed=False)
- shell_command_in_ir (error)
- backend_leak_in_ir (error)
"""

from __future__ import annotations


def _baseline_ir():
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import (
        DependencyEdge,
        ExpectedOutput,
        TaskIR,
        WorkflowIR,
    )

    build = TaskIR(
        id="build",
        name="Build system",
        purpose="Pack water box",
        task_type="molecule_builder",
        inputs={"n_chains": ParameterValue(value=100, source="user_provided")},
        outputs={"structure": "structure.pdb"},
    )
    run_md = TaskIR(
        id="run_md",
        name="Run NEMD",
        purpose="Propagate dynamics under field",
        task_type="md_runner",
        inputs={"structure": ParameterValue(value="structure.pdb", source="user_provided")},
        outputs={"trajectory": "traj.dcd"},
    )
    return WorkflowIR(
        id="wf-001",
        name="water_nemd",
        objective="Compute ionic mobility under field",
        inputs={
            "temperature_K": ParameterValue(value=300.0, source="user_provided"),
            "n_chains": ParameterValue(value=100, source="user_provided"),
        },
        tasks=[build, run_md],
        edges=[DependencyEdge(source_task_id="build", target_task_id="run_md")],
        expected_outputs=[
            ExpectedOutput(name="trajectory", kind="dataset", description="MD trajectory"),
        ],
    )


def _codes(report) -> list[str]:
    return [v.code for v in report.violations]


# ------------------------------------------------------------------ baseline


def test_baseline_ir_is_clean() -> None:
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    report = validate_workflow_ir(_baseline_ir())
    assert report.passed is True
    assert report.violations == []
    assert report.target_kind == "workflow_ir"
    assert report.target_id == "wf-001"


def test_validate_workflow_ir_signature_and_import() -> None:
    """ac-004: importable from both paths; signature returns ValidationReport."""
    from molexp.harness import validate_workflow_ir as top
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.validators import validate_workflow_ir as via_pkg
    from molexp.harness.validators.workflow_ir import validate_workflow_ir as via_mod

    assert top is via_pkg is via_mod

    report = top(_baseline_ir())
    assert isinstance(report, ValidationReport)


# ------------------------------------------------------------------ violations


def test_duplicate_task_id() -> None:
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    dup_task = TaskIR(
        id="build",  # duplicate
        name="Other",
        purpose="x",
        task_type="x",
        inputs={},
        outputs={"x": "x"},
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, dup_task]})
    report = validate_workflow_ir(ir)
    assert report.passed is False
    assert "duplicate_task_id" in _codes(report)


def test_unknown_edge_source() -> None:
    from molexp.harness.schemas.workflow_ir import DependencyEdge
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    ir = ir.model_copy(
        update={"edges": [DependencyEdge(source_task_id="ghost", target_task_id="run_md")]}
    )
    report = validate_workflow_ir(ir)
    assert "unknown_edge_source" in _codes(report)
    assert report.passed is False


def test_unknown_edge_target() -> None:
    from molexp.harness.schemas.workflow_ir import DependencyEdge
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    ir = ir.model_copy(
        update={"edges": [DependencyEdge(source_task_id="build", target_task_id="ghost")]}
    )
    report = validate_workflow_ir(ir)
    assert "unknown_edge_target" in _codes(report)


def test_cyclic_dependency() -> None:
    from molexp.harness.schemas.workflow_ir import DependencyEdge
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    ir = ir.model_copy(
        update={
            "edges": [
                DependencyEdge(source_task_id="build", target_task_id="run_md"),
                DependencyEdge(source_task_id="run_md", target_task_id="build"),  # cycle
            ]
        }
    )
    report = validate_workflow_ir(ir)
    assert "cyclic_dependency" in _codes(report)


def test_missing_producer() -> None:
    from molexp.harness.schemas.workflow_ir import ExpectedOutput
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    ir = ir.model_copy(
        update={
            "expected_outputs": [
                ExpectedOutput(name="ghost_output", kind="dataset", description="x", required=True),
            ],
        }
    )
    report = validate_workflow_ir(ir)
    assert "missing_producer" in _codes(report)


def test_missing_producer_skips_optional_outputs() -> None:
    """required=False ExpectedOutput does NOT trigger missing_producer."""
    from molexp.harness.schemas.workflow_ir import ExpectedOutput
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    ir = ir.model_copy(
        update={
            "expected_outputs": [
                ExpectedOutput(
                    name="optional_thing", kind="dataset", description="x", required=False
                ),
            ],
        }
    )
    report = validate_workflow_ir(ir)
    assert "missing_producer" not in _codes(report)


def test_unresolved_input() -> None:
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    # run_md depends on "structure" (from "build"). Add a 3rd task whose
    # input name is unresolved.
    orphan = TaskIR(
        id="orphan",
        name="Orphan",
        purpose="x",
        task_type="x",
        inputs={"ghost_input": ParameterValue(value="?", source="user_provided")},
        outputs={"y": "y"},
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, orphan]})
    report = validate_workflow_ir(ir)
    assert "unresolved_input" in _codes(report)


def test_agent_inferred_not_flagged_is_warning() -> None:
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    risky = TaskIR(
        id="risky",
        name="risky",
        purpose="x",
        task_type="x",
        inputs={"field_strength": ParameterValue(value=1e6, source="agent_inferred")},
        outputs={"x": "x"},
        # review_flags intentionally missing field_strength
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, risky]})
    report = validate_workflow_ir(ir)
    flagged = [v for v in report.violations if v.code == "agent_inferred_not_flagged"]
    assert flagged, "expected an agent_inferred_not_flagged warning"
    assert flagged[0].severity == "warning"
    # If only warnings remain, passed should still be True.
    if all(v.severity == "warning" for v in report.violations):
        assert report.passed is True


def test_agent_inferred_in_review_flags_is_clean() -> None:
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    flagged_task = TaskIR(
        id="flagged",
        name="flagged",
        purpose="x",
        task_type="x",
        inputs={"field_strength": ParameterValue(value=1e6, source="agent_inferred")},
        outputs={"x": "x"},
        review_flags=["field_strength"],  # acknowledged
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, flagged_task]})
    report = validate_workflow_ir(ir)
    assert "agent_inferred_not_flagged" not in _codes(report)


def test_shell_command_in_ir_bash() -> None:
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    dirty = TaskIR(
        id="dirty",
        name="dirty",
        purpose="Run bash -c 'rm -rf /'",
        task_type="x",
        inputs={},
        outputs={"x": "x"},
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, dirty]})
    report = validate_workflow_ir(ir)
    assert "shell_command_in_ir" in _codes(report)


def test_shell_command_in_ir_subprocess() -> None:
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    dirty = TaskIR(
        id="dirty",
        name="dirty",
        purpose="invoke subprocess.run(['lmp', '-in', 'in.lammps'])",
        task_type="x",
        inputs={},
        outputs={"x": "x"},
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, dirty]})
    report = validate_workflow_ir(ir)
    assert "shell_command_in_ir" in _codes(report)


def test_shell_command_in_ir_semicolon() -> None:
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    dirty = TaskIR(
        id="dirty",
        name="run a; run b",
        purpose="x",
        task_type="x",
        inputs={},
        outputs={"x": "x"},
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, dirty]})
    report = validate_workflow_ir(ir)
    assert "shell_command_in_ir" in _codes(report)


def test_backend_leak_in_ir_slurm() -> None:
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    dirty = TaskIR(
        id="dirty",
        name="dirty",
        purpose="submit via slurm",
        task_type="x",
        inputs={},
        outputs={"x": "x"},
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, dirty]})
    report = validate_workflow_ir(ir)
    assert "backend_leak_in_ir" in _codes(report)


def test_backend_leak_in_ir_sbatch() -> None:
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    dirty = TaskIR(
        id="dirty",
        name="sbatch run.sh",
        purpose="x",
        task_type="x",
        inputs={},
        outputs={"x": "x"},
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, dirty]})
    report = validate_workflow_ir(ir)
    assert "backend_leak_in_ir" in _codes(report)


def test_backend_leak_in_ir_module_load() -> None:
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    dirty = TaskIR(
        id="dirty",
        name="dirty",
        purpose="needs module load gcc/11",
        task_type="x",
        inputs={},
        outputs={"x": "x"},
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, dirty]})
    report = validate_workflow_ir(ir)
    assert "backend_leak_in_ir" in _codes(report)


def test_shell_command_in_parameter_value_inputs() -> None:
    """An agent that smuggles a shell injection through ``inputs[k].value``
    (rather than the natural-language ``purpose``) must still trip the
    deny list. Regression: the previous ``_string_fields`` only scanned
    free-text fields, leaving parameter values as a soft channel."""
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    dirty = TaskIR(
        id="dirty",
        name="dirty",
        purpose="harmless",
        task_type="x",
        inputs={
            "command": ParameterValue(
                value="cleanup && rm -rf /tmp/data",
                source="agent_inferred",
                approved=True,
            ),
        },
        outputs={"x": "x"},
        review_flags=["command"],
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, dirty]})
    report = validate_workflow_ir(ir)
    assert "shell_command_in_ir" in _codes(report)


def test_backtick_in_acceptance_criteria_is_not_flagged() -> None:
    """Natural-language acceptance criteria often quote filenames with
    backticks (e.g. ``output matches `expected.csv` ``). The old
    deny-list included ``` ` ``` which produced false positives on
    perfectly fine prose. Regression: it must no longer trip."""
    from molexp.harness.schemas.workflow_ir import TaskIR
    from molexp.harness.validators.workflow_ir import validate_workflow_ir

    ir = _baseline_ir()
    clean = TaskIR(
        id="prose",
        name="prose",
        purpose="document the run",
        task_type="x",
        inputs={},
        outputs={"x": "x"},
        acceptance_criteria=["output matches `expected.csv` exactly"],
    )
    ir = ir.model_copy(update={"tasks": [*ir.tasks, clean]})
    report = validate_workflow_ir(ir)
    assert "shell_command_in_ir" not in _codes(report)
