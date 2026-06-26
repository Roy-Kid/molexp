"""Tests for the new plan-pipeline stages (9-step redesign).

Covers GenerateExperimentSpec, ValidateExperimentSpec, ResolveCapabilities,
GenerateInputSet, ValidateInputSet, CompileWorkflow, GenerateExecutionReport.
Each uses a small ctx + the in-memory StubAgentGateway / DryRunExecutor.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def _ctx(tmp_path: Path, *, with_stub=False, registry=None):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.gateways.stub import StubAgentGateway
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    gateway = StubAgentGateway(artifact_store=a) if with_stub else None
    return HarnessRunContext(
        run_id="run-x",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=SQLiteEventLog(path=db),
        lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=a),
        agent_gateway=gateway,
        capability_registry=registry,
    )


def _seed_report(store, *, user_questions=None):
    return store.put_json(
        kind="experiment_report",
        obj={
            "title": "t",
            "objective": "o",
            "system_description": "s",
            "experimental_design": "e",
            "variables": ["E_field"],
            "user_questions": user_questions or [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _spec_obj(*, resolved=None, variables=None):
    return {
        "id": "spec-1",
        "experiment_report_id": "rep-1",
        "title": "t",
        "objective": "o",
        "variables": variables
        if variables is not None
        else [{"name": "E_field", "value": {"value": 1e6, "source": "agent_inferred"}}],
        "controlled_conditions": [],
        "resolved_questions": resolved or [],
        "assumptions": [],
    }


def _seed_spec(store, **kw: object):
    return store.put_json(
        kind="experiment_spec", obj=_spec_obj(**kw), created_by="seed", parent_ids=[]
    )


def _seed_ir(store, *, inputs=("n_steps",)):
    return store.put_json(
        kind="workflow_ir",
        obj={
            "id": "wf-ir-1",
            "name": "demo",
            "objective": "x",
            "inputs": {k: {"value": 500, "source": "user_provided"} for k in inputs},
            "tasks": [],
            "edges": [],
            "expected_outputs": [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _seed_bound(store):
    return store.put_json(
        kind="bound_workflow",
        obj={
            "id": "bw-1",
            "workflow_ir_id": "wf-ir-1",
            "tasks": [],
            "edges": [],
            "execution_backend": "local",
            "environment": {"python_version": "3.12"},
            "resource_policy": {
                "backend": "local",
                "max_runtime_s": 3600,
                "denied_paths": ["/", "~/.ssh"],
            },
            "review_flags": ["check field strength"],
        },
        created_by="seed",
        parent_ids=[],
    )


# ---------------------------------------------------------- GenerateExperimentSpec


def test_generate_experiment_spec(tmp_path: Path) -> None:
    from molexp.harness.stages.generate_experiment_spec import GenerateExperimentSpec

    ctx = _ctx(tmp_path, with_stub=True)
    report = _seed_report(ctx.artifact_store, user_questions=["which water model?"])
    ctx.agent_gateway.register(
        agent_name="experiment_spec_generator",
        output=_spec_obj(resolved=[{"question": "which water model?", "answer": "SPC/E"}]),
        output_kind="experiment_spec",
    )
    ref = asyncio.run(GenerateExperimentSpec().run(ctx))
    assert ref.kind == "experiment_spec"
    assert report.id in ref.parent_ids


def test_generate_experiment_spec_fail_fast_without_gateway(tmp_path: Path) -> None:
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.generate_experiment_spec import GenerateExperimentSpec

    ctx = _ctx(tmp_path)
    _seed_report(ctx.artifact_store)
    with pytest.raises(StageExecutionError):
        asyncio.run(GenerateExperimentSpec().run(ctx))


# ---------------------------------------------------------- ValidateExperimentSpec


def test_validate_experiment_spec_passes_when_questions_resolved(tmp_path: Path) -> None:
    from molexp.harness.schemas import ValidationReport
    from molexp.harness.stages.validate_experiment_spec import ValidateExperimentSpec

    ctx = _ctx(tmp_path)
    _seed_report(ctx.artifact_store, user_questions=["which water model?"])
    _seed_spec(ctx.artifact_store, resolved=[{"question": "which water model?", "answer": "SPC/E"}])
    ref = asyncio.run(ValidateExperimentSpec().run(ctx))
    report = ValidationReport.model_validate_json(ctx.artifact_store.get(ref.id))
    assert report.passed


def test_validate_experiment_spec_flags_unresolved_question(tmp_path: Path) -> None:
    from molexp.harness.errors import StagePersistedFailureError
    from molexp.harness.stages.validate_experiment_spec import ValidateExperimentSpec

    ctx = _ctx(tmp_path)
    _seed_report(ctx.artifact_store, user_questions=["which water model?"])
    _seed_spec(ctx.artifact_store, resolved=[])  # not resolved
    with pytest.raises(StagePersistedFailureError):
        asyncio.run(ValidateExperimentSpec().run(ctx))


# ---------------------------------------------------------- ResolveCapabilities


def _cap(cap_id: str, name: str, desc: str):
    from molexp.harness.schemas import ToolCapability

    return ToolCapability(
        id=cap_id,
        package=cap_id.split(".", 1)[0],
        name=name,
        description=desc,
        input_schema={"type": "object"},
        output_schema={},
        callable_path=cap_id,
        supported_backends=["local"],
        tags=["class"],
    )


def _two_cap_registry():
    from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

    return InMemoryCapabilityRegistry(
        [
            _cap("molpy.core.cg.CoarseGrain", "CoarseGrain", "Build a CG structure."),
            _cap("molpy.io.writers.write_gromacs", "write_gromacs", "Write a GROMACS topology."),
        ]
    )


def test_resolve_capabilities_llm_selects_subset(tmp_path: Path) -> None:
    """With a gateway, the LLM-selected subset (with reasons) is what lands in the catalog."""
    from molexp.harness.stages.resolve_capabilities import ResolveCapabilities

    ctx = _ctx(tmp_path, with_stub=True, registry=_two_cap_registry())
    _seed_spec(ctx.artifact_store)
    ctx.agent_gateway.register(
        agent_name="capability_selector",
        output={
            "selected": [{"id": "molpy.core.cg.CoarseGrain", "reason": "builds the CG beads"}],
            "notes": "",
        },
        output_kind="capability_selection",
    )
    ref = asyncio.run(ResolveCapabilities().run(ctx))
    assert ref.kind == "capability_catalog"
    text = ctx.artifact_store.get(ref.id).decode("utf-8")
    assert "molpy.core.cg.CoarseGrain" in text
    assert "builds the CG beads" in text  # the LLM's reason is shown
    assert "write_gromacs" not in text  # the un-selected capability is omitted


def test_resolve_capabilities_full_catalog_without_gateway(tmp_path: Path) -> None:
    """Registry but no LLM gateway → loud note + the full grounded catalog (no silent drop)."""
    from molexp.harness.stages.resolve_capabilities import ResolveCapabilities

    ctx = _ctx(tmp_path, registry=_two_cap_registry())
    ref = asyncio.run(ResolveCapabilities().run(ctx))
    text = ctx.artifact_store.get(ref.id).decode("utf-8")
    assert "LLM selector unavailable" in text
    assert "molpy.core.cg.CoarseGrain" in text and "write_gromacs" in text


def test_resolve_capabilities_without_registry(tmp_path: Path) -> None:
    from molexp.harness.stages.resolve_capabilities import ResolveCapabilities

    ctx = _ctx(tmp_path)
    ref = asyncio.run(ResolveCapabilities().run(ctx))
    assert ref.kind == "capability_catalog"
    assert "No capability registry" in ctx.artifact_store.get(ref.id).decode("utf-8")


# ---------------------------------------------------------- GenerateInputSet


def test_generate_input_set(tmp_path: Path) -> None:
    from molexp.harness.stages.generate_input_set import GenerateInputSet

    ctx = _ctx(tmp_path, with_stub=True)
    spec = _seed_spec(ctx.artifact_store)
    ir = _seed_ir(ctx.artifact_store)
    ctx.agent_gateway.register(
        agent_name="input_set_generator",
        output={
            "id": "is-1",
            "experiment_spec_id": "spec-1",
            "title": "sweep",
            "sweep_axes": [{"name": "n_steps", "values": [1000, 2000], "source": "user_provided"}],
            "strategy": "grid",
            "total_runs": 2,
        },
        output_kind="input_set",
    )
    ref = asyncio.run(GenerateInputSet().run(ctx))
    assert ref.kind == "input_set"
    assert spec.id in ref.parent_ids and ir.id in ref.parent_ids


# ---------------------------------------------------------- ValidateInputSet


def _seed_input_set(store, *, axis_name="n_steps", total_runs=2):
    return store.put_json(
        kind="input_set",
        obj={
            "id": "is-1",
            "experiment_spec_id": "spec-1",
            "title": "sweep",
            "sweep_axes": [{"name": axis_name, "values": [1000, 2000], "source": "user_provided"}],
            "strategy": "grid",
            "total_runs": total_runs,
        },
        created_by="seed",
        parent_ids=[],
    )


def test_validate_input_set_passes(tmp_path: Path) -> None:
    from molexp.harness.schemas import ValidationReport
    from molexp.harness.stages.validate_input_set import ValidateInputSet

    ctx = _ctx(tmp_path)
    _seed_ir(ctx.artifact_store, inputs=("n_steps",))
    _seed_input_set(ctx.artifact_store, axis_name="n_steps", total_runs=2)
    ref = asyncio.run(ValidateInputSet().run(ctx))
    assert ValidationReport.model_validate_json(ctx.artifact_store.get(ref.id)).passed


def test_validate_input_set_flags_unknown_axis(tmp_path: Path) -> None:
    from molexp.harness.errors import StagePersistedFailureError
    from molexp.harness.stages.validate_input_set import ValidateInputSet

    ctx = _ctx(tmp_path)
    _seed_ir(ctx.artifact_store, inputs=("n_steps",))
    _seed_input_set(ctx.artifact_store, axis_name="temperature", total_runs=2)  # not an IR input
    with pytest.raises(StagePersistedFailureError):
        asyncio.run(ValidateInputSet().run(ctx))


# ---------------------------------------------------------- GenerateExecutionReport


def test_generate_execution_report_with_target(tmp_path: Path) -> None:
    from molexp.harness.schemas import ExecutionReport
    from molexp.harness.stages.generate_execution_report import GenerateExecutionReport
    from molexp.workspace.models import ComputeTarget

    ctx = _ctx(tmp_path)
    bw = _seed_bound(ctx.artifact_store)
    _seed_input_set(ctx.artifact_store, total_runs=2)
    target = ComputeTarget(
        name="hpc1",
        host="me@cluster.example.org",
        scheduler="slurm",
        scratch_root="/scratch/me/molexp",
        default_scheduling={"account": "proj-1234", "queue": "normal"},
    )
    ref = asyncio.run(GenerateExecutionReport(compute_target=target).run(ctx))
    assert ref.kind == "execution_report"
    assert bw.id in ref.parent_ids
    report = ExecutionReport.model_validate_json(ctx.artifact_store.get(ref.id))
    assert report.target_name == "hpc1"
    assert report.scheduler == "slurm"
    assert report.account == "proj-1234"
    assert report.total_runs == 2


def test_generate_execution_report_local_default(tmp_path: Path) -> None:
    from molexp.harness.schemas import ExecutionReport
    from molexp.harness.stages.generate_execution_report import GenerateExecutionReport

    ctx = _ctx(tmp_path)
    _seed_bound(ctx.artifact_store)
    ref = asyncio.run(GenerateExecutionReport().run(ctx))
    report = ExecutionReport.model_validate_json(ctx.artifact_store.get(ref.id))
    assert report.target_name == "local"
    assert report.scheduler == "local"
    assert report.total_runs == 1


# ---------------------------------------------------------- CompileWorkflow


def test_compile_workflow_dry_executor(tmp_path: Path) -> None:
    """CompileWorkflow wiring: DryRunExecutor → succeeded execution_result, mode=compile."""
    from molexp.harness.executors import DryRunExecutor
    from molexp.harness.schemas import ExecutionResult
    from molexp.harness.stages.compile_workflow import CompileWorkflow

    ctx = _ctx(tmp_path)
    ctx.artifact_store.put_json(
        kind="workflow_source",
        obj={
            "source": "def build_workflow():\n    return None\n",
            "module_name": "generated_workflow",
            "bound_workflow_id": "bw-1",
            "symbols": [],
        },
        created_by="seed",
        parent_ids=[],
    )
    ref = asyncio.run(CompileWorkflow(DryRunExecutor()).run(ctx))
    assert ref.kind == "execution_result"
    result = ExecutionResult.model_validate_json(ctx.artifact_store.get(ref.id))
    assert result.status == "succeeded"
    assert result.metadata.get("mode") == "compile"
