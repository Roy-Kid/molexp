"""Tests for the ``GenerateWorkflowSource`` stage (harness pipeline orchestration).

Drives the stage with a ``StubAgentGateway`` and locks the stage's four
distinct behaviours: fail-fast without a gateway; the writer sees both the
BoundWorkflow and the concrete ExperimentSpec; the persisted artifact carries
both inputs as lineage parents; a >1-task bound workflow fans out one call per
task and synthesizes a deterministic assembly module.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

# Mirror the schema fixture in test_validate_workflow_source so the
# generated source the stub returns is itself a valid molexp.workflow program.
_VALID_SOURCE = """\
from molexp.workflow import Task, TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def load(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    @wf.task(depends_on=["load"])
    async def square(ctx: TaskContext) -> list[int]:
        return [x * x for x in ctx.inputs]

    return wf
"""


def _workflow_source_canned() -> dict:
    return {
        "source": _VALID_SOURCE,
        "module_name": "generated_workflow",
        "bound_workflow_id": "bw-x",
        "symbols": ["WorkflowCompiler", "Task", "TaskContext"],
    }


def _seed_bw_ref(artifact_store):
    return artifact_store.put_json(
        kind="bound_workflow",
        obj={
            "id": "bw-x",
            "workflow_ir_id": "wf-x",
            "tasks": [],
            "edges": [],
            "execution_backend": "local",
            "environment": {},
            "resource_policy": {
                "backend": "local",
                "max_runtime_s": 3600,
                "denied_paths": ["/", "~/.ssh"],
            },
        },
        created_by="seed",
        parent_ids=[],
    )


def _seed_spec_ref(artifact_store):
    """The concrete ExperimentSpec the writer must implement faithfully."""
    return artifact_store.put_json(
        kind="experiment_spec",
        obj={
            "id": "spec-x",
            "experiment_report_id": "rep-x",
            "title": "t",
            "objective": "o",
            "variables": [{"name": "n_points", "value": {"value": 200, "source": "user_provided"}}],
            "controlled_conditions": [],
            "resolved_questions": [],
            "assumptions": [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _seed_two_task_bw(artifact_store):
    """A 2-task bound workflow (build -> pack) that triggers per-task fan-out."""

    def _task(tid, ir):
        return {
            "id": tid,
            "ir_task_id": ir,
            "capability_id": "molpy.x",
            "package": "molpy",
            "callable": "x",
            "parameters": {},
            "inputs": {},
            "outputs": {"out": "any"},
        }

    return artifact_store.put_json(
        kind="bound_workflow",
        obj={
            "id": "bw-2",
            "workflow_ir_id": "wf-2",
            "tasks": [_task("b-build", "build_monomer"), _task("b-pack", "pack_box")],
            "edges": [{"source_task_id": "b-build", "target_task_id": "b-pack"}],
            "execution_backend": "local",
            "environment": {},
            "resource_policy": {
                "backend": "local",
                "max_runtime_s": 3600,
                "denied_paths": ["/", "~/.ssh"],
            },
        },
        created_by="seed",
        parent_ids=[],
    )


@pytest.fixture()
def ctx_no_gw(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-gws",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


@pytest.fixture()
def ctx_with_gw(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.gateways.stub import StubAgentGateway
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    stub = StubAgentGateway(artifact_store=a)
    return HarnessRunContext(
        run_id="run-gws",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
        agent_gateway=stub,
    )


class TestGenerateWorkflowSource:
    def test_fail_fast_when_agent_gateway_is_none(self, ctx_no_gw) -> None:
        from molexp.harness.errors import StageExecutionError
        from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource

        _seed_bw_ref(ctx_no_gw.artifact_store)
        stage = GenerateWorkflowSource()
        with pytest.raises(StageExecutionError) as exc:
            asyncio.run(stage.run(ctx_no_gw))
        assert "agent_gateway" in str(exc.value)

    def test_writer_sees_bound_workflow_and_experiment_spec(self, ctx_with_gw) -> None:
        """The writer's AgentCallSpec inputs are the BoundWorkflow AND the ExperimentSpec.

        Regression: the writer used to receive only the bound workflow, while
        ``ReviewPlan`` judged its output against the experiment report — an
        information asymmetry that made the repair loop unable to converge on
        the spec's concrete numbers (production: two models x 4 attempts each
        kept inventing their own sigma grids / point counts / tolerances).
        """
        from molexp.harness.gateways.gateway import AgentGateway
        from molexp.harness.schemas import AgentCallResult, AgentCallSpec, WorkflowSource
        from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource

        bw_ref = _seed_bw_ref(ctx_with_gw.artifact_store)
        spec_ref = _seed_spec_ref(ctx_with_gw.artifact_store)
        real_gw = ctx_with_gw.agent_gateway
        real_gw.register(
            agent_name="workflow_source_writer",
            output=_workflow_source_canned(),
            output_kind="workflow_source",
        )
        captured: list[AgentCallSpec] = []

        class Cap:
            async def call(self, spec: AgentCallSpec) -> AgentCallResult:
                captured.append(spec)
                return await real_gw.call(spec)

        object.__setattr__(ctx_with_gw, "_frozen", False)
        ctx_with_gw.agent_gateway = cast(AgentGateway, Cap())
        object.__setattr__(ctx_with_gw, "_frozen", True)

        asyncio.run(GenerateWorkflowSource().run(ctx_with_gw))
        assert len(captured) == 1
        spec = captured[0]
        assert spec.agent_name == "workflow_source_writer"
        assert spec.input_artifact_ids == [bw_ref.id, spec_ref.id]
        assert spec.output_schema == WorkflowSource.model_json_schema()

    def test_persists_artifact_with_bound_workflow_and_spec_lineage(self, ctx_with_gw) -> None:
        from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource

        bw_ref = _seed_bw_ref(ctx_with_gw.artifact_store)
        spec_ref = _seed_spec_ref(ctx_with_gw.artifact_store)
        ctx_with_gw.agent_gateway.register(
            agent_name="workflow_source_writer",
            output=_workflow_source_canned(),
            output_kind="workflow_source",
        )
        ref = asyncio.run(GenerateWorkflowSource().run(ctx_with_gw))
        assert ref.kind == "workflow_source"
        assert bw_ref.id in ref.parent_ids
        assert spec_ref.id in ref.parent_ids

    def test_multi_task_fans_out_per_task_and_synthesizes_assembly(self, ctx_with_gw) -> None:
        """A >1-task bound workflow drives one workflow_source_file_writer call per
        task and assembles a deterministic ``workflow/__init__.py`` from the IR."""
        from molexp.harness.schemas import BoundWorkflow, WorkflowSource
        from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource

        bw_ref = _seed_two_task_bw(ctx_with_gw.artifact_store)
        _seed_spec_ref(ctx_with_gw.artifact_store)
        gw = ctx_with_gw.agent_gateway

        def _responder(spec, store):
            slice_wf = BoundWorkflow.model_validate_json(store.get(spec.input_artifact_ids[0]))
            task = slice_wf.tasks[0]
            # The model names the function differently on purpose — the stage MUST
            # rename it to the slug, so the assembly wiring still resolves.
            src = f"async def whatever_{task.ir_task_id}(ctx) -> dict:\n    return {{}}\n"
            return gw.make_response(
                {
                    "source": src,
                    "module_name": task.ir_task_id,
                    "bound_workflow_id": slice_wf.id,
                    "symbols": [],
                    "files": [{"path": f"workflow/{task.ir_task_id}.py", "source": src}],
                },
                output_kind="workflow_source_file",
            )

        gw.register_responder("workflow_source_file_writer", _responder)

        ref = asyncio.run(GenerateWorkflowSource().run(ctx_with_gw))
        assert ref.kind == "workflow_source"
        merged = WorkflowSource.model_validate_json(ctx_with_gw.artifact_store.get(ref.id))

        # One module per task, named by slug; function renamed to the slug.
        paths = {f.path for f in merged.files}
        assert paths == {"workflow/build_monomer.py", "workflow/pack_box.py"}
        by_path = {f.path: f.source for f in merged.files}
        assert "def build_monomer(" in by_path["workflow/build_monomer.py"]
        assert "def pack_box(" in by_path["workflow/pack_box.py"]

        # The synthesized assembly imports both (topo order) + wires the dependency.
        asm = merged.source
        assert "from workflow.build_monomer import build_monomer" in asm
        assert "from workflow.pack_box import pack_box" in asm
        assert "wf.task(build_monomer)" in asm
        assert "wf.task(depends_on=['build_monomer'])(pack_box)" in asm
        assert bw_ref.id in ref.parent_ids
