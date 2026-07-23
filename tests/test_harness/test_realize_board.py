"""RED tests for ``RealizeBoard`` + ``input_set_to_param_space`` (plan-emergent-06).

``RealizeBoard`` (``molexp.harness.stages.realize_board``) is the deterministic
map -> reduce -> compile realizer of a *frozen* task board:

- **Map (parallel)** — ``asyncio.gather`` over EVERY ``BoundTask``, one
  ``realize_one_task`` worker per task (coverage guaranteed by structure, not
  chosen by a model).
- **Reduce** — collect the greens into a single ``workflow_source`` (the
  ``build_workflow`` assembly + one per-task file each) + a single
  ``test_source``.
- **Block branch** — if >= 1 task blocks, persist a durable
  ``intervention_request`` artifact then raise ``TaskRealizationBlockedError``
  BEFORE any compile (never compile / assemble / fall back on a block).
- **Compile** — all-green only: ``MaterializeExecution().run(ctx)`` then
  ``CompileWorkflow(executor).run(ctx)`` (``--compile-only``), returning the
  compile ``execution_result`` ref.

``input_set_to_param_space`` bridges an ``InputSet`` to the workspace
``ParamSpace`` family (grid -> ``GridSpace``; uniform -> ``UniformSpace``).

Stub-driven + offline: ``StubAgentGateway`` codegen responders + ``DryRunExecutor``
(happy) / a slug-keyed canned ``Executor`` (block). NO real LLM, NO real pytest
subprocess. The production modules do not exist yet — an ``ImportError`` at
collection is the valid RED signal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.schemas import (
    CommandResult,
    ExecutionResult,
    InterventionRequest,
    WorkflowSource,
)
from molexp.harness.stages.input_space import input_set_to_param_space
from molexp.harness.stages.realize_board import RealizeBoard

pytestmark = pytest.mark.asyncio

_MODULE_SRC = "async def make_task(ctx) -> dict:\n    return {}\n"
_TEST_SRC = "def test_it():\n    assert True\n"
_FAILURE_MARKER = "MARKER_TASK_FAILURE_XYZ"


class _CapturingGateway:
    """Wrap a real gateway, recording every ``AgentCallSpec`` that flows through."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list = []

    async def call(self, spec):
        self.calls.append(spec)
        return await self._inner.call(spec)  # type: ignore[attr-defined]


class _SlugScriptedExecutor:
    """Canned ``Executor``: red for one target slug's test, green for the rest.

    Keys purely on the pytest ``cmd`` (``tests/test_<slug>.py``), so it is
    robust to which generated source happens to be on disk. Records every spec
    so a test can assert the ``--compile-only`` command never ran.
    """

    def __init__(self, red_slug: str, *, red_stdout: str = "") -> None:
        self._red_slug = red_slug
        self._red_stdout = red_stdout
        self.specs: list = []

    async def execute(self, spec, *, artifact_store) -> CommandResult:
        self.specs.append(spec)
        is_red = f"test_{self._red_slug}.py" in " ".join(spec.cmd)
        now = datetime.now(UTC)
        out = artifact_store.put_text(
            kind="stdout",
            text=self._red_stdout if is_red else "",
            created_by="scripted",
            parent_ids=[],
        )
        err = artifact_store.put_text(kind="stderr", text="", created_by="scripted", parent_ids=[])
        return CommandResult(
            exit_code=1 if is_red else 0,
            started_at=now,
            ended_at=now,
            stdout_artifact=out,
            stderr_artifact=err,
        )


def _make_ctx(tmp_path: Path, store, gateway):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=store)
    return HarnessRunContext(
        run_id="run-realize-board",
        workspace_root=tmp_path,
        artifact_store=store,
        event_log=e,
        lineage_store=p,
        agent_gateway=gateway,  # type: ignore[arg-type]
    )


def _seed_three_task_board(store):
    """A 3-task frozen BoundWorkflow (alpha -> beta -> gamma)."""
    from molexp.harness.schemas import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )

    def _task(tid: str, ir: str) -> BoundTask:
        return BoundTask(
            id=tid,
            ir_task_id=ir,
            capability_id="molpy.x",
            package="molpy",
            callable="x",
            parameters={},
            inputs={},
            outputs={"out": "any"},
        )

    bound = BoundWorkflow(
        id="bw-3",
        workflow_ir_id="wf-3",
        tasks=[_task("b-alpha", "alpha"), _task("b-beta", "beta"), _task("b-gamma", "gamma")],
        edges=[
            {"source_task_id": "b-alpha", "target_task_id": "b-beta"},
            {"source_task_id": "b-beta", "target_task_id": "b-gamma"},
        ],
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend="local", max_runtime_s=3600, denied_paths=["/", "~/.ssh"]
        ),
    )
    return store.put_json(
        kind="bound_workflow",
        obj=bound.model_dump(mode="json"),
        created_by="seed",
        parent_ids=[],
    )


def _seed_spec(store):
    return store.put_json(
        kind="experiment_spec",
        obj={
            "id": "spec-3",
            "experiment_report_id": "rep-3",
            "title": "t",
            "objective": "o",
            "variables": [],
            "controlled_conditions": [],
            "resolved_questions": [],
            "assumptions": [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _seed_workflow_ir(store):
    return store.put_json(
        kind="workflow_ir",
        obj={
            "id": "wf-ir-3",
            "name": "demo",
            "objective": "realize a 3-task board",
            "inputs": {},
            "tasks": [],
            "edges": [],
            "expected_outputs": [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _register_codegen(stub) -> None:
    """Per-call responders returning generic (renamed-downstream) source."""

    def _wf(spec, store):
        return stub.make_response(
            {
                "source": _MODULE_SRC,
                "module_name": "m",
                "bound_workflow_id": "bw-3",
                "symbols": [],
            },
            output_kind="workflow_source_file",
        )

    def _test(spec, store):
        return stub.make_response(
            {
                "source": _TEST_SRC,
                "module_name": "test_m",
                "test_spec_id": "ts-3",
                "bound_workflow_id": "bw-3",
                "symbols": [],
            },
            output_kind="test_source_file",
        )

    stub.register_responder("workflow_source_file_writer", _wf)
    stub.register_responder("test_code_file_writer", _test)


def _new_store(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=tmp_path / "artifacts")


class TestRealizeBoardHappy:
    async def test_maps_all_tasks_reduces_and_compiles_green(self, tmp_path: Path) -> None:
        from molexp.harness.executors.dry_run import DryRunExecutor
        from molexp.harness.gateways.stub import StubAgentGateway

        store = _new_store(tmp_path)
        stub = StubAgentGateway(artifact_store=store)
        _register_codegen(stub)
        cap = _CapturingGateway(stub)
        ctx = _make_ctx(tmp_path, store, cap)

        _seed_three_task_board(ctx.artifact_store)
        _seed_spec(ctx.artifact_store)
        _seed_workflow_ir(ctx.artifact_store)

        ref = await RealizeBoard(DryRunExecutor()).run(ctx)

        # Full coverage: the codegen subagent fired once per board task.
        wf_calls = [c for c in cap.calls if c.agent_name == "workflow_source_file_writer"]
        test_calls = [c for c in cap.calls if c.agent_name == "test_code_file_writer"]
        assert len(wf_calls) == 3
        assert len(test_calls) == 3

        # Reduce: exactly one workflow_source (assembly + 3 per-task files) + one test_source.
        wf = WorkflowSource.model_validate_json(
            ctx.artifact_store.get(ctx.artifact_store.latest_by_kind("workflow_source").id)
        )
        assert "def build_workflow" in wf.source
        assert {f.path for f in wf.files} == {
            "workflow/alpha.py",
            "workflow/beta.py",
            "workflow/gamma.py",
        }
        assert ctx.artifact_store.latest_by_kind("test_source") is not None

        # Compile: the injected DryRunExecutor yields a succeeded execution_result.
        assert ref.kind == "execution_result"
        result = ExecutionResult.model_validate_json(ctx.artifact_store.get(ref.id))
        assert result.status == "succeeded"


class TestRealizeBoardBlock:
    async def test_blocked_task_raises_durable_intervention_before_compile(
        self, tmp_path: Path
    ) -> None:
        from molexp.harness.errors import TaskRealizationBlockedError
        from molexp.harness.gateways.stub import StubAgentGateway

        store = _new_store(tmp_path)
        stub = StubAgentGateway(artifact_store=store)
        _register_codegen(stub)
        ctx = _make_ctx(tmp_path, store, stub)

        _seed_three_task_board(ctx.artifact_store)
        _seed_spec(ctx.artifact_store)
        _seed_workflow_ir(ctx.artifact_store)

        executor = _SlugScriptedExecutor("beta", red_stdout=_FAILURE_MARKER)

        with pytest.raises(TaskRealizationBlockedError) as exc_info:
            await RealizeBoard(executor, attempts=2).run(ctx)

        # A durable intervention_request was persisted (persist-then-raise).
        ivr_refs = ctx.artifact_store.list_by_kind("intervention_request")
        assert len(ivr_refs) == 1
        request = InterventionRequest.model_validate_json(ctx.artifact_store.get(ivr_refs[0].id))
        assert request.bound_workflow_id == "bw-3"
        blocked = {bt.slug: bt for bt in request.blocked_tasks}
        assert set(blocked) == {"beta"}
        beta = blocked["beta"]
        assert beta.task_id == "b-beta"
        assert _FAILURE_MARKER in beta.error
        assert beta.question

        # The error names the blocked task and points at the persisted request.
        assert exc_info.value.request_ref.kind == "intervention_request"
        assert "b-beta" in exc_info.value.blocked_task_ids

        # Compile NEVER ran: no execution_result, and no --compile-only command.
        assert ctx.artifact_store.list_by_kind("execution_result") == []
        assert not any("--compile-only" in " ".join(s.cmd) for s in executor.specs)


class TestInputSetToParamSpace:
    def test_grid_yields_gridspace_with_cartesian_cell_count(self) -> None:
        from molexp.harness.schemas import InputSet
        from molexp.workspace.param import GridSpace

        iset = InputSet(
            id="is-grid",
            experiment_spec_id="spec-3",
            title="grid",
            sweep_axes=[
                {"name": "temp", "values": [300, 350]},
                {"name": "sigma", "values": [0.8, 0.9, 1.0]},
            ],
            strategy="grid",
            total_runs=6,
        )
        space = input_set_to_param_space(iset)
        assert isinstance(space, GridSpace)
        assert len(space) == 6

    def test_uniform_yields_uniformspace_with_total_runs_and_seed(self) -> None:
        from molexp.harness.schemas import InputSet
        from molexp.workspace.param import UniformSpace

        iset = InputSet(
            id="is-uni",
            experiment_spec_id="spec-3",
            title="uniform",
            sweep_axes=[{"name": "temp", "values": [300, 350, 400]}],
            strategy="uniform",
            total_runs=5,
            random_seed=42,
        )
        space = input_set_to_param_space(iset)
        assert isinstance(space, UniformSpace)
        assert space.n_samples == 5
        assert space.seed == 42
        assert len(space) == 5

    def test_single_value_axis_is_one_cell(self) -> None:
        from molexp.harness.schemas import InputSet

        iset = InputSet(
            id="is-single",
            experiment_spec_id="spec-3",
            title="single",
            sweep_axes=[{"name": "temp", "values": [300]}],
            strategy="grid",
            total_runs=1,
        )
        space = input_set_to_param_space(iset)
        assert len(space) == 1

    def test_no_axes_is_a_single_degenerate_cell(self) -> None:
        from molexp.harness.schemas import InputSet

        iset = InputSet(
            id="is-none",
            experiment_spec_id="spec-3",
            title="fixed-only",
            sweep_axes=[],
            fixed_params={"n_steps": 1000},
            strategy="grid",
            total_runs=1,
        )
        space = input_set_to_param_space(iset)
        assert len(space) == 1


class TestRealizeBoardExport:
    def test_realize_board_exported_from_stages(self) -> None:
        import molexp.harness.stages as stages_mod

        assert "RealizeBoard" in stages_mod.__all__
        assert stages_mod.RealizeBoard is RealizeBoard
