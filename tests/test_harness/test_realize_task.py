"""RED tests for ``realize_one_task`` — the per-task self-repair worker.

``realize_one_task`` (``molexp.harness.stages.realize_task``) is the map-body
of plan-emergent-06's realization phase: a coroutine (NOT a ``Stage``) that,
for ONE frozen ``BoundTask``, loops up to ``attempts`` times — codegen the task
module + its unit test through the injected ``AgentGateway``, run that task's
unit test through the injected ``Executor``, and on red feed the failure back
and regenerate (``use_mcp=True`` on the repair turns). It returns a result
object:

- green  → ``result.status == "green"``, carrying ``module_src`` + ``test_src``.
- blocked → ``result.status == "blocked"``, carrying a ``BlockedTask`` payload
  (``task_id`` / ``ir_task_id`` / ``slug`` / ``attempts`` / last-failure text +
  a failure artifact ref + a ``question``). The load-bearing new behavior: at
  the attempt ceiling it *returns* blocked and NEVER raises, so the board's map
  can continue the other tasks.

Pinned call contract (RED — the caller writes GREEN to match)::

    async def realize_one_task(
        ctx, *, task, bound, bound_ref, experiment_spec_id,
        executor, attempts=3, timeout_s=300,
    ) -> <result with .status / .module_src / .test_src / .blocked>

Stub-driven + offline: ``StubAgentGateway`` for codegen, ``DryRunExecutor`` or a
canned in-memory ``Executor`` for the per-task pytest (NO real LLM, NO real
pytest subprocess). The production module does not exist yet — an ``ImportError``
at collection is the valid RED signal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.schemas import BlockedTask, CommandResult
from molexp.harness.stages.generate_workflow_source import _slug
from molexp.harness.stages.realize_task import realize_one_task

pytestmark = pytest.mark.asyncio

# A generic task body + test the codegen stub returns; the worker renames the
# module's top function to the task slug, so the exact name here is irrelevant.
_MODULE_SRC = "async def make_task(ctx) -> dict:\n    return {}\n"
_TEST_SRC = "def test_it():\n    assert True\n"

_WORKFLOW_SOURCE_FILE = {
    "source": _MODULE_SRC,
    "module_name": "m",
    "bound_workflow_id": "bw-solo",
    "symbols": [],
}
_TEST_SOURCE_FILE = {
    "source": _TEST_SRC,
    "module_name": "test_m",
    "test_spec_id": "ts-solo",
    "bound_workflow_id": "bw-solo",
    "symbols": [],
}

_FAILURE_MARKER = "MARKER_TASK_FAILURE_XYZ"


class _CapturingGateway:
    """Wrap a real gateway, recording every ``AgentCallSpec`` that flows through."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list = []

    async def call(self, spec):
        self.calls.append(spec)
        return await self._inner.call(spec)  # type: ignore[attr-defined]


class _ScriptedExecutor:
    """In-memory ``Executor`` returning a scripted sequence of exit codes.

    NOT named ``DryRunExecutor``, so the worker's per-task pytest is really
    dispatched to it (rather than skipped) — but no subprocess ever runs.
    """

    def __init__(self, exit_codes: list[int], *, stdout_text: str = "") -> None:
        self._codes = list(exit_codes)
        self._stdout = stdout_text
        self.specs: list = []

    async def execute(self, spec, *, artifact_store) -> CommandResult:
        self.specs.append(spec)
        code = self._codes.pop(0) if len(self._codes) > 1 else self._codes[0]
        now = datetime.now(UTC)
        out = artifact_store.put_text(
            kind="stdout", text=self._stdout, created_by="scripted", parent_ids=[]
        )
        err = artifact_store.put_text(kind="stderr", text="", created_by="scripted", parent_ids=[])
        return CommandResult(
            exit_code=code,
            started_at=now,
            ended_at=now,
            stdout_artifact=out,
            stderr_artifact=err,
        )


def _seed_solo_bound(store):
    """A one-task frozen BoundWorkflow; returns (bound_ref, bound_obj)."""
    from molexp.harness.schemas import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )

    task = BoundTask(
        id="b-solo",
        ir_task_id="solo",
        capability_id="molpy.x",
        package="molpy",
        callable="x",
        parameters={},
        inputs={},
        outputs={"out": "any"},
    )
    bound = BoundWorkflow(
        id="bw-solo",
        workflow_ir_id="wf-solo",
        tasks=[task],
        edges=[],
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend="local", max_runtime_s=3600, denied_paths=["/", "~/.ssh"]
        ),
    )
    ref = store.put_json(
        kind="bound_workflow",
        obj=bound.model_dump(mode="json"),
        created_by="seed",
        parent_ids=[],
    )
    return ref, bound, task


def _seed_spec(store):
    return store.put_json(
        kind="experiment_spec",
        obj={
            "id": "spec-solo",
            "experiment_report_id": "rep-solo",
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
    """MaterializeExecution (used by the per-task pytest helper) needs a workflow_ir."""
    return store.put_json(
        kind="workflow_ir",
        obj={
            "id": "wf-ir-solo",
            "name": "demo",
            "objective": "realize one task",
            "inputs": {},
            "tasks": [],
            "edges": [],
            "expected_outputs": [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _register_codegen(stub) -> None:
    stub.register(
        agent_name="workflow_source_file_writer",
        output=_WORKFLOW_SOURCE_FILE,
        output_kind="workflow_source_file",
    )
    stub.register(
        agent_name="test_code_file_writer",
        output=_TEST_SOURCE_FILE,
        output_kind="test_source_file",
    )


class TestRealizeOneTaskGreen:
    async def test_green_first_try_returns_module_and_test_src(self, tmp_path: Path) -> None:
        """A passing per-task test (DryRunExecutor skips the real subprocess) →
        the worker returns a green result carrying both generated sources."""
        from molexp.harness.executors.dry_run import DryRunExecutor
        from molexp.harness.gateways.stub import StubAgentGateway

        # Build the store first so the stub can share it, then the ctx.
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        store = FileArtifactStore(root=tmp_path / "artifacts")
        stub = StubAgentGateway(artifact_store=store)
        _register_codegen(stub)
        ctx = _make_ctx_with_store(tmp_path, store, stub)

        bound_ref, bound, task = _seed_solo_bound(ctx.artifact_store)
        spec_ref = _seed_spec(ctx.artifact_store)
        _seed_workflow_ir(ctx.artifact_store)

        result = await realize_one_task(
            ctx,
            task=task,
            bound=bound,
            bound_ref=bound_ref,
            experiment_spec_id=spec_ref.id,
            executor=DryRunExecutor(),
        )

        assert result.status == "green"
        assert result.blocked is None
        assert result.module_src and "def " in result.module_src
        assert result.test_src


class TestRealizeOneTaskRepair:
    async def test_first_red_then_green_uses_mcp_on_repair(self, tmp_path: Path) -> None:
        """A first-round red followed by a green retry returns green, and the
        repair-round codegen call sets ``use_mcp=True`` while the first round
        did not."""
        from molexp.harness.gateways.stub import StubAgentGateway
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        store = FileArtifactStore(root=tmp_path / "artifacts")
        stub = StubAgentGateway(artifact_store=store)
        _register_codegen(stub)
        cap = _CapturingGateway(stub)
        ctx = _make_ctx_with_store(tmp_path, store, cap)

        bound_ref, bound, task = _seed_solo_bound(ctx.artifact_store)
        spec_ref = _seed_spec(ctx.artifact_store)
        _seed_workflow_ir(ctx.artifact_store)

        executor = _ScriptedExecutor([1, 0], stdout_text=_FAILURE_MARKER)
        result = await realize_one_task(
            ctx,
            task=task,
            bound=bound,
            bound_ref=bound_ref,
            experiment_spec_id=spec_ref.id,
            executor=executor,
            attempts=3,
        )

        assert result.status == "green"
        # Some codegen call on the repair turn requested MCP grounding, and at
        # least one first-round call did not (use_mcp starts False).
        assert any(spec.use_mcp for spec in cap.calls)
        assert any(spec.use_mcp is False for spec in cap.calls)


class TestRealizeOneTaskBlocked:
    async def test_ceiling_red_returns_blocked_without_raising(self, tmp_path: Path) -> None:
        """A persistently red per-task test exhausts the attempt budget and the
        worker RETURNS a blocked result (does not raise) carrying the last
        failure text + a failure artifact ref + the task identity."""
        from molexp.harness.gateways.stub import StubAgentGateway
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        store = FileArtifactStore(root=tmp_path / "artifacts")
        stub = StubAgentGateway(artifact_store=store)
        _register_codegen(stub)
        ctx = _make_ctx_with_store(tmp_path, store, stub)

        bound_ref, bound, task = _seed_solo_bound(ctx.artifact_store)
        spec_ref = _seed_spec(ctx.artifact_store)
        _seed_workflow_ir(ctx.artifact_store)

        executor = _ScriptedExecutor([1], stdout_text=_FAILURE_MARKER)
        result = await realize_one_task(
            ctx,
            task=task,
            bound=bound,
            bound_ref=bound_ref,
            experiment_spec_id=spec_ref.id,
            executor=executor,
            attempts=2,
        )

        # Returned, not raised.
        assert result.status == "blocked"
        assert isinstance(result.blocked, BlockedTask)
        blocked = result.blocked
        assert blocked.task_id == task.id
        assert blocked.ir_task_id == task.ir_task_id
        assert blocked.slug == _slug(task.ir_task_id)
        assert blocked.attempts == 2
        assert _FAILURE_MARKER in blocked.error
        assert blocked.error_artifact_id is not None
        assert blocked.question


def _make_ctx_with_store(tmp_path: Path, store, gateway):
    """Build a ``HarnessRunContext`` around an already-constructed artifact store."""
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=store)
    return HarnessRunContext(
        run_id="run-realize-task",
        workspace_root=tmp_path,
        artifact_store=store,
        event_log=e,
        lineage_store=p,
        agent_gateway=gateway,  # type: ignore[arg-type]
    )
