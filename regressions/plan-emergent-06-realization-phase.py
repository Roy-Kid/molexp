"""Regression: the realization phase (``RealizeBoard``) end to end, offline.

Binding runtime example for spec ``plan-emergent-06-realization-phase``. A
library-external user drives the harness realization stage
(:class:`molexp.harness.stages.realize_board.RealizeBoard`) over a *frozen*
3-task :class:`BoundWorkflow` using only a :class:`StubAgentGateway` for codegen
and injected executors — NO real LLM, NO real pytest subprocess. It mirrors the
construction in ``tests/test_harness/test_realize_board.py`` exactly.

Two reference scenarios (each printed, then asserted):

1. **all-green** — every task's codegen returns compilable source and a
   :class:`DryRunExecutor` skips the per-task pytest (always green). The board
   maps one self-repairing worker per task (full coverage: 3
   ``workflow_source_file_writer`` + 3 ``test_code_file_writer`` calls),
   reduces the greens into one ``workflow_source`` (a ``build_workflow``
   assembly + one per-task module file) + one ``test_source``, then compiles
   (``--compile-only``) to a ``succeeded`` ``execution_result``. Prints
   ``realized 3/3 tasks; compile=succeeded``.
2. **one blocked** — a slug-scripted canned executor forces the ``beta`` task's
   per-task test red on every attempt (``attempts=2``) while ``alpha`` /
   ``gamma`` stay green, so ``beta`` exhausts its budget. The board persists a
   durable ``intervention_request`` artifact and raises
   :class:`TaskRealizationBlockedError` BEFORE any compile — no
   ``execution_result`` is ever written. Prints
   ``blocked 1/3; intervention_request raised: <question>``.

Deterministic + offline: no network, no subprocess, no server, no CLI, no real
LLM. Run standalone with
``python regressions/plan-emergent-06-realization-phase.py``; the final line on
success is ``plan-emergent-06-realization-phase: ok``.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import TaskRealizationBlockedError
from molexp.harness.executors.dry_run import DryRunExecutor
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.schemas import (
    AgentCallResult,
    AgentCallSpec,
    BoundTask,
    BoundWorkflow,
    CommandResult,
    CommandSpec,
    ExecutionEnvironment,
    ExecutionResult,
    InterventionRequest,
    ResourcePolicy,
    WorkflowSource,
)
from molexp.harness.stages.realize_board import RealizeBoard
from molexp.harness.store.artifact_store import ArtifactStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

# Compilable per-task codegen outputs (renamed downstream by the board).
_MODULE_SRC = "async def make_task(ctx) -> dict:\n    return {}\n"
_TEST_SRC = "def test_it():\n    assert True\n"
_FAILURE_MARKER = "MARKER_TASK_FAILURE_XYZ"


class _CapturingGateway:
    """Wrap a real gateway, recording every ``AgentCallSpec`` that flows through."""

    def __init__(self, inner: StubAgentGateway) -> None:
        self._inner = inner
        self.calls: list[AgentCallSpec] = []

    async def call(self, spec: AgentCallSpec) -> AgentCallResult:
        self.calls.append(spec)
        return await self._inner.call(spec)


class _SlugScriptedExecutor:
    """Canned ``Executor``: red for one target slug's test, green for the rest.

    Keys purely on the pytest ``cmd`` (``tests/test_<slug>.py``), so it is
    robust to which generated source happens to be on disk. Records every spec
    so a caller can assert the ``--compile-only`` command never ran.
    """

    def __init__(self, red_slug: str, *, red_stdout: str = "") -> None:
        self._red_slug = red_slug
        self._red_stdout = red_stdout
        self.specs: list[CommandSpec] = []

    async def execute(self, spec: CommandSpec, *, artifact_store: ArtifactStore) -> CommandResult:
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


def _make_ctx(root: Path, store: FileArtifactStore, gateway: object) -> HarnessRunContext:
    db = root / "events.sqlite"
    event_log = SQLiteEventLog(path=db)
    lineage = SQLiteArtifactLineageStore(path=db, artifact_store=store)
    return HarnessRunContext(
        run_id="run-realize-board",
        workspace_root=root,
        artifact_store=store,
        event_log=event_log,
        lineage_store=lineage,
        agent_gateway=gateway,  # type: ignore[arg-type]
    )


def _seed_three_task_board(store: ArtifactStore) -> None:
    """A 3-task frozen ``BoundWorkflow`` (alpha -> beta -> gamma)."""

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
    store.put_json(
        kind="bound_workflow",
        obj=bound.model_dump(mode="json"),
        created_by="seed",
        parent_ids=[],
    )


def _seed_spec(store: ArtifactStore) -> None:
    store.put_json(
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


def _seed_workflow_ir(store: ArtifactStore) -> None:
    store.put_json(
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


def _register_codegen(stub: StubAgentGateway) -> None:
    """Per-call responders returning generic (renamed-downstream) source."""

    def _wf(spec: AgentCallSpec, store: ArtifactStore) -> object:
        del spec, store
        return stub.make_response(
            {
                "source": _MODULE_SRC,
                "module_name": "m",
                "bound_workflow_id": "bw-3",
                "symbols": [],
            },
            output_kind="workflow_source_file",
        )

    def _test(spec: AgentCallSpec, store: ArtifactStore) -> object:
        del spec, store
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

    stub.register_responder("workflow_source_file_writer", _wf)  # type: ignore[arg-type]
    stub.register_responder("test_code_file_writer", _test)  # type: ignore[arg-type]


def _seed_board(store: ArtifactStore) -> None:
    _seed_three_task_board(store)
    _seed_spec(store)
    _seed_workflow_ir(store)


async def _scenario_all_green(root: Path) -> None:
    """Every task greens -> reduce -> compile succeeds; full 3/3 coverage."""
    store = FileArtifactStore(root=root / "artifacts")
    stub = StubAgentGateway(artifact_store=store)
    _register_codegen(stub)
    cap = _CapturingGateway(stub)
    ctx = _make_ctx(root, store, cap)
    _seed_board(ctx.artifact_store)

    ref = await RealizeBoard(DryRunExecutor()).run(ctx)

    wf_calls = [c for c in cap.calls if c.agent_name == "workflow_source_file_writer"]
    test_calls = [c for c in cap.calls if c.agent_name == "test_code_file_writer"]
    wf = WorkflowSource.model_validate_json(
        ctx.artifact_store.get(ctx.artifact_store.latest_by_kind("workflow_source").id)
    )
    file_paths = {f.path for f in wf.files}
    result = ExecutionResult.model_validate_json(ctx.artifact_store.get(ref.id))

    # Reference observations (printed before asserting).
    print(f"[all-green] workflow_source_file_writer calls = {len(wf_calls)}")
    print(f"[all-green] test_code_file_writer calls       = {len(test_calls)}")
    print(f"[all-green] reduced workflow_source files      = {sorted(file_paths)}")
    print(f"[all-green] build_workflow assembled           = {'def build_workflow' in wf.source}")
    print(f"[all-green] compile execution_result kind      = {ref.kind!r}")
    print(f"[all-green] compile execution_result status    = {result.status!r}")

    assert len(wf_calls) == 3, f"codegen must fire once per board task, got {len(wf_calls)}"
    assert len(test_calls) == 3, (
        f"test codegen must fire once per board task, got {len(test_calls)}"
    )
    assert "def build_workflow" in wf.source, "reduce must assemble a build_workflow function"
    assert file_paths == {
        "workflow/alpha.py",
        "workflow/beta.py",
        "workflow/gamma.py",
    }, f"reduce must emit one per-task module file, got {sorted(file_paths)}"
    assert ctx.artifact_store.latest_by_kind("test_source") is not None, (
        "reduce must emit a test_source"
    )
    assert ref.kind == "execution_result", (
        f"compile must return an execution_result, got {ref.kind!r}"
    )
    assert result.status == "succeeded", f"compile must succeed, got {result.status!r}"

    print("realized 3/3 tasks; compile=succeeded")


async def _scenario_one_blocked(root: Path) -> None:
    """One task stays red -> durable intervention_request, no compile."""
    store = FileArtifactStore(root=root / "artifacts")
    stub = StubAgentGateway(artifact_store=store)
    _register_codegen(stub)
    ctx = _make_ctx(root, store, stub)
    _seed_board(ctx.artifact_store)

    executor = _SlugScriptedExecutor("beta", red_stdout=_FAILURE_MARKER)

    raised: TaskRealizationBlockedError | None = None
    try:
        await RealizeBoard(executor, attempts=2).run(ctx)
    except TaskRealizationBlockedError as exc:
        raised = exc

    ivr_refs = ctx.artifact_store.list_by_kind("intervention_request")
    exec_refs = ctx.artifact_store.list_by_kind("execution_result")

    assert raised is not None, "a persistently-red task must raise TaskRealizationBlockedError"
    assert len(ivr_refs) == 1, (
        f"exactly one durable intervention_request expected, got {len(ivr_refs)}"
    )
    request = InterventionRequest.model_validate_json(ctx.artifact_store.get(ivr_refs[0].id))
    blocked = {bt.slug: bt for bt in request.blocked_tasks}
    total_tasks = 3
    beta = blocked["beta"]

    # Reference observations (printed before asserting).
    print(f"[blocked] TaskRealizationBlockedError raised = {raised is not None}")
    print(f"[blocked] intervention_request count         = {len(ivr_refs)}")
    print(f"[blocked] blocked slugs                      = {sorted(blocked)}")
    print(f"[blocked] beta.task_id                       = {beta.task_id!r}")
    print(f"[blocked] failure marker in beta.error       = {_FAILURE_MARKER in beta.error}")
    print(f"[blocked] execution_result (compile) count   = {len(exec_refs)}")
    print(
        f"[blocked] --compile-only ever ran            = "
        f"{any('--compile-only' in ' '.join(s.cmd) for s in executor.specs)}"
    )

    assert request.bound_workflow_id == "bw-3", "intervention_request must name the bound workflow"
    assert set(blocked) == {"beta"}, f"only beta must block, got {sorted(blocked)}"
    assert beta.task_id == "b-beta", f"blocked task id must be b-beta, got {beta.task_id!r}"
    assert _FAILURE_MARKER in beta.error, (
        "blocked error must carry the captured pytest failure text"
    )
    assert beta.question, "blocked task must carry a human-facing question"
    assert raised.request_ref.kind == "intervention_request", (
        "error must point at the persisted request"
    )
    assert "b-beta" in raised.blocked_task_ids, "error must name the blocked task id"
    assert exec_refs == [], "compile must NEVER run on a block (no execution_result)"
    assert not any("--compile-only" in " ".join(s.cmd) for s in executor.specs), (
        "the --compile-only command must never have executed on a block"
    )

    print(f"blocked {len(blocked)}/{total_tasks}; intervention_request raised: {beta.question}")


async def main() -> int:
    """Drive both realization scenarios; print the success marker."""
    tmp = Path(tempfile.mkdtemp(prefix="plan-emergent-06-realization-phase-"))
    try:
        print("== scenario 1: all-green board realizes + compiles ==")
        await _scenario_all_green(tmp / "green")
        print("== scenario 2: one blocked task raises a durable intervention ==")
        await _scenario_one_blocked(tmp / "block")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("plan-emergent-06-realization-phase: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
