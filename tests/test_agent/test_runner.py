"""Phase 1b/1c: AgentRunner end-to-end against FakeModelClient."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from molexp.agent import (
    AgentService,
    Goal,
    SessionStatus,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from molexp.agent.orchestration import (
    ModelResponded,
    PlanCreated,
    PlanDecided,
    SessionStarted,
    ToolCallCompleted,
    ToolCallRequested,
)
from molexp.agent.testing import FakeModelClient
from molexp.agent.types import AgentMode


def _spec(name: str, **kwargs) -> ToolSpec:
    base = {
        "name": name,
        "description": "",
        "input_schema": {"type": "object", "properties": {}},
    }
    base.update(kwargs)
    return ToolSpec(**base)


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.mark.asyncio
async def test_text_only_turn_persists_messages_and_events(workspace_path: Path) -> None:
    model = FakeModelClient()
    model.queue_text("hello back")
    service = AgentService.from_workspace(workspace_path, model=model)
    session = service.start_session(Goal(description="say hi"))

    received = []

    async def collect() -> None:
        async for ev in session.stream_events():
            received.append(ev)
            if isinstance(ev, ModelResponded):
                break

    await asyncio.wait_for(collect(), timeout=2.0)
    # Drain the inbox so the runner doesn't sit blocked forever.
    await service.cancel(session.session_id)
    await asyncio.sleep(0)

    assert any(isinstance(e, SessionStarted) for e in received)
    assert any(isinstance(e, ModelResponded) for e in received)

    sessions_root = workspace_path / ".molexp-agent" / "sessions" / session.session_id
    messages_path = sessions_root / "messages.jsonl"
    events_path = sessions_root / "events.jsonl"
    assert messages_path.exists()
    assert events_path.exists()
    msgs = [json.loads(line) for line in messages_path.read_text().splitlines() if line.strip()]
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]
    assert msgs[0]["content"] == "say hi"
    assert msgs[1]["content"] == "hello back"


@pytest.mark.asyncio
async def test_tool_call_loop_dispatches_native_tool(workspace_path: Path) -> None:
    model = FakeModelClient()
    model.queue_tool_call("native:echo", {"x": 7}, call_id="c1")
    model.queue_text("done")

    service = AgentService.from_workspace(workspace_path, model=model)

    captured = {}

    async def echo(args: dict, ctx: ToolContext) -> ToolResult:
        captured["args"] = args
        captured["session"] = ctx.session_id
        return ToolResult(ok=True, value={"echoed": args})

    service.registry.register(_spec("native:echo"), echo)

    session = service.start_session(Goal(description="run echo"))

    seen_call = asyncio.Event()
    seen_result = asyncio.Event()

    async def collect() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, ToolCallRequested):
                seen_call.set()
            if isinstance(ev, ToolCallCompleted):
                seen_result.set()
                break

    await asyncio.wait_for(collect(), timeout=2.0)
    await service.cancel(session.session_id)

    assert seen_call.is_set() and seen_result.is_set()
    assert captured["args"] == {"x": 7}

    msgs_path = (
        workspace_path / ".molexp-agent" / "sessions" / session.session_id / "messages.jsonl"
    )
    msgs = [json.loads(line) for line in msgs_path.read_text().splitlines() if line.strip()]
    roles = [m["role"] for m in msgs]
    # user goal, assistant tool-call (no text → no assistant), tool result, assistant final text
    assert "tool" in roles
    assert msgs[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_plan_mode_emits_and_resolves(workspace_path: Path) -> None:
    model = FakeModelClient()
    plan_text = (
        'Step 1: do the thing\n```json\n{"workflow_ir": {"task_configs": [{"task_id": "t1"}]}}\n```'
    )
    model.queue_text(plan_text)
    model.queue_text("Plan complete; proceeding.")

    service = AgentService.from_workspace(workspace_path, model=model)
    goal = Goal(description="plan something", mode=AgentMode.PLAN)
    session = service.start_session(goal)

    plan_created: list[PlanCreated] = []

    async def watch_plan() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanCreated):
                plan_created.append(ev)
                break

    await asyncio.wait_for(watch_plan(), timeout=2.0)
    assert plan_created, "PlanCreated should have fired"
    request_id = plan_created[0].request_id

    plan_decided: list[PlanDecided] = []

    async def watch_decided() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanDecided):
                plan_decided.append(ev)
                break

    decided_task = asyncio.create_task(watch_decided())
    await asyncio.sleep(0)
    ok = await session.respond_plan(request_id=request_id, approved=True)
    assert ok is True
    await asyncio.wait_for(decided_task, timeout=2.0)
    assert plan_decided[0].approved is True

    # Wait for the runner to finish the post-approval turn.
    await asyncio.sleep(0.05)
    await service.cancel(session.session_id)

    ckpt_dir = workspace_path / ".molexp-agent" / "sessions" / session.session_id / "checkpoints"
    assert ckpt_dir.exists()
    files = list(ckpt_dir.glob("*.json"))
    assert files, "plan-decision checkpoint should have been written"


@pytest.mark.asyncio
async def test_plan_rejection_loops_back_with_synthetic_user_message(
    workspace_path: Path,
) -> None:
    model = FakeModelClient()
    model.queue_text("draft plan v1")
    model.queue_text("revised plan v2")
    model.queue_text("ok done")

    service = AgentService.from_workspace(workspace_path, model=model)
    goal = Goal(description="plan it", mode=AgentMode.PLAN)
    session = service.start_session(goal)

    first_created: list[PlanCreated] = []

    async def wait_first() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanCreated):
                first_created.append(ev)
                break

    await asyncio.wait_for(wait_first(), timeout=2.0)
    assert await session.respond_plan(
        request_id=first_created[0].request_id,
        approved=False,
        feedback="be terser",
    )

    # Runner should emit PlanDecided then loop back; second PlanCreated arrives.
    second_created: list[PlanCreated] = []

    async def wait_second() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanCreated):
                second_created.append(ev)
                break

    await asyncio.wait_for(wait_second(), timeout=2.0)
    assert second_created, "model should be re-prompted with reject feedback"
    await service.cancel(session.session_id)

    msgs_path = (
        workspace_path / ".molexp-agent" / "sessions" / session.session_id / "messages.jsonl"
    )
    msgs = [json.loads(line) for line in msgs_path.read_text().splitlines() if line.strip()]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert any("Plan rejected." in m["content"] for m in user_msgs)
    assert any("be terser" in m["content"] for m in user_msgs)


@pytest.mark.asyncio
async def test_no_model_skips_runner_keeps_phase_1a_surface(workspace_path: Path) -> None:
    """Phase 1a fixtures pass no model — start_session must stay synchronous."""

    service = AgentService.from_workspace(workspace_path)
    session = service.start_session(Goal(description="hi"))
    assert session.status is SessionStatus.PENDING
    assert session.task is None


@pytest.mark.asyncio
async def test_orphaned_running_session_marked_interrupted_on_restart(
    workspace_path: Path,
) -> None:
    """Decision O3 phase-1: previous-process sessions surface as ``interrupted``."""

    service = AgentService.from_workspace(workspace_path)
    session = service.start_session(Goal(description="never finishes"))
    # Pretend the previous process died mid-run.
    session.status = SessionStatus.RUNNING
    service.state.sessions.write_metadata(
        type(service.state.sessions.read_metadata(session.session_id))(
            session_id=session.session_id,
            goal=session.goal,
            status=SessionStatus.RUNNING,
        )
    )

    fresh = AgentService.from_workspace(workspace_path)
    meta = fresh.state.sessions.read_metadata(session.session_id)
    assert meta is not None
    assert meta.status is SessionStatus.INTERRUPTED


@pytest.mark.asyncio
async def test_recovery_policy_retries_transient_model_errors(
    workspace_path: Path,
) -> None:
    """``SimpleRetryPolicy`` retries one ``MODEL_ERROR`` then succeeds."""

    class _FlakyModel:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            from molexp.agent.model import ModelResponse

            return ModelResponse(text="recovered", finish_reason="stop")

        def stream(self, request):  # noqa: ANN001
            raise NotImplementedError

    model = _FlakyModel()
    service = AgentService.from_workspace(workspace_path, model=model)
    session = service.start_session(Goal(description="go"))

    async def wait_for_text() -> ModelResponded:
        async for ev in session.stream_events():
            if isinstance(ev, ModelResponded):
                return ev
        raise AssertionError("no ModelResponded event")

    ev = await asyncio.wait_for(wait_for_text(), timeout=2.0)
    await service.cancel(session.session_id)
    assert ev.finish_reason == "stop"
    assert model.calls == 2


@pytest.mark.asyncio
async def test_evaluator_records_eval_checkpoint_at_terminal(
    workspace_path: Path,
) -> None:
    """``Evaluator`` runs at session terminal-state and writes a checkpoint.

    Drives ``_finalize_session`` directly so the test is independent of
    cancellation semantics — that path is exercised at every natural
    drive-loop exit.
    """

    from molexp.agent.orchestration.session import AgentSession

    model = FakeModelClient()
    service = AgentService.from_workspace(workspace_path, model=model)
    session = AgentSession(session_id="sess-eval", goal=Goal(description="hi"))
    runner = service._build_runner(session)

    await runner._finalize_session(session, turn_count=3)

    eval_path = (
        workspace_path / ".molexp-agent" / "sessions" / "sess-eval" / "checkpoints" / "eval.json"
    )
    assert eval_path.exists()
    record = json.loads(eval_path.read_text())
    assert record["evaluator"] == "noop"
    assert "ts" in record


@pytest.mark.asyncio
async def test_resume_session_replays_persisted_history(workspace_path: Path) -> None:
    """A ``RESUMABLE`` session re-spawns its runner from messages.jsonl."""

    from molexp.agent.model import ModelRequest, ModelResponse

    class _RecordingModel:
        name = "rec"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            return ModelResponse(text="resumed reply", finish_reason="stop")

        def stream(self, request):  # noqa: ANN001
            raise NotImplementedError

    # First process: queue a goal + a follow-up turn so messages.jsonl
    # records something replayable.
    fake = FakeModelClient()
    fake.queue_text("first reply")
    service = AgentService.from_workspace(workspace_path, model=fake)
    sess = service.start_session(Goal(description="resume me"))

    async def wait_responded() -> None:
        async for ev in sess.stream_events():
            if isinstance(ev, ModelResponded):
                return

    await asyncio.wait_for(wait_responded(), timeout=2.0)
    await service.cancel(sess.session_id)

    # Second process: a fresh service with a recording model. Startup
    # should mark the session RESUMABLE; resume_session replays the
    # persisted user goal + assistant reply into the new runner.
    rec = _RecordingModel()
    fresh = AgentService.from_workspace(workspace_path, model=rec)
    meta = fresh.state.sessions.read_metadata(sess.session_id)
    assert meta is not None
    assert meta.status is SessionStatus.RESUMABLE

    resumed = fresh.resume_session(sess.session_id)

    async def wait_resumed() -> None:
        async for ev in resumed.stream_events():
            if isinstance(ev, ModelResponded):
                return

    await asyncio.wait_for(wait_resumed(), timeout=2.0)
    await fresh.cancel(resumed.session_id)

    assert rec.requests, "recording model should have seen a request"
    # The very first request the resumed runner makes carries the
    # persisted history from the prior process.
    roles = [m.role for m in rec.requests[0].messages]
    assert roles == ["user", "assistant"]
    assert rec.requests[0].messages[0].content == "resume me"
    assert rec.requests[0].messages[1].content == "first reply"


@pytest.mark.asyncio
async def test_token_budget_caps_session_with_context_overflow(
    workspace_path: Path,
) -> None:
    """``max_total_input_tokens`` budget fires a CONTEXT_OVERFLOW failure."""

    from molexp.agent.model import ModelRequest, ModelResponse
    from molexp.agent.orchestration import FailureRecorded
    from molexp.agent.orchestration.runner import AgentRunner
    from molexp.agent.orchestration.session import AgentSession
    from molexp.agent.recovery.constraints import ConstraintSet
    from molexp.agent.tools.dispatcher import ToolDispatcher
    from molexp.agent.tools.registry import ToolRegistry
    from molexp.agent.types import Usage

    class _GreedyModel:
        name = "greedy"

        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                text="ok",
                usage=Usage(input_tokens=200, output_tokens=10),
                finish_reason="stop",
            )

        def stream(self, request):  # noqa: ANN001
            raise NotImplementedError

    service = AgentService.from_workspace(workspace_path)
    registry = ToolRegistry()
    runner = AgentRunner(
        model=_GreedyModel(),
        registry=registry,
        store=service.state.sessions,
        dispatcher=ToolDispatcher(registry),
        constraints=ConstraintSet(max_total_input_tokens=150),
    )
    # Pre-load usage so the very first turn trips the cap.
    runner.usage.input_tokens = 200

    session = AgentSession(session_id="sess-budget", goal=Goal(description="burn"))
    # Subscribe *before* driving so we capture every published event.
    events = session.stream_events()
    await runner.drive_session(session)
    await session.bus.close()

    failures: list[FailureRecorded] = []
    async for ev in events:
        if isinstance(ev, FailureRecorded):
            failures.append(ev)

    assert failures
    assert failures[0].failure.kind.value == "context_overflow"


def test_legacy_sessions_migration_writes_tombstones(workspace_path: Path) -> None:
    """``migrate_legacy_sessions`` lays down ``status=legacy`` tombstones."""

    from molexp.agent.sessions import SessionStore
    from molexp.agent.sessions.migrate import migrate_legacy_sessions

    legacy_dir = workspace_path / "sessions" / "old-1"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "metadata.json").write_text(
        json.dumps(
            {
                "session_id": "old-1",
                "status": "running",
                "goal": {
                    "description": "do legacy work",
                    "constraints": {},
                    "success_criteria": [],
                },
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    store = SessionStore(workspace_path / ".molexp-agent" / "sessions")
    result = migrate_legacy_sessions(workspace_path, store)
    assert result.migrated == ("old-1",)
    assert result.skipped == ()

    meta = store.read_metadata("old-1")
    assert meta is not None
    assert meta.status is SessionStatus.LEGACY
    assert meta.goal.description == "do legacy work"
    assert "legacy session" in meta.summary

    # Re-running is idempotent: existing tombstones are skipped, not duplicated.
    second = migrate_legacy_sessions(workspace_path, store)
    assert second.migrated == ()
    assert second.skipped == ("old-1",)
