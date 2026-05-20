"""``AgentHarness`` behaviour tests (spec ac-009, ac-010)."""

from __future__ import annotations

import pytest

from molexp.agent.harness.events import (
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    CompactionPerformedEvent,
    ErrorEvent,
    StageCompletedEvent,
    StageStartedEvent,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.hooks import HookContext, HookPoint
from molexp.agent.harness.session import Session
from molexp.agent.harness.session_entry import CompactionEntry
from molexp.agent.modes._planning import ApprovalGate
from molexp.agent.review import ReviewDecision
from molexp.agent.types import Message


def _harness(session: Session, **kwargs: object) -> AgentHarness:
    events: list[object] = []

    async def sink(event: object) -> None:
        events.append(event)

    harness = AgentHarness(session=session, event_sink=sink, **kwargs)  # type: ignore[arg-type]
    harness._test_events = events  # type: ignore[attr-defined]
    return harness


# ── stage() context manager (ac-010) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_stage_emits_started_and_completed(session: Session) -> None:
    harness = _harness(session)
    async with harness.stage("draft"):
        pass
    events = harness._test_events  # type: ignore[attr-defined]
    assert isinstance(events[0], StageStartedEvent)
    assert events[0].stage_name == "draft"
    assert isinstance(events[1], StageCompletedEvent)
    assert events[1].stage_name == "draft"


@pytest.mark.asyncio
async def test_stage_emits_error_event_on_exception(session: Session) -> None:
    harness = _harness(session)
    with pytest.raises(ValueError, match="kaboom"):
        async with harness.stage("risky"):
            raise ValueError("kaboom")
    events = harness._test_events  # type: ignore[attr-defined]
    assert isinstance(events[0], StageStartedEvent)
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].stage_name == "risky"
    assert "kaboom" in events[-1].message


@pytest.mark.asyncio
async def test_stage_records_a_stage_entry(session: Session) -> None:
    harness = _harness(session)
    async with harness.stage("phase-1"):
        pass
    entries = session.path_to_root()
    stage_names = [e.stage_name for e in entries if hasattr(e, "stage_name")]
    assert "phase-1" in stage_names


@pytest.mark.asyncio
async def test_stage_fires_before_and_after_hooks(session: Session) -> None:
    harness = _harness(session)
    fired: list[str] = []

    async def before(ctx: HookContext) -> None:
        fired.append(f"before:{ctx.stage_name}")

    async def after(ctx: HookContext) -> None:
        fired.append(f"after:{ctx.stage_name}")

    harness.hooks.register(HookPoint.before_stage, before)
    harness.hooks.register(HookPoint.after_stage, after)
    async with harness.stage("s"):
        fired.append("body")
    assert fired == ["before:s", "body", "after:s"]


# ── approve() — unified ApprovalGate (ac-009) ──────────────────────────────


class _View:
    """Minimal review view — just a summary string."""

    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.step_id = "step"
        self.artifact_paths: tuple[object, ...] = ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate",
    [
        ApprovalGate.approve_direction,
        ApprovalGate.approve_materialization,
        ApprovalGate.approve_execution,
    ],
)
async def test_approve_default_approves_and_emits_events(
    session: Session, gate: ApprovalGate
) -> None:
    harness = _harness(session)
    decision = await harness.approve(gate, _View("looks good"))
    assert decision.approved is True
    events = harness._test_events  # type: ignore[attr-defined]
    assert isinstance(events[0], ApprovalRequestedEvent)
    assert events[0].gate == gate.value
    assert isinstance(events[1], ApprovalDecidedEvent)
    assert events[1].approved is True


@pytest.mark.asyncio
async def test_approve_denying_hook_yields_rejection(session: Session) -> None:
    harness = _harness(session)

    async def deny(ctx: HookContext) -> ReviewDecision:
        return ReviewDecision(approved=False, reason="not yet")

    harness.hooks.register(HookPoint.before_approval, deny)
    decision = await harness.approve(ApprovalGate.approve_execution, _View("x"))
    assert decision.approved is False
    assert decision.reason == "not yet"
    events = harness._test_events  # type: ignore[attr-defined]
    decided = [e for e in events if isinstance(e, ApprovalDecidedEvent)]
    assert decided[0].approved is False


@pytest.mark.asyncio
async def test_approve_records_an_approval_entry(session: Session) -> None:
    harness = _harness(session)
    await harness.approve(ApprovalGate.approve_direction, _View("ok"))
    entries = session.path_to_root()
    gates = [e.gate for e in entries if hasattr(e, "gate")]
    assert ApprovalGate.approve_direction.value in gates


# ── run_subprocess delegation ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_subprocess_delegates_to_execution_env(
    session: Session, fake_execution_env: object
) -> None:
    harness = _harness(session, execution_env=fake_execution_env)
    result = await harness.run_subprocess(["echo", "hi"])
    assert result.exit_code == 0
    assert fake_execution_env.calls[0]["command"] == ["echo", "hi"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_run_subprocess_without_env_raises(session: Session) -> None:
    harness = _harness(session)
    with pytest.raises(RuntimeError, match="execution_env"):
        await harness.run_subprocess(["echo", "x"])


# ── compact() (ac-005 boundary) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_summarizes_via_router_and_appends_entry(
    session: Session, scripted_router: object
) -> None:
    # Build a long conversation that exceeds the keep-recent window.
    for _ in range(8):
        session.append_message(Message(role="user", content="x" * 400))
    from molexp.agent.harness.compaction import CompactionSettings

    harness = _harness(
        session,
        router=scripted_router,
        compaction_settings=CompactionSettings(keep_recent_tokens=250),
    )
    performed = await harness.compact()
    assert performed is True
    # A CompactionEntry now sits on the active branch.
    entries = session.path_to_root()
    compactions = [e for e in entries if isinstance(e, CompactionEntry)]
    assert len(compactions) == 1
    assert compactions[0].summary == "canned summary"
    events = harness._test_events  # type: ignore[attr-defined]
    assert any(isinstance(e, CompactionPerformedEvent) for e in events)


@pytest.mark.asyncio
async def test_compact_noop_on_short_session(session: Session, scripted_router: object) -> None:
    session.append_message(Message(role="user", content="hi"))
    harness = _harness(session, router=scripted_router)
    assert await harness.compact() is False


@pytest.mark.asyncio
async def test_compact_before_compact_hook_can_veto(
    session: Session, scripted_router: object
) -> None:
    for _ in range(8):
        session.append_message(Message(role="user", content="x" * 400))
    from molexp.agent.harness.compaction import CompactionSettings

    harness = _harness(
        session,
        router=scripted_router,
        compaction_settings=CompactionSettings(keep_recent_tokens=250),
    )

    async def veto(ctx: HookContext) -> dict[str, bool]:
        return {"veto": True}

    harness.hooks.register(HookPoint.before_compact, veto)
    assert await harness.compact() is False
    entries = session.path_to_root()
    assert not [e for e in entries if isinstance(e, CompactionEntry)]


@pytest.mark.asyncio
async def test_compact_without_router_raises(session: Session) -> None:
    for _ in range(8):
        session.append_message(Message(role="user", content="x" * 400))
    from molexp.agent.harness.compaction import CompactionSettings

    harness = _harness(session, compaction_settings=CompactionSettings(keep_recent_tokens=250))
    with pytest.raises(RuntimeError, match="router"):
        await harness.compact()


# ── emit() ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_emit_forwards_to_sink(session: Session) -> None:
    harness = _harness(session)
    await harness.emit(StageStartedEvent(stage_name="manual"))
    events = harness._test_events  # type: ignore[attr-defined]
    assert isinstance(events[0], StageStartedEvent)


def test_harness_exposes_session_and_hooks(session: Session) -> None:
    harness = _harness(session)
    assert harness.session is session
    assert harness.hooks is not None
