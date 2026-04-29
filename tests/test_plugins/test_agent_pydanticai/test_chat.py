"""Tests for mid-session chat plumbing on PydanticAISession."""

from __future__ import annotations

import asyncio

import pytest

from molexp.plugins.agent_pydanticai._pydantic_ai.session import PydanticAISession
from molexp.plugins.agent_pydanticai.types import (
    Goal,
    UserMessageEvent,
    UserMessageRequestEvent,
)


def _make_session() -> PydanticAISession:
    return PydanticAISession(
        session_id="sess-test",
        goal=Goal(description="demo"),
        workspace=None,
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_await_user_message_resolves_when_responded():
    session = _make_session()

    async def responder() -> None:
        # Wait until the request appears, then resolve it.
        for _ in range(50):
            events = session.drain_pending_events()
            for ev in events:
                if isinstance(ev, UserMessageRequestEvent):
                    await session.respond_user_message("scope=project", ev.request_id)
                    return
            await asyncio.sleep(0.01)
        raise AssertionError("UserMessageRequestEvent never emitted")

    task = asyncio.create_task(responder())
    reply = await asyncio.wait_for(session.await_user_message("Which scope?"), timeout=2.0)
    await task
    assert reply == "scope=project"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_unsolicited_followup_emits_user_message_event():
    """When no request is pending, respond_user_message still records the event.

    Without an attached agent, no follow-up agent.run is dispatched —
    we just verify the user-message event lands on the queue.
    """
    session = _make_session()
    # No agent attached (None) — _kick_followup will warn and return early.
    await session.respond_user_message("hi there")
    events = session.drain_pending_events()
    user_msgs = [e for e in events if isinstance(e, UserMessageEvent)]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "hi there"
    assert user_msgs[0].request_id is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_drain_pending_events_filters_done_sentinel():
    from molexp.plugins.agent_pydanticai._pydantic_ai.session import _DONE

    session = _make_session()
    # Push a few raw events plus a _DONE; drain should skip the sentinel.
    await session._event_queue.put(UserMessageEvent(content="a"))
    await session._event_queue.put(_DONE)
    await session._event_queue.put(UserMessageEvent(content="b"))

    events = session.drain_pending_events()
    assert [e.content for e in events if isinstance(e, UserMessageEvent)] == ["a", "b"]
