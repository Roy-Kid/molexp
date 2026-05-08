"""``AgentSession`` unit tests."""

from __future__ import annotations

from molexp.agent.session import AgentSession
from molexp.agent.types import Message


def test_session_assigns_random_id_when_unspecified() -> None:
    s1 = AgentSession()
    s2 = AgentSession()
    assert s1.session_id != s2.session_id
    assert len(s1.session_id) >= 8


def test_session_preserves_explicit_id() -> None:
    s = AgentSession(session_id="alpha")
    assert s.session_id == "alpha"


def test_append_grows_history() -> None:
    s = AgentSession()
    assert s.history == []
    s.append(Message(role="user", content="hi"))
    s.append(Message(role="assistant", content="hello"))
    assert [m.role for m in s.history] == ["user", "assistant"]


def test_mode_state_defaults_empty_and_is_mutable() -> None:
    s = AgentSession()
    assert s.mode_state == {}
    s.mode_state["plan_step"] = 3
    assert s.mode_state["plan_step"] == 3
