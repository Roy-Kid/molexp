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


def test_model_messages_defaults_empty_tuple() -> None:
    """Fresh sessions start with no pydantic-ai history."""
    s = AgentSession()
    assert s.model_messages == ()


def test_model_messages_accepts_seed_at_construction() -> None:
    """Restored sessions can carry prior ``ModelMessage`` history.

    The history is opaque ``Any`` at this layer — the constructor just
    has to accept and store it as a tuple (so modes can hot-swap it).
    """
    seed: tuple[object, ...] = ("placeholder-msg",)
    s = AgentSession(model_messages=seed)
    assert s.model_messages == seed


def test_model_messages_is_mutable_attribute() -> None:
    """Modes overwrite ``model_messages`` wholesale after each turn."""
    s = AgentSession()
    s.model_messages = ("m1", "m2")
    assert s.model_messages == ("m1", "m2")
