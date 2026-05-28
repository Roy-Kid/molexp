"""``AgentMode`` ABC contract — harness-based (spec 02).

``AgentMode.run`` is now an async generator taking ``harness`` +
``user_input`` and yielding :data:`AgentEvent`\\ s; ``AgentRunResult``
gains an ``events`` field. ``resume()`` is unchanged.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest

from molexp.agent.events import AgentEvent, ModeCompletedEvent
from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.runtime import AgentHarness


def test_agent_mode_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        AgentMode()  # type: ignore[abstract]


def test_agent_mode_subclass_must_implement_run() -> None:
    class Incomplete(AgentMode):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_agent_run_result_is_frozen() -> None:
    result = AgentRunResult(text="hi")
    with pytest.raises(Exception):  # noqa: B017
        result.text = "changed"  # type: ignore[misc]


def test_agent_run_result_events_defaults_empty() -> None:
    assert AgentRunResult(text="hi").events == ()


def test_agent_mode_run_is_harness_based() -> None:
    sig = inspect.signature(AgentMode.run)
    assert "harness" in sig.parameters
    assert "user_input" in sig.parameters
    assert "router" not in sig.parameters
    assert "session" not in sig.parameters


# ── resume() contract ──────────────────────────────────────────────────────


def test_agent_mode_resume_raises_not_implemented_by_default() -> None:
    with pytest.raises(NotImplementedError, match="does not support resume"):
        AgentMode.resume()


def test_agent_mode_resume_default_applies_to_subclass_without_override() -> None:
    class NoResume(AgentMode):
        name = "no-resume"

        async def run(self, *, harness: AgentHarness, user_input: str) -> AsyncIterator[AgentEvent]:
            yield ModeCompletedEvent(text="ok")

    with pytest.raises(NotImplementedError, match="does not support resume"):
        NoResume.resume()


def test_agent_mode_subclass_can_override_resume() -> None:
    class WithResume(AgentMode):
        name = "with-resume"

        def __init__(self, value: str = "") -> None:
            self.value = value

        async def run(self, *, harness: AgentHarness, user_input: str) -> AsyncIterator[AgentEvent]:
            yield ModeCompletedEvent(text=self.value)

        @classmethod
        def resume(cls, **kwargs: Any) -> WithResume:
            return cls(value=kwargs.get("value", "resumed"))

    instance = WithResume.resume(value="hello")
    assert isinstance(instance, WithResume)
    assert instance.value == "hello"
