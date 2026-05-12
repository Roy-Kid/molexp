"""``AgentMode`` ABC contract (spec ac-007, ac-001, ac-002)."""

from __future__ import annotations

from typing import Any

import pytest

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.router import Router
from molexp.agent.session import AgentSession


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
    with pytest.raises(Exception):
        result.text = "changed"  # type: ignore[misc]


# ── ac-001, ac-002: AgentMode.resume() contract ───────────────────────────

def test_agent_mode_resume_raises_not_implemented_by_default() -> None:
    """ac-001: base class resume() raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="does not support resume"):
        AgentMode.resume()


def test_agent_mode_resume_default_applies_to_subclass_without_override() -> None:
    """ac-001: subclass that doesn't override inherits default behaviour."""

    class NoResume(AgentMode):
        name = "no-resume"

        async def run(self, *, router: Router, session: AgentSession, user_input: str) -> AgentRunResult:
            return AgentRunResult(text="ok")

    with pytest.raises(NotImplementedError, match="does not support resume"):
        NoResume.resume()


def test_agent_mode_subclass_can_override_resume() -> None:
    """ac-002: subclass that overrides resume() returns an instance of itself."""

    class WithResume(AgentMode):
        name = "with-resume"

        def __init__(self, value: str = "") -> None:
            self.value = value

        async def run(self, *, router: Router, session: AgentSession, user_input: str) -> AgentRunResult:
            return AgentRunResult(text=self.value)

        @classmethod
        def resume(cls, **kwargs: Any) -> "WithResume":
            return cls(value=kwargs.get("value", "resumed"))

    instance = WithResume.resume(value="hello")
    assert isinstance(instance, WithResume)
    assert instance.value == "hello"
