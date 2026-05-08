"""``AgentMode`` ABC contract (spec ac-007)."""

from __future__ import annotations

import pytest

from molexp.agent.mode import AgentMode, AgentRunResult


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
