"""Tests for the layered system-prompt composer."""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai._pydantic_ai.system_prompt import (
    BASE_SYSTEM_PROMPT,
    PLAN_MODE_ADDENDUM,
    compose_system_prompt,
)


@pytest.mark.unit
def test_default_returns_base_only():
    out = compose_system_prompt()
    assert out.startswith("You are a research experiment assistant")
    assert PLAN_MODE_ADDENDUM not in out


@pytest.mark.unit
def test_workspace_instructions_appended():
    out = compose_system_prompt(workspace_instructions="Always cite sources.")
    assert "Always cite sources." in out
    assert out.index("Always cite sources.") > out.index("You are a research")


@pytest.mark.unit
def test_skill_instructions_appended_after_workspace():
    out = compose_system_prompt(
        workspace_instructions="WS-RULES",
        skill_instructions="SKILL-RULES",
    )
    assert out.index("WS-RULES") < out.index("SKILL-RULES")


@pytest.mark.unit
def test_session_override_replaces_layers():
    out = compose_system_prompt(
        workspace_instructions="ignored",
        skill_instructions="ignored",
        session_override="just this",
    )
    assert "You are a research" not in out
    assert "ignored" not in out
    assert out.startswith("just this")


@pytest.mark.unit
def test_plan_mode_addendum_is_last():
    out = compose_system_prompt(plan_mode=True)
    assert out.endswith(PLAN_MODE_ADDENDUM.strip())


@pytest.mark.unit
def test_plan_mode_with_override_keeps_override_then_addendum():
    out = compose_system_prompt(session_override="custom", plan_mode=True)
    assert out.startswith("custom")
    assert out.endswith(PLAN_MODE_ADDENDUM.strip())


@pytest.mark.unit
def test_empty_layers_do_not_introduce_blank_lines():
    out = compose_system_prompt(workspace_instructions="   ", skill_instructions="")
    # No double blank-line drift between sections.
    assert "\n\n\n" not in out
    assert out.rstrip() == BASE_SYSTEM_PROMPT.rstrip()
