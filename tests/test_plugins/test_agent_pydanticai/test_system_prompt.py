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


@pytest.mark.unit
def test_plan_mode_addendum_does_not_restrict_tools():
    """Plan mode in molexp doesn't hide tools — it constrains the
    output. The addendum must NOT tell the agent its surface is read-only
    (a previous design did, and it actively prevented the agent from
    drafting valid IRs because list_task_types alone wasn't enough)."""
    lower = PLAN_MODE_ADDENDUM.lower()
    assert "read-only" not in lower
    assert "unavailable" not in lower
    # And it should explicitly advertise that the surface is open.
    assert "does NOT restrict" in PLAN_MODE_ADDENDUM


@pytest.mark.unit
def test_plan_mode_addendum_mandates_exit_plan_mode_call():
    """The addendum must point the agent at exit_plan_mode + the two
    arguments the UI consumes."""
    assert "exit_plan_mode" in PLAN_MODE_ADDENDUM
    assert "plan_markdown" in PLAN_MODE_ADDENDUM
    assert "workflow_preview" in PLAN_MODE_ADDENDUM


@pytest.mark.unit
def test_plan_mode_addendum_specifies_workflow_ir_shape():
    """The IR shape must be enumerated so the agent produces an IR
    that the UI can render and the runtime can execute."""
    assert "task_configs" in PLAN_MODE_ADDENDUM
    assert "task_id" in PLAN_MODE_ADDENDUM
    assert "task_type" in PLAN_MODE_ADDENDUM
    assert "links" in PLAN_MODE_ADDENDUM


@pytest.mark.unit
def test_plan_mode_addendum_says_every_step_is_a_node():
    """The unified design hinges on this invariant — the prompt must
    spell out that prose steps and IR nodes are 1:1 so the agent does
    not produce a ``plan_markdown`` that drifts from the IR."""
    lower = PLAN_MODE_ADDENDUM.lower()
    assert "every step" in lower or "every numbered step" in lower
    assert "node" in lower


@pytest.mark.unit
def test_plan_mode_addendum_treats_investigation_as_nodes():
    """Investigation steps (read literature, grep codebase, inspect
    runs) must be encoded as task nodes in the IR. The addendum must
    name at least one investigation slug as an example so the LLM
    doesn't try to skip the IR for exploratory plans."""
    text = PLAN_MODE_ADDENDUM.lower()
    assert "investigation" in text
    assert (
        "inspect_dataset" in text
        or "inspect_run" in text
        or "list_runs" in text
        or "read_asset" in text
    )


@pytest.mark.unit
def test_plan_mode_addendum_rejects_empty_task_configs():
    """An empty workflow is never valid — the addendum must say so
    explicitly to prevent the LLM from emitting ``task_configs: []``
    when it means "I haven't decided yet"."""
    text = PLAN_MODE_ADDENDUM
    lower = text.lower()
    assert "at least one" in lower
    assert "empty" in lower or "task_configs" in text


@pytest.mark.unit
def test_plan_mode_addendum_warns_against_freeform_final_message():
    """A common LLM failure mode is to dump the plan as a markdown
    final message instead of calling exit_plan_mode. The addendum must
    explicitly call this out so the structured handoff actually fires."""
    assert "free-form" in PLAN_MODE_ADDENDUM.lower() or "freeform" in PLAN_MODE_ADDENDUM.lower()
