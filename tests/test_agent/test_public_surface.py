"""Public-surface contract for ``molexp.agent`` (spec ac-001 / ac-002 / ac-003 / ac-006).

The original "four-name contract" covered the mode-orchestration core
(:class:`AgentRunner`, :class:`AgentMode`, :class:`AgentRunResult`,
:class:`AgentSession`). It has been extended with workflow-orthogonal
review primitives — :class:`ReviewPolicy`, :class:`ReviewDecision`,
:class:`ReviewView`, :class:`StepView`, :class:`BypassPolicy`,
:class:`AutoPolicy`, :class:`HumanPolicy`, :func:`cli_ask` — because
the review hook is not mode-specific and any workflow-bearing mode
consumes it.  :class:`HumanPolicy` is UI-agnostic by construction —
the rendering surface is the ``ask`` callable, of which
:func:`cli_ask` is the bundled default.
"""

from __future__ import annotations

import inspect

import molexp.agent as agent
import molexp.agent.modes as modes


def test_agent_all_is_the_public_contract() -> None:
    assert set(agent.__all__) == {
        # Mode orchestration core.
        "AgentRunner",
        "AgentMode",
        "AgentRunResult",
        "AgentSession",
        # Workflow-orthogonal review primitives (parallel to mode).
        "ReviewPolicy",
        "ReviewDecision",
        "ReviewView",
        "StepView",
        "BypassPolicy",
        "AutoPolicy",
        "HumanPolicy",
        "cli_ask",
    }


def test_modes_all_is_the_chat_plan_and_author_contract() -> None:
    """``ChatMode`` / ``PlanMode`` / ``AuthorMode`` ship today — Run /
    Review are rebuilt on the harness by later specs 05-06."""
    assert set(modes.__all__) == {
        "ApprovedPlanHandoff",
        "AuthorMode",
        "AuthorModeConfig",
        "ChatMode",
        "ChatModeConfig",
        "MaterializedWorkspaceHandoff",
        "PlanFolder",
        "PlanMode",
        "PlanModeConfig",
    }


def test_no_pydantic_ai_or_pydantic_graph_in_public_surface() -> None:
    bad = [
        n
        for n in (*agent.__all__, *modes.__all__)
        if n.startswith(("pydantic_ai", "pydantic_graph"))
    ]
    assert not bad, bad


def test_no_factory_callables_in_public_surface() -> None:
    """Anti-factory invariant — ac-006."""
    for ns in (agent, modes):
        for name, obj in inspect.getmembers(ns):
            if name in ns.__all__ and callable(obj):
                assert not name.startswith(("create_", "build_", "make_")), name
