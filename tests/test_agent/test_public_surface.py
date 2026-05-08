"""Public-surface contract for ``molexp.agent`` (spec ac-001 / ac-002 / ac-003 / ac-006)."""

from __future__ import annotations

import inspect

import molexp.agent as agent
import molexp.agent.modes as modes


def test_agent_all_is_the_four_name_contract() -> None:
    assert set(agent.__all__) == {
        "AgentRunner",
        "AgentMode",
        "AgentRunResult",
        "AgentSession",
    }


def test_modes_all_is_the_three_pair_contract() -> None:
    assert set(modes.__all__) == {
        "PlanMode",
        "PlanModeConfig",
        "ChatMode",
        "ChatModeConfig",
        "ReviewMode",
        "ReviewModeConfig",
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
