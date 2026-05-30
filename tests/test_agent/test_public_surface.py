"""Public-surface contract for ``molexp.agent`` (post spec 03b).

The agent layer is a pydantic-ai facade. Surface contract:

* Loop-orchestration core: :class:`AgentRunner`, :class:`AgentLoop`,
  :class:`AgentRunResult`, :class:`AgentRuntime`, :class:`AgentSession`.

Loops contract — only :class:`ChatLoop` (one round-trip) and the
emergent :class:`InteractiveLoop` ship; the prior pipeline modes
(Plan / Author / Run / Review) moved to :mod:`molexp.harness`.
"""

from __future__ import annotations

import inspect

import molexp.agent as agent
import molexp.agent.loops as loops


def test_agent_all_is_the_public_contract() -> None:
    assert set(agent.__all__) == {
        # Loop orchestration core.
        "AgentRunner",
        "AgentLoop",
        "AgentRunResult",
        "AgentRuntime",
        "AgentSession",
    }


def test_loops_all_is_chat_and_interactive_only() -> None:
    """Post spec 03b, ``molexp.agent.loops`` re-exports ChatLoop +
    InteractiveLoop (+configs). The prior pipeline modes moved to
    :mod:`molexp.harness`."""
    assert set(loops.__all__) == {
        "ChatLoop",
        "ChatLoopConfig",
        "InteractiveLoop",
        "InteractiveLoopConfig",
    }


def test_no_pydantic_ai_or_pydantic_graph_in_public_surface() -> None:
    bad = [
        n
        for n in (*agent.__all__, *loops.__all__)
        if n.startswith(("pydantic_ai", "pydantic_graph"))
    ]
    assert not bad, bad


def test_no_factory_callables_in_public_surface() -> None:
    """Anti-factory invariant."""
    for ns in (agent, loops):
        for name, obj in inspect.getmembers(ns):
            if name in ns.__all__ and callable(obj):
                assert not name.startswith(("create_", "build_", "make_")), name
