"""Public-surface contract for ``molexp.agent`` (post spec 03b).

The agent layer is a pydantic-ai facade. Surface contract:

* Mode-orchestration core: :class:`AgentRunner`, :class:`AgentMode`,
  :class:`AgentRunResult`, :class:`AgentRuntime`, :class:`AgentSession`.
* Workflow-orthogonal approval primitives: :class:`ReviewDecision`,
  the :data:`ReviewPolicy` callable alias, and the bundled
  :func:`cli_ask` policy.

Modes contract — only :class:`ChatMode` (one round-trip) and the
emergent :class:`InteractiveMode` ship; the prior pipeline modes
(Plan / Author / Run / Review) moved to :mod:`molexp.harness`.
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
        "AgentRuntime",
        "AgentSession",
        # Workflow-orthogonal approval primitives (parallel to mode).
        "ReviewPolicy",
        "ReviewDecision",
        "cli_ask",
    }


def test_modes_all_is_chat_and_interactive_only() -> None:
    """Post spec 03b, ``molexp.agent.modes`` re-exports ChatMode +
    InteractiveMode (+configs). The prior pipeline modes moved to
    :mod:`molexp.harness`."""
    assert set(modes.__all__) == {
        "ChatMode",
        "ChatModeConfig",
        "InteractiveMode",
        "InteractiveModeConfig",
    }


def test_no_pydantic_ai_or_pydantic_graph_in_public_surface() -> None:
    bad = [
        n
        for n in (*agent.__all__, *modes.__all__)
        if n.startswith(("pydantic_ai", "pydantic_graph"))
    ]
    assert not bad, bad


def test_no_factory_callables_in_public_surface() -> None:
    """Anti-factory invariant."""
    for ns in (agent, modes):
        for name, obj in inspect.getmembers(ns):
            if name in ns.__all__ and callable(obj):
                assert not name.startswith(("create_", "build_", "make_")), name
