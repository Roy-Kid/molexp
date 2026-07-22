"""Public-surface contract for ``molexp.agent`` (post spec 03b).

The agent layer is a pydantic-ai facade. Surface contract:

* Loop-orchestration core: :class:`AgentRunner`, :class:`AgentLoop`,
  :class:`AgentRunResult`, :class:`AgentRuntime`, :class:`AgentSession`.

Loops contract — only :class:`ChatLoop` (one round-trip) and the
emergent :class:`InteractiveLoop` ship; the prior pipeline modes
(Plan / Author / Run / Review) moved to :mod:`molexp.harness`.
"""

from __future__ import annotations

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


def test_loops_all_is_loops_plus_hook_vocabulary() -> None:
    """``molexp.agent.loops`` re-exports the two shipping loops (ChatLoop +
    InteractiveLoop, +configs) plus the SDK-free neutral tool/hook vocabulary
    (``HookOutcome`` / ``HookDecision`` / ``LoopState`` / the three hook
    Protocols / the ``invoke_*`` helpers) that phase-02's emergent loop and
    harness-injected gates reuse (spec ``plan-emergent-01-agent-hooks``), plus
    the ``LoopHooks`` bundle the outer emergent loop consumes (spec
    ``plan-emergent-02-loop-refactor``). The prior pipeline modes moved to
    :mod:`molexp.harness`."""
    assert set(loops.__all__) == {
        # Shipping loops.
        "ChatLoop",
        "ChatLoopConfig",
        "InteractiveLoop",
        "InteractiveLoopConfig",
        # Neutral tool/hook vocabulary (plan-emergent-01).
        "HookDecision",
        "HookOutcome",
        "LoopState",
        "BeforeToolHook",
        "AfterToolHook",
        "ShouldStopGuard",
        "invoke_before_tool",
        "invoke_after_tool",
        "invoke_should_stop",
        # Hook bundle consumed by the outer emergent loop (plan-emergent-02).
        "LoopHooks",
    }
