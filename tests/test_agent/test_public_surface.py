"""Public-surface locks for the ``molexp.agent`` facade.

The agent layer is a pydantic-ai facade. Two ``__all__`` contracts are
frozen here:

* :mod:`molexp.agent` — the loop-orchestration core (five names).
* :mod:`molexp.agent.loops` — the two shipping loops (:class:`ChatLoop` /
  :class:`InteractiveLoop`) plus the SDK-free tool/hook vocabulary the
  emergent loop and harness-injected gates reuse.

The prior pipeline modes (Plan / Author / Run / Review) moved to
:mod:`molexp.harness` and must never reappear in either surface.
"""

from __future__ import annotations

import molexp.agent as agent
import molexp.agent.loops as loops


def test_agent_public_surface_is_the_five_name_core() -> None:
    assert set(agent.__all__) == {
        "AgentRunner",
        "AgentLoop",
        "AgentRunResult",
        "AgentRuntime",
        "AgentSession",
    }


def test_loops_public_surface_is_loops_plus_hook_vocabulary() -> None:
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
