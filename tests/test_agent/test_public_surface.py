"""Public-surface locks for the ``molexp.agent`` facade.

The agent layer is a pydantic-ai facade. Two ``__all__`` contracts are
frozen here:

* :mod:`molexp.agent` — runner + result + runtime + session (four names).
* :mod:`molexp.agent.loops` — the SDK-free tool/hook vocabulary ReAct uses.

There is no ChatLoop / InteractiveLoop / AgentLoop.
"""

from __future__ import annotations

import molexp.agent as agent
import molexp.agent.loops as loops


def test_agent_public_surface_is_the_four_name_core() -> None:
    assert set(agent.__all__) == {
        "AgentRunner",
        "AgentRunResult",
        "AgentRuntime",
        "AgentSession",
    }


def test_loops_public_surface_is_hook_vocabulary() -> None:
    assert set(loops.__all__) == {
        "HookDecision",
        "HookOutcome",
        "LoopState",
        "BeforeToolHook",
        "AfterToolHook",
        "ShouldStopGuard",
        "invoke_before_tool",
        "invoke_after_tool",
        "invoke_should_stop",
        "LoopHooks",
    }
