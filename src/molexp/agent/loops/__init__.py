"""SDK-free tool/hook vocabulary for ``Router.stream_agentic``.

Hooks attach to one ReAct (before/after tool, optional should-stop on
the pydantic-ai inner loop). They are not a molexp conversation loop.
"""

from molexp.agent.loops.hooks import (
    AfterToolHook,
    BeforeToolHook,
    HookDecision,
    HookOutcome,
    LoopHooks,
    LoopState,
    ShouldStopGuard,
    invoke_after_tool,
    invoke_before_tool,
    invoke_should_stop,
)

__all__ = [
    "AfterToolHook",
    "BeforeToolHook",
    "HookDecision",
    "HookOutcome",
    "LoopHooks",
    "LoopState",
    "ShouldStopGuard",
    "invoke_after_tool",
    "invoke_before_tool",
    "invoke_should_stop",
]
