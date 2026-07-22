"""Public loop surface — :class:`~molexp.agent.loop.AgentLoop` subclasses.

Two loops ship post spec ``harness-as-mode-substrate-03b``:

* :class:`ChatLoop` — single LLM round-trip; the minimal reference
  loop.
* :class:`InteractiveLoop` — the emergent tool-using loop. LLM drives
  a read-only tool loop through pydantic-ai's ``Agent.iter()`` via
  the :class:`~molexp.agent.router.Router` Protocol.

Pipeline-style orchestration (Plan / Author / Run / Review) moved to
:mod:`molexp.harness`; agent.loops is now a pydantic-ai facade for
LLM-only loops.

The package also re-exports the SDK-free neutral tool/hook vocabulary
from :mod:`molexp.agent.loops.hooks` (``HookOutcome`` / ``HookDecision``
/ ``LoopState`` / the ``BeforeToolHook`` / ``AfterToolHook`` /
``ShouldStopGuard`` Protocols / the ``invoke_*`` helpers) that phase-02's
emergent loop and harness-injected gates reuse (spec
``plan-emergent-01-agent-hooks``). Re-exporting stays pure — importing
:mod:`molexp.agent.loops` must never pull ``pydantic_ai``.
"""

from molexp.agent.loops.chat import ChatLoop, ChatLoopConfig
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
from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig

__all__ = [
    "AfterToolHook",
    "BeforeToolHook",
    "ChatLoop",
    "ChatLoopConfig",
    "HookDecision",
    "HookOutcome",
    "InteractiveLoop",
    "InteractiveLoopConfig",
    "LoopHooks",
    "LoopState",
    "ShouldStopGuard",
    "invoke_after_tool",
    "invoke_before_tool",
    "invoke_should_stop",
]
