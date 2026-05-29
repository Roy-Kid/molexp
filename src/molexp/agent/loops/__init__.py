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
"""

from molexp.agent.loops.chat import ChatLoop, ChatLoopConfig
from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig

__all__ = [
    "ChatLoop",
    "ChatLoopConfig",
    "InteractiveLoop",
    "InteractiveLoopConfig",
]
