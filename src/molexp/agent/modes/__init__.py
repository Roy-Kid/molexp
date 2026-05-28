"""Public mode surface — :class:`~molexp.agent.mode.AgentMode` subclasses.

Two modes ship post spec ``harness-as-mode-substrate-03b``:

* :class:`ChatMode` — single LLM round-trip; the minimal reference
  mode.
* :class:`InteractiveMode` — the emergent tool-using loop. LLM drives
  a read-only tool loop through pydantic-ai's ``Agent.iter()`` via
  the :class:`~molexp.agent.router.Router` Protocol.

Pipeline-style orchestration (Plan / Author / Run / Review) moved to
:mod:`molexp.harness`; agent.modes is now a pydantic-ai facade for
LLM-only modes.
"""

from molexp.agent.modes.chat import ChatMode, ChatModeConfig
from molexp.agent.modes.interactive import InteractiveMode, InteractiveModeConfig

__all__ = [
    "ChatMode",
    "ChatModeConfig",
    "InteractiveMode",
    "InteractiveModeConfig",
]
