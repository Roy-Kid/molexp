"""Emergent :class:`InteractiveLoop` — the CLI's default agentic loop.

InteractiveLoop is the one *emergent* :class:`~molexp.agent.loop.AgentLoop`:
the LLM autonomously decides → calls tools (read, write, execute_python,
knowledge) → observes → loops. Drives
:meth:`molexp.agent.router.Router.stream_agentic`; reaches no structured-
output pipeline (those live in ``molexp.harness``).
"""

from molexp.agent.loops.interactive.loop import InteractiveLoop, InteractiveLoopConfig

__all__ = ["InteractiveLoop", "InteractiveLoopConfig"]
