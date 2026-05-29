"""Emergent :class:`InteractiveLoop` — the CLI's default agentic loop.

InteractiveLoop is the one *emergent* :class:`~molexp.agent.loop.AgentLoop`:
the LLM autonomously decides → calls a read-only tool → observes →
loops. Drives :meth:`molexp.agent.router.Router.stream_agentic`; reaches
no structured-output pipeline (those moved to ``molexp.harness`` in spec
03b).
"""

from molexp.agent.loops.interactive.loop import InteractiveLoop, InteractiveLoopConfig

__all__ = ["InteractiveLoop", "InteractiveLoopConfig"]
