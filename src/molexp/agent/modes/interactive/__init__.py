"""Emergent :class:`InteractiveMode` — the CLI's default agentic loop.

InteractiveMode is the one *emergent* :class:`~molexp.agent.mode.AgentMode`:
the LLM autonomously decides → calls a read-only tool → observes →
loops. Drives :meth:`molexp.agent.router.Router.stream_agentic`; reaches
no structured-output pipeline (those moved to ``molexp.harness`` in spec
03b).
"""

from molexp.agent.modes.interactive.mode import InteractiveMode, InteractiveModeConfig

__all__ = ["InteractiveMode", "InteractiveModeConfig"]
