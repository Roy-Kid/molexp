"""Emergent :class:`InteractiveMode` — the CLI's default agentic loop.

InteractiveMode is the one *emergent* :class:`~molexp.agent.mode.AgentMode`:
the LLM autonomously decides → calls a read-only tool → observes →
loops, and may delegate to the structured PlanMode pipeline. It is a
sibling of the declarative modes, composing — never inheriting — them.
"""

from molexp.agent.modes.interactive.mode import InteractiveMode, InteractiveModeConfig

__all__ = ["InteractiveMode", "InteractiveModeConfig"]
