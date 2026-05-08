"""Public agent surface.

The entire ``molexp.agent`` layer is rebuilt around four user-visible
names. ``pydantic_ai`` is an implementation detail hidden inside the
private ``_pydanticai/`` subpackage. ``pydantic_graph`` is **not**
imported anywhere under ``agent/`` — multi-step modes (PlanMode) drive
their workflows through the public ``molexp.workflow`` API, the only
sanctioned pydantic-graph site in the project.
"""

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.runner import AgentRunner
from molexp.agent.session import AgentSession

__all__ = [
    "AgentMode",
    "AgentRunResult",
    "AgentRunner",
    "AgentSession",
]
