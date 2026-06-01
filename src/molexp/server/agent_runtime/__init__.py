"""Server-side agent session runtime (spec 00a).

The plain runtime containers that own live agent sessions on the server:
:class:`AgentSessionRegistry` (per-workspace lookup, seated on
``app.state.agent_runtime``), :class:`AgentSessionRuntime` (one runner +
session + current turn), and :class:`AgentTurn` (one background turn task).

Boundary: ``molexp.server.schemas`` must never import this subpackage, and
these runtime objects must never appear in a FastAPI ``response_model`` — the
route layer translates them to frozen wire ``*Response`` shapes explicitly.
"""

from __future__ import annotations

from molexp.server.agent_runtime.registry import AgentSessionRegistry
from molexp.server.agent_runtime.runtime import AgentSessionRuntime
from molexp.server.agent_runtime.turn import AgentTurn

__all__ = ["AgentSessionRegistry", "AgentSessionRuntime", "AgentTurn"]
