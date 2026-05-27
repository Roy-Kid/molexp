"""Agent gateway contract for ``molexp.harness``.

Phase 2 ships only the :class:`AgentGateway` Protocol. The in-memory
:class:`StubAgentGateway` lives at ``molexp.harness.agents.stub`` and is
intentionally **not** re-exported here — production code can only see the
Protocol; tests reach for the stub via its full dotted path so a stray
``from molexp.harness.agents import StubAgentGateway`` will fail loudly.
"""

from __future__ import annotations

from molexp.harness.agents.gateway import AgentGateway

__all__ = ["AgentGateway"]
