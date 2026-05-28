"""Agent gateway contract + production implementations for ``molexp.harness``.

Phase 2 shipped only the :class:`AgentGateway` Protocol. Spec
``harness-as-mode-substrate-03a`` adds the production
:class:`RouterBackedAgentGateway` impl, driven by
:class:`molexp.agent.router.Router`. The in-memory
:class:`StubAgentGateway` lives at ``molexp.harness.gateways.stub`` and is
intentionally **not** re-exported — tests reach for it via its full
dotted path so a stray ``from molexp.harness.gateways import StubAgentGateway``
will fail loudly.
"""

from __future__ import annotations

from molexp.harness.gateways.gateway import AgentGateway
from molexp.harness.gateways.router_backed import RouterBackedAgentGateway

__all__ = ["AgentGateway", "RouterBackedAgentGateway"]
