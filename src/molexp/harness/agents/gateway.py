"""``AgentGateway`` — single Protocol every agent-call backend implements.

Harness stages program against this Protocol, never against a concrete
class. Two implementations ship today:
:class:`molexp.harness.agents.stub.StubAgentGateway` (in-memory, test-only)
and :class:`molexp.harness.agents.router_backed.RouterBackedAgentGateway`
(production, driven by :class:`molexp.agent.router.Router`; added by spec
``harness-as-mode-substrate-03a``).

Per :file:`.claude/notes/harness-goal.md` §10.2: every implementation MUST
persist both ``AgentCallResult.output_artifact`` and
``AgentCallResult.raw_response_artifact`` to the run's artifact store
before returning. The raw artifact is the verbatim audit record;
downstream stages read only the parsed output.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from molexp.harness.schemas import AgentCallResult, AgentCallSpec

__all__ = ["AgentGateway"]


@runtime_checkable
class AgentGateway(Protocol):
    """Structural type for any agent-call backend."""

    async def call(self, spec: AgentCallSpec) -> AgentCallResult: ...
