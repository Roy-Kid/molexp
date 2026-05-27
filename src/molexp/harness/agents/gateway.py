"""``AgentGateway`` — single Protocol every agent-call backend implements.

Harness stages program against this Protocol, never against a concrete
class. The Phase-2 substrate ships only the contract + a test stub
(:class:`molexp.harness.agents.stub.StubAgentGateway`); the real
LLM-backed impl lands in Phase 5+.

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
