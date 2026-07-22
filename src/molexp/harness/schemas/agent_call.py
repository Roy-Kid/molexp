"""``AgentCallSpec`` + ``AgentCallResult`` — the agent-gateway wire contract.

Per ``.claude/notes/harness-goal.md`` §10.1-10.2: every harness call into
an agent flows through this typed envelope, so the audit log can answer
"which agent ran, against which inputs, returning which artifacts" without
guessing. The two ``PlanArtifactRef`` fields on the result are an explicit
invariant: any :class:`molexp.harness.gateways.gateway.AgentGateway` impl
MUST persist both the parsed output and the raw response before returning,
giving the audit pipeline a verbatim record of what the LLM emitted (not
just the parsed shape downstream stages see).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.artifact import PlanArtifactRef

__all__ = ["AgentCallResult", "AgentCallSpec"]


class AgentCallSpec(BaseModel):
    """Request to an :class:`AgentGateway`.

    ``output_schema`` is intentionally a free-form ``dict`` (typically
    populated from ``SomeModel.model_json_schema()`); the Phase-1 substrate
    does not enforce it at the boundary — that's the real gateway impl's
    job once it lands.
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str
    input_artifact_ids: list[str]
    prompt_artifact_id: str | None = None
    output_schema: dict
    temperature: float = 0.2
    metadata: dict[str, str] = Field(default_factory=dict)
    #: Explicit per-call tier override (``"cheap"`` / ``"default"`` / ``"heavy"``).
    #: When set, wins over the agent registry map. Never implicit — stages that
    #: need HEAVY for repair must pass this field.
    tier: str | None = None
    #: When ``False``, strip MCP toolsets for this call (catalog / compose-only).
    #: When ``True``, require the agent's configured MCP servers. ``None`` keeps
    #: the agent registry default (codegen agents get molmcp).
    use_mcp: bool | None = None


class AgentCallResult(BaseModel):
    """Response from an :class:`AgentGateway`.

    Both ``output_artifact`` and ``raw_response_artifact`` MUST already be
    persisted in the run's :class:`ArtifactStore` before the gateway
    returns this result; the refs carried here are the audit record.
    """

    model_config = ConfigDict(frozen=True)

    output_artifact: PlanArtifactRef
    raw_response_artifact: PlanArtifactRef
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
