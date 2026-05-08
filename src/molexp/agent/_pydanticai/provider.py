"""``PydanticAIProvider`` — concrete :class:`Provider` impl on top of pydantic-ai.

Sole pydantic-ai-using site for the structured Provider abstraction
consumed by ``molexp.agent.modes.plan``. Resolves tiers to concrete
model IDs at construction time; lazily materialises one
``pydantic_ai.Agent`` per ``(tier, schema)`` pair.

The model JSON output is parsed with the requested pydantic schema;
malformed output raises :class:`pydantic.ValidationError` straight
through to the calling task — provider does not retry on parse failure.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from molexp.agent.modes._plan_protocols import ModelTier

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Default tier → model mapping. Operators override via constructor kwargs;
# the values below are placeholders chosen by capability class, not
# binding contracts.
_DEFAULT_TIERS: dict[ModelTier, str] = {
    ModelTier.CHEAP: "anthropic:claude-haiku-4-5",
    ModelTier.DEFAULT: "anthropic:claude-sonnet-4-6",
    ModelTier.HEAVY: "anthropic:claude-opus-4-7",
}


class PydanticAIProvider:
    """Tier-routed, schema-typed provider built on ``pydantic_ai.Agent``."""

    def __init__(self, tiers: dict[ModelTier, str] | None = None) -> None:
        self._tier_models: dict[ModelTier, str] = dict(_DEFAULT_TIERS)
        if tiers:
            self._tier_models.update(tiers)
        self._agents: dict[tuple[ModelTier, type[BaseModel]], Agent[None, Any]] = {}

    def _agent_for(
        self,
        tier: ModelTier,
        schema: type[SchemaT],
        system: str,
    ) -> Agent[None, SchemaT]:
        key = (tier, schema)
        if key not in self._agents:
            model = self._tier_models[tier]
            self._agents[key] = Agent(
                model=model,
                output_type=schema,
                system_prompt=system,
            )
        return self._agents[key]  # type: ignore[return-value]

    async def invoke(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",  # noqa: ARG002 — accepted for symmetry, used by tracing impls
    ) -> SchemaT:
        agent = self._agent_for(tier, schema, system)
        result = await agent.run(user)
        output = getattr(result, "output", None)
        if not isinstance(output, schema):
            raise TypeError(
                f"Provider expected {schema.__name__} from tier={tier.value}; "
                f"received {type(output).__name__}."
            )
        return output


__all__ = ["PydanticAIProvider"]
