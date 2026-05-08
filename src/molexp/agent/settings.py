"""Agent settings.

Workspace-scoped tunables for the new ``AgentRunner`` surface. The old
``ModelConfig`` / ``ProviderConfig`` aliases were removed alongside the
``ModelClient`` boundary; users now pass a model string straight to
``AgentRunner(..., model="openai:gpt-5.2")``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AgentSettings(BaseModel):
    """Workspace-scoped agent configuration."""

    model_config = ConfigDict(frozen=True)

    base_system_prompt: str = ""
    workspace_addendum: str = ""
    default_model: str | None = None
    max_context_chars: int = 200_000
    extras: dict[str, object] = Field(default_factory=dict)


__all__ = ["AgentSettings"]
