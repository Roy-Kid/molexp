"""Agent settings + provider config.

No provider SDK imports here. ``ProviderConfig`` is the generic shape
the UI and admin routes work with; the model plugin attaches concrete
factory + validator implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from molexp.agent.model import ModelConfig


@dataclass(frozen=True)
class AgentSettings:
    """Workspace-scoped agent configuration."""

    base_system_prompt: str = ""
    workspace_addendum: str = ""
    default_provider: str | None = None
    max_context_chars: int = 200_000
    extras: dict[str, object] = field(default_factory=dict)


# Re-export for convenience: the harness treats ``ProviderConfig`` and
# ``ModelConfig`` as synonyms; ``ModelConfig`` is the canonical name
# from the model boundary, ``ProviderConfig`` is the admin-route name.
ProviderConfig = ModelConfig

__all__ = ["AgentSettings", "ProviderConfig", "ModelConfig"]
