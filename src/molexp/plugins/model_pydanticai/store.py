"""Workspace-scoped :class:`ModelConfig` store for the PydanticAI plugin.

Persists to ``.molexp-agent/provider.json`` via atomic temp+rename so
concurrent route handlers cannot interleave updates. The file holds
the plaintext API key — never expose it through the HTTP API; admin
routes return :func:`mask_api_key` previews.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Literal

from molexp.agent.model import ModelConfig

PROVIDER_FILENAME = "provider.json"
AGENT_DIRNAME = ".molexp-agent"

ProviderName = Literal[
    "anthropic", "openai", "google", "deepseek", "openai-compatible"
]

SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = (
    "anthropic",
    "openai",
    "google",
    "deepseek",
    "openai-compatible",
)

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash",
    "deepseek": "deepseek-chat",
    "openai-compatible": "gpt-4o",
}

# DeepSeek exposes an OpenAI-compatible API at this endpoint; the
# plugin substitutes it when the user picks "deepseek" but leaves
# ``base_url`` empty.
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

# Env vars pydantic-ai consults when no explicit api_key is set.
ENV_VAR_FOR_PROVIDER: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def default_config() -> ModelConfig:
    return ModelConfig(
        provider_name="anthropic",
        model=DEFAULT_MODELS["anthropic"],
        api_key=None,
        base_url=None,
        instructions="",
    )


def mask_api_key(key: str | None) -> str:
    """Mask an API key for display: ``"sk-...1234"`` (last 4 chars)."""

    if not key:
        return ""
    if len(key) < 8:
        return "*" * len(key)
    return f"{key[:3]}...{key[-4:]}"


class ProviderStore:
    """File-backed store for the workspace's :class:`ModelConfig`.

    Reads parse the JSON file on demand; writes are atomic and
    serialized by an internal lock. Missing files yield the default
    config so the UI can render an empty state without erroring.
    """

    def __init__(self, root: str | Path) -> None:
        self._path = Path(root) / AGENT_DIRNAME / PROVIDER_FILENAME
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> ModelConfig:
        if not self._path.exists():
            return default_config()
        payload = json.loads(self._path.read_text())
        if not isinstance(payload, dict):
            return default_config()
        return ModelConfig(
            provider_name=str(payload.get("provider_name") or "anthropic"),
            model=str(payload.get("model") or DEFAULT_MODELS["anthropic"]),
            api_key=payload.get("api_key") or None,
            base_url=payload.get("base_url") or None,
            instructions=str(payload.get("instructions") or ""),
            extras=dict(payload.get("extras") or {}),
        )

    def save(self, config: ModelConfig) -> ModelConfig:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "provider_name": config.provider_name,
                        "model": config.model,
                        "api_key": config.api_key or "",
                        "base_url": config.base_url or "",
                        "instructions": config.instructions,
                        "extras": dict(config.extras),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            os.replace(tmp, self._path)
        return config

    def update(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        instructions: str | None = None,
    ) -> ModelConfig:
        """Patch fields. ``api_key=""`` clears the key; ``None`` leaves it.

        Switching ``provider_name`` without an explicit ``model`` resets
        the model to that provider's default.
        """

        current = self.load()
        next_provider = provider_name or current.provider_name
        if model is not None:
            next_model = model
        elif provider_name is not None and provider_name != current.provider_name:
            next_model = DEFAULT_MODELS.get(provider_name, current.model)
        else:
            next_model = current.model
        next_api_key = api_key if api_key is not None else (current.api_key or "")
        next_base_url = base_url if base_url is not None else (current.base_url or "")
        next_instructions = (
            instructions if instructions is not None else current.instructions
        )
        updated = ModelConfig(
            provider_name=next_provider,
            model=next_model,
            api_key=next_api_key or None,
            base_url=next_base_url or None,
            instructions=next_instructions,
            extras=dict(current.extras),
        )
        return self.save(updated)


__all__ = [
    "AGENT_DIRNAME",
    "DEEPSEEK_DEFAULT_BASE_URL",
    "DEFAULT_MODELS",
    "ENV_VAR_FOR_PROVIDER",
    "PROVIDER_FILENAME",
    "ProviderName",
    "ProviderStore",
    "SUPPORTED_PROVIDERS",
    "default_config",
    "mask_api_key",
]
