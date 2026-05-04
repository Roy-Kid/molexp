"""Provider validator + factory for the PydanticAI plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from molexp.agent.model import ModelClient, ModelConfig
from molexp.plugins.model_pydanticai.store import (
    DEEPSEEK_DEFAULT_BASE_URL,
    SUPPORTED_PROVIDERS,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

ModelIoSink = Callable[[str, dict[str, Any]], None]


class PydanticAIProviderValidator:
    """Per-provider field rules.

    One instance per supported sub-provider (anthropic / openai /
    openai-compatible / deepseek / google) so the core registry can
    look up validation per ``ModelConfig.provider_name``.
    """

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def validate(self, config: ModelConfig) -> tuple[str, ...]:
        errors: list[str] = []
        if config.provider_name != self.provider_name:
            errors.append(
                f"validator bound to '{self.provider_name}', got '{config.provider_name}'"
            )
        if config.provider_name not in SUPPORTED_PROVIDERS:
            errors.append(
                f"unsupported provider '{config.provider_name}'; "
                f"expected one of {sorted(SUPPORTED_PROVIDERS)}"
            )
        if not config.model:
            errors.append("model is required")
        if config.provider_name == "openai-compatible" and not config.base_url:
            errors.append("base_url is required for openai-compatible")
        return tuple(errors)


class PydanticAIModelClientFactory:
    """Construct a :class:`PydanticAIModelClient` from a :class:`ModelConfig`.

    One instance per supported sub-provider — the core registry maps
    ``ModelConfig.provider_name`` directly onto a factory.
    """

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def create(
        self,
        config: ModelConfig,
        *,
        model_io_sink: ModelIoSink | None = None,
    ) -> ModelClient:
        from molexp.plugins.model_pydanticai.client import PydanticAIModelClient

        model = build_model(config)
        return PydanticAIModelClient(
            model=model,
            model_name=config.model,
            model_io_sink=model_io_sink,
        )


def build_model(config: ModelConfig) -> "Model":
    """Return a pydantic-ai :class:`Model` matching ``config``.

    Each branch instantiates the provider-specific Provider with the
    user's API key (and optional ``base_url``) so we never mutate the
    process env.
    """

    if config.provider_name == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            api_key=config.api_key or "",
            base_url=config.base_url or None,
        )
        return AnthropicModel(config.model, provider=provider)

    if config.provider_name in ("openai", "openai-compatible"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=config.api_key or "",
            base_url=config.base_url or None,
        )
        return OpenAIChatModel(config.model, provider=provider)

    if config.provider_name == "deepseek":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=config.api_key or "",
            base_url=config.base_url or DEEPSEEK_DEFAULT_BASE_URL,
        )
        return OpenAIChatModel(config.model, provider=provider)

    if config.provider_name == "google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        provider = GoogleProvider(
            api_key=config.api_key or "",
            base_url=config.base_url or None,
        )
        return GoogleModel(config.model, provider=provider)

    raise ValueError(f"Unsupported provider '{config.provider_name}'")


__all__ = [
    "PydanticAIModelClientFactory",
    "PydanticAIProviderValidator",
    "SUPPORTED_PROVIDERS",
    "build_model",
]
