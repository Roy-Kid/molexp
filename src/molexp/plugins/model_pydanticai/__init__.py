"""PydanticAI as a :class:`molexp.agent.ModelClient` plugin.

Importing this package registers the factory + validator with the
core model registry. The plugin owns ``model_io.jsonl`` writes and
``provider_blobs/`` storage exclusively; the harness never parses
either layer.
"""

from __future__ import annotations

from molexp.agent.model_registry import register_model_provider
from molexp.plugins.model_pydanticai.client import PydanticAIModelClient
from molexp.plugins.model_pydanticai.credentials import (
    CredentialStatus,
    ProbeResult,
    ProviderConfigPublic,
    check_credentials,
    probe_provider,
    to_public,
)
from molexp.plugins.model_pydanticai.provider import (
    PydanticAIModelClientFactory,
    PydanticAIProviderValidator,
    SUPPORTED_PROVIDERS,
)
from molexp.plugins.model_pydanticai.store import (
    DEFAULT_MODELS,
    ProviderStore,
    mask_api_key,
)


def _register() -> None:
    for name in SUPPORTED_PROVIDERS:
        register_model_provider(
            PydanticAIModelClientFactory(name),
            PydanticAIProviderValidator(name),
        )


_register()


__all__ = [
    "CredentialStatus",
    "DEFAULT_MODELS",
    "ProbeResult",
    "ProviderConfigPublic",
    "PydanticAIModelClient",
    "PydanticAIModelClientFactory",
    "PydanticAIProviderValidator",
    "ProviderStore",
    "SUPPORTED_PROVIDERS",
    "check_credentials",
    "mask_api_key",
    "probe_provider",
    "to_public",
]
