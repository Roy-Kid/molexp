"""Process-level registry for model plugins (spec §7.1).

Plugins register themselves on import; the harness looks them up by
``provider_name`` when constructing an :class:`AgentService` from a
workspace's :class:`ModelConfig`. The registry is intentionally small
and stdlib-only.
"""

from __future__ import annotations

from threading import Lock

from molexp.agent.model import (
    ModelClient,
    ModelClientFactory,
    ModelConfig,
    ProviderConfigValidator,
)


class UnknownProviderError(KeyError):
    """Raised when a workspace requests a provider with no registered factory."""


class _ModelProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ModelClientFactory] = {}
        self._validators: dict[str, ProviderConfigValidator] = {}
        self._lock = Lock()

    def register(
        self,
        factory: ModelClientFactory,
        validator: ProviderConfigValidator | None = None,
    ) -> None:
        with self._lock:
            self._factories[factory.provider_name] = factory
            if validator is not None:
                self._validators[validator.provider_name] = validator

    def unregister(self, provider_name: str) -> None:
        with self._lock:
            self._factories.pop(provider_name, None)
            self._validators.pop(provider_name, None)

    def providers(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    def get_factory(self, provider_name: str) -> ModelClientFactory:
        with self._lock:
            factory = self._factories.get(provider_name)
        if factory is None:
            raise UnknownProviderError(provider_name)
        return factory

    def get_validator(
        self, provider_name: str
    ) -> ProviderConfigValidator | None:
        with self._lock:
            return self._validators.get(provider_name)

    def create_client(self, config: ModelConfig) -> ModelClient:
        validator = self.get_validator(config.provider_name)
        if validator is not None:
            errors = validator.validate(config)
            if errors:
                raise ValueError(
                    f"Provider config invalid for '{config.provider_name}': "
                    + "; ".join(errors)
                )
        return self.get_factory(config.provider_name).create(config)


_default_registry = _ModelProviderRegistry()


def register_model_provider(
    factory: ModelClientFactory,
    validator: ProviderConfigValidator | None = None,
) -> None:
    """Register a model plugin's factory + optional validator."""

    _default_registry.register(factory, validator)


def unregister_model_provider(provider_name: str) -> None:
    _default_registry.unregister(provider_name)


def list_providers() -> tuple[str, ...]:
    return _default_registry.providers()


def get_model_provider(provider_name: str) -> ModelClientFactory:
    return _default_registry.get_factory(provider_name)


def get_provider_validator(provider_name: str) -> ProviderConfigValidator | None:
    return _default_registry.get_validator(provider_name)


def create_model_client(config: ModelConfig) -> ModelClient:
    """Validate ``config`` and instantiate the plugin's :class:`ModelClient`."""

    return _default_registry.create_client(config)


__all__ = [
    "UnknownProviderError",
    "create_model_client",
    "get_model_provider",
    "get_provider_validator",
    "list_providers",
    "register_model_provider",
    "unregister_model_provider",
]
