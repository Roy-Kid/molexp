"""Process-level registry for model plugins.

Plugins register themselves on import; the harness looks them up by
``provider_name`` when constructing an :class:`AgentService` from a
workspace's :class:`ModelConfig`. The registry is intentionally small
and stdlib-only.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable

from molexp.agent.model import (
    ModelClient,
    ModelClientFactory,
    ModelConfig,
    ProviderConfigValidator,
)

ModelIoSink = Callable[[str, dict[str, Any]], None]
"""Optional callback the plugin can use to record one
``(request, response)`` pair to ``model_io.jsonl``."""


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

    def get_validator(self, provider_name: str) -> ProviderConfigValidator | None:
        with self._lock:
            return self._validators.get(provider_name)

    def create_client(
        self,
        config: ModelConfig,
        *,
        model_io_sink: ModelIoSink | None = None,
    ) -> ModelClient:
        validator = self.get_validator(config.provider_name)
        if validator is not None:
            errors = validator.validate(config)
            if errors:
                raise ValueError(
                    f"Provider config invalid for '{config.provider_name}': " + "; ".join(errors)
                )
        factory = self.get_factory(config.provider_name)
        if model_io_sink is None:
            return factory.create(config)
        return factory.create(config, model_io_sink=model_io_sink)


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


def create_model_client(
    config: ModelConfig,
    *,
    model_io_sink: ModelIoSink | None = None,
) -> ModelClient:
    """Validate ``config`` and instantiate the plugin's :class:`ModelClient`."""

    return _default_registry.create_client(config, model_io_sink=model_io_sink)


__all__ = [
    "UnknownProviderError",
    "create_model_client",
    "get_model_provider",
    "get_provider_validator",
    "list_providers",
    "register_model_provider",
    "unregister_model_provider",
]
