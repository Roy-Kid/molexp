"""Capability-registry plugin."""

from __future__ import annotations

from molexp.harness.host.context import Context
from molexp.harness.host.keys import Keys
from molexp.harness.registry.capability_registry import CapabilityRegistry

__all__ = ["CapabilitiesPlugin"]


class CapabilitiesPlugin:
    """Publish an existing :class:`CapabilityRegistry`."""

    name = "capabilities"
    inject: tuple[str, ...] = ()

    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def apply(self, ctx: Context) -> None:
        """Provide ``ctx.capabilities``."""
        ctx.provide(Keys.CAPABILITIES, self._registry)
