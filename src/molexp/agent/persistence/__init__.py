"""Generic three-layer resource storage.

Shared by every kind that exposes a registrations + user + workspace
tier (skills, MCP tool catalog, MCP servers, future extensions). Each
kind subclasses :class:`TieredResourceStore` with its own
:class:`ResourceSpec` subclass.
"""

from molexp.agent.persistence.router import tiered_router_factory
from molexp.agent.persistence.tiered import (
    ResourceSpec,
    Scope,
    TieredResourceStore,
)

__all__ = [
    "ResourceSpec",
    "Scope",
    "TieredResourceStore",
    "tiered_router_factory",
]
