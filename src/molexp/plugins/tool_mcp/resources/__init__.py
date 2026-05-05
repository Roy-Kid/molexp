"""Generic resource layer for the agent plugin.

This package contains the kind-agnostic primitives (:class:`Scope`,
:class:`ResourceSpec`, :class:`TieredResourceStore`) shared by skill,
tool, and MCP-server stores. Per-kind storage modules live next to it
in ``agent_pydanticai/`` and instantiate the generic with their own
spec types.
"""

from .base import ResourceSpec, Scope, TieredResourceStore
from .router import tiered_router_factory

__all__ = [
    "ResourceSpec",
    "Scope",
    "TieredResourceStore",
    "tiered_router_factory",
]
