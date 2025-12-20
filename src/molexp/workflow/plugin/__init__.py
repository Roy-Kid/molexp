"""Plugin system for molexp nodes.

This package provides the infrastructure for discovering and loading
node plugins via Python entry points.
"""

from .loader import load_plugins
from .metadata import NodeMetadata, PortMetadata
from .registry import NodeRegistry, get_node_registry

__all__ = [
    "NodeMetadata",
    "PortMetadata",
    "NodeRegistry",
    "get_node_registry",
    "load_plugins",
]
