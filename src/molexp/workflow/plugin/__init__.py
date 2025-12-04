"""Plugin system for molexp nodes.

This package provides the infrastructure for discovering and loading
node plugins via Python entry points.
"""

from .metadata import NodeMetadata, PortMetadata
from .registry import NodeRegistry, get_node_registry
from .loader import load_plugins

__all__ = [
    "NodeMetadata",
    "PortMetadata",
    "NodeRegistry",
    "get_node_registry",
    "load_plugins",
]
