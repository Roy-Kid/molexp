"""Node Registry for managing registered node types."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from ..node import Node
from .metadata import NodeMetadata

logger = logging.getLogger(__name__)


class NodeRegistration:
    """Registration entry for a node type.
    
    Attributes:
        node_id: Unique node identifier
        node_class: Node class
        config_class: Configuration model class
        metadata: Node metadata
    """
    
    def __init__(
        self,
        node_id: str,
        node_class: Type[Node],
        config_class: Type[BaseModel],
        metadata: NodeMetadata,
    ):
        self.node_id = node_id
        self.node_class = node_class
        self.config_class = config_class
        self.metadata = metadata
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses.
        
        Returns:
            Dictionary with node metadata and config schema
        """
        return {
            "id": self.node_id,
            "label": self.metadata.label,
            "category": self.metadata.category,
            "description": self.metadata.description,
            "inputs": [port.model_dump(mode="json") for port in self.metadata.inputs],
            "outputs": [port.model_dump(mode="json") for port in self.metadata.outputs],
            "icon": self.metadata.icon,
            "tags": self.metadata.tags,
            "config_schema": self.config_class.model_json_schema(),
        }


class NodeRegistry:
    """Central registry for all node types.
    
    This registry is populated via entry points and provides
    metadata and configuration schemas for all registered nodes.
    """
    
    def __init__(self):
        self._nodes: Dict[str, NodeRegistration] = {}
        self._loaded = False
    
    def register(
        self,
        node_id: str,
        node_class: Type[Node],
        metadata: NodeMetadata,
        config_class: Optional[Type[BaseModel]] = None,
    ) -> None:
        """Register a node type.
        
        Args:
            node_id: Unique node identifier (e.g., "io.read_file")
            node_class: Node class
            metadata: Node metadata
            config_class: Configuration class (defaults to node_class.config_type)
            
        Raises:
            ValueError: If node_id is already registered
        """
        if node_id in self._nodes:
            raise ValueError(
                f"Node ID '{node_id}' is already registered. "
                f"Existing: {self._nodes[node_id].node_class.__name__}, "
                f"New: {node_class.__name__}"
            )
        
        # Use node's config_type if not provided
        if config_class is None:
            if not hasattr(node_class, "config_type"):
                raise ValueError(
                    f"Node class {node_class.__name__} must have 'config_type' attribute "
                    f"or config_class must be provided"
                )
            config_class = node_class.config_type
        
        registration = NodeRegistration(
            node_id=node_id,
            node_class=node_class,
            config_class=config_class,
            metadata=metadata,
        )
        
        self._nodes[node_id] = registration
        logger.info(f"Registered node: {node_id} ({node_class.__name__})")

        # Bridge to IR Registry for execution
        try:
            from molexp.ir.registry import registry as ir_registry
            # The ir_registry.register returns a decorator, so we call it with the class
            ir_registry.register(node_id, config_class)(node_class)
            logger.debug(f"Synced node to IR registry: {node_id}")
        except ImportError:
            logger.warning(f"Could not sync {node_id} to IR registry (module not found)")
    
    def get(self, node_id: str) -> Optional[NodeRegistration]:
        """Get registration for a node ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Registration entry or None if not found
        """
        return self._nodes.get(node_id)
    
    def list_all(self) -> List[NodeRegistration]:
        """List all registered nodes.
        
        Returns:
            List of all registrations
        """
        return list(self._nodes.values())
    
    def list_by_category(self, category: str) -> List[NodeRegistration]:
        """List nodes in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of registrations in that category
        """
        return [
            reg for reg in self._nodes.values()
            if reg.metadata.category == category
        ]
    
    def to_dict(self) -> dict[str, Any]:
        """Export all nodes as dictionary for API.
        
        Returns:
            Dictionary with all node definitions
        """
        return {
            "nodes": [reg.to_dict() for reg in self._nodes.values()]
        }
    
    def mark_loaded(self) -> None:
        """Mark registry as loaded (plugins have been discovered)."""
        self._loaded = True
    
    def is_loaded(self) -> bool:
        """Check if plugins have been loaded.
        
        Returns:
            True if plugins loaded
        """
        return self._loaded


# Global registry instance
_global_registry: Optional[NodeRegistry] = None


def get_node_registry() -> NodeRegistry:
    """Get the global node registry instance.
    
    Returns:
        Global NodeRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = NodeRegistry()
    return _global_registry
