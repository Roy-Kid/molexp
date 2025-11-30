"""Task graph data structures for workflow representation.

This module defines the core data structures for representing workflows as
directed acyclic graphs (DAGs) of tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .task_base import Task


@dataclass
class Edge:
    """Directed edge in a task graph.
    
    Represents a dependency relationship between two tasks in a workflow.
    
    Attributes
    ----------
    from_id : str
        Source node ID.
    to_id : str
        Target node ID.
    kind : str | None
        Edge type (e.g., "depends", "data"). Defaults to "depends".
    metadata : dict[str, Any]
        Additional metadata for UI or other purposes.
    """
    
    from_id: str
    to_id: str
    kind: str | None = "depends"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNode:
    """Wrapper for a Task with graph metadata.
    
    Associates a Task instance with a unique ID and additional metadata
    needed for the workflow graph.
    
    Attributes
    ----------
    id : str
        Unique identifier for this node within the graph.
    task : Task
        The task instance to execute.
    label : str | None
        Human-readable label for display purposes.
    metadata : dict[str, Any]
        Additional metadata (e.g., UI layout information).
    """
    
    id: str
    task: Task
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def type(self) -> str:
        """Return the task type (class name)."""
        return self.task.__class__.__name__
    
    @property
    def params(self) -> dict[str, Any]:
        """Extract task parameters as a dict."""
        # Get the config model and extract its fields
        if hasattr(self.task, 'cfg_model'):
            schema = self.task.cfg_model.model_json_schema()
            props = schema.get('properties', {})
            # Build params from defaults or current state
            result = {}
            for key in props:
                if hasattr(self.task.cfg_model, key):
                    result[key] = getattr(self.task.cfg_model, key)
            return result
        return {}


@dataclass
class TaskGraph:
    """Complete workflow graph.
    
    Represents a full workflow as a collection of tasks (nodes) and their
    dependencies (edges).
    
    Attributes
    ----------
    name : str
        Human-readable workflow name.
    nodes : dict[str, TaskNode]
        Mapping from node ID to TaskNode.
    edges : list[Edge]
        List of directed edges between nodes.
    version : str | None
        Optional version identifier.
    metadata : dict[str, Any]
        Additional workflow-level metadata.
    """
    
    name: str
    nodes: dict[str, TaskNode]
    edges: list[Edge]
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_node(self, node_id: str) -> TaskNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def add_node(self, node: TaskNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
    
    def get_dependencies(self, node_id: str) -> list[str]:
        """Get IDs of all nodes that the given node depends on."""
        return [edge.from_id for edge in self.edges if edge.to_id == node_id]
    
    def get_dependents(self, node_id: str) -> list[str]:
        """Get IDs of all nodes that depend on the given node."""
        return [edge.to_id for edge in self.edges if edge.from_id == node_id]
