"""TaskGraphCompiler for converting between TaskGraph and JSON IR.

This compiler provides bidirectional conversion between Python TaskGraph objects
and a language-agnostic JSON intermediate representation (IR). The JSON IR serves
as the interchange format between the Python backend and TypeScript frontend.

The compiler is responsible for:
- Serializing TaskGraph to JSON (to_json)
- Deserializing JSON to TaskGraph (from_json)
- Validating JSON structure

The compiler does NOT execute tasks or workflows - it only handles representation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from .task_base import Task, EmptyConfig
from .task_graph import Edge, TaskGraph, TaskNode


class TaskGraphCompiler:
    """Compiles TaskGraph to/from JSON intermediate representation.
    
    The JSON IR is the single source of truth for workflow definitions
    shared between Python backend and TypeScript frontend.
    
    Parameters
    ----------
    task_registry : dict[str, type[Task]] | None
        Optional registry mapping task type names to Task classes.
        Required for from_json() to reconstruct actual Task instances.
    """
    
    def __init__(self, task_registry: dict[str, type[Task]] | None = None):
        """Initialize compiler with optional task registry."""
        self.task_registry = task_registry or {}
    
    def to_json(self, graph: TaskGraph) -> dict[str, Any]:
        """Convert TaskGraph to JSON intermediate representation.
        
        Produces a dictionary that can be serialized to JSON and matches
        the TaskGraphJson schema.
        
        Parameters
        ----------
        graph : TaskGraph
            The task graph to serialize.
            
        Returns
        -------
        dict[str, Any]
            JSON-serializable dictionary following the IR schema:
            {
                "name": str,
                "version": str | null,
                "nodes": [TaskNodeJson, ...],
                "edges": [EdgeJson, ...],
                "metadata": {...}
            }
        """
        # Serialize nodes
        nodes_json = []
        for node in graph.nodes.values():
            node_json = {
                "id": node.id,
                "type": node.type,
                "label": node.label,
                "params": self._serialize_task_params(node.task),
                "metadata": node.metadata,
            }
            nodes_json.append(node_json)
        
        # Serialize edges
        edges_json = []
        for edge in graph.edges:
            edge_json = {
                "from": edge.from_id,
                "to": edge.to_id,
                "kind": edge.kind,
                "metadata": edge.metadata,
            }
            edges_json.append(edge_json)
        
        # Build complete graph JSON
        return {
            "name": graph.name,
            "version": graph.version,
            "nodes": nodes_json,
            "edges": edges_json,
            "metadata": graph.metadata,
        }
    
    def from_json(self, payload: dict[str, Any]) -> TaskGraph:
        """Reconstruct TaskGraph from JSON intermediate representation.
        
        Validates the JSON structure and rebuilds a TaskGraph instance.
        If a task_registry is provided, reconstructs actual Task instances;
        otherwise creates placeholder tasks.
        
        Parameters
        ----------
        payload : dict[str, Any]
            JSON IR following the TaskGraphJson schema.
            
        Returns
        -------
        TaskGraph
            Reconstructed task graph.
            
        Raises
        ------
        ValueError
            If JSON structure is invalid or required fields are missing.
        """
        # Validate required top-level fields
        self._validate_json_structure(payload)
        
        # Reconstruct nodes
        nodes: dict[str, TaskNode] = {}
        for node_json in payload["nodes"]:
            task = self._reconstruct_task(
                task_type=node_json["type"],
                params=node_json.get("params", {}),
            )
            node = TaskNode(
                id=node_json["id"],
                task=task,
                label=node_json.get("label"),
                metadata=node_json.get("metadata", {}),
            )
            nodes[node.id] = node
        
        # Reconstruct edges
        edges: list[Edge] = []
        for edge_json in payload["edges"]:
            edge = Edge(
                from_id=edge_json["from"],
                to_id=edge_json["to"],
                kind=edge_json.get("kind", "depends"),
                metadata=edge_json.get("metadata", {}),
            )
            edges.append(edge)
        
        # Build graph
        graph = TaskGraph(
            name=payload["name"],
            nodes=nodes,
            edges=edges,
            version=payload.get("version"),
            metadata=payload.get("metadata", {}),
        )
        
        return graph
    
    def _serialize_task_params(self, task: Task) -> dict[str, Any]:
        """Extract and serialize task configuration parameters.
        
        Parameters
        ----------
        task : Task
            Task instance to extract params from.
            
        Returns
        -------
        dict[str, Any]
            Serialized parameters.
        """
        # Get the default config model
        cfg_model = task.cfg_model
        
        # If it's the empty config, return empty dict
        if cfg_model == EmptyConfig:
            return {}
        
        # Get the JSON schema to understand available fields
        schema = cfg_model.model_json_schema()
        properties = schema.get("properties", {})
        
        # Build params dict from schema defaults
        params = {}
        for field_name, field_info in properties.items():
            # Get default value if available
            if "default" in field_info:
                params[field_name] = field_info["default"]
        
        return params
    
    def _reconstruct_task(self, task_type: str, params: dict[str, Any]) -> Task:
        """Reconstruct a Task instance from type and params.
        
        Parameters
        ----------
        task_type : str
            Task class name.
        params : dict[str, Any]
            Task parameters.
            
        Returns
        -------
        Task
            Reconstructed task instance or placeholder.
        """
        # Look up task class in registry
        task_class = self.task_registry.get(task_type)
        
        if task_class is None:
            # Create a placeholder task if not in registry
            return PlaceholderTask(task_type=task_type, params=params)
        
        # Instantiate the task
        # Note: This assumes tasks can be instantiated without upstreams
        # In practice, we may need to reconstruct upstreams from edges
        task = task_class()
        
        # TODO: Set task configuration from params
        # This requires storing cfg on the task or re-implementing task initialization
        
        return task
    
    def _validate_json_structure(self, payload: dict[str, Any]) -> None:
        """Validate that JSON has required fields and correct types.
        
        Parameters
        ----------
        payload : dict[str, Any]
            JSON IR to validate.
            
        Raises
        ------
        ValueError
            If structure is invalid.
        """
        # Check top-level fields
        if "name" not in payload:
            raise ValueError("JSON IR missing required field: 'name'")
        if "nodes" not in payload:
            raise ValueError("JSON IR missing required field: 'nodes'")
        if "edges" not in payload:
            raise ValueError("JSON IR missing required field: 'edges'")
        
        if not isinstance(payload["nodes"], list):
            raise ValueError("Field 'nodes' must be a list")
        if not isinstance(payload["edges"], list):
            raise ValueError("Field 'edges' must be a list")
        
        # Validate each node
        for i, node in enumerate(payload["nodes"]):
            if not isinstance(node, dict):
                raise ValueError(f"Node {i} must be a dict")
            if "id" not in node:
                raise ValueError(f"Node {i} missing required field: 'id'")
            if "type" not in node:
                raise ValueError(f"Node {i} missing required field: 'type'")
        
        # Validate each edge
        for i, edge in enumerate(payload["edges"]):
            if not isinstance(edge, dict):
                raise ValueError(f"Edge {i} must be a dict")
            if "from" not in edge:
                raise ValueError(f"Edge {i} missing required field: 'from'")
            if "to" not in edge:
                raise ValueError(f"Edge {i} missing required field: 'to'")


class PlaceholderTask(Task[EmptyConfig, Any]):
    """Placeholder task for types not in registry.
    
    This allows from_json() to succeed even when task classes
    aren't available, storing the type and params for later use.
    """
    
    def __init__(self, task_type: str, params: dict[str, Any]):
        """Initialize placeholder with type and params."""
        super().__init__(name=task_type)
        self.stored_type = task_type
        self.stored_params = params
    
    def forward(self, *data_args: Any, cfg: EmptyConfig) -> Any:
        """Placeholder tasks cannot execute."""
        raise NotImplementedError(
            f"Cannot execute placeholder task of type '{self.stored_type}'. "
            f"Task class not found in registry."
        )
