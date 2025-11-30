"""Workflow registry for managing pre-defined Python workflows.

This module provides a simple registry to store and retrieve TaskGraph
instances that are defined in Python code.
"""

from __future__ import annotations

from .task_graph import TaskGraph


class WorkflowRegistry:
    """Registry of Python-defined workflows.
    
    The registry allows workflows defined in Python to be stored and
    retrieved by ID, enabling export to JSON IR via the API.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._workflows: dict[str, TaskGraph] = {}
    
    def register(self, workflow_id: str, graph: TaskGraph) -> None:
        """Register a workflow graph.
        
        Parameters
        ----------
        workflow_id : str
            Unique identifier for the workflow.
        graph : TaskGraph
            The workflow graph to register.
        """
        self._workflows[workflow_id] = graph
    
    def get(self, workflow_id: str) -> TaskGraph | None:
        """Get a workflow by ID.
        
        Parameters
        ----------
        workflow_id : str
            Workflow identifier.
            
        Returns
        -------
        TaskGraph | None
            The workflow graph, or None if not found.
        """
        return self._workflows.get(workflow_id)
    
    def list(self) -> list[str]:
        """List all registered workflow IDs.
        
        Returns
        -------
        list[str]
            List of workflow IDs.
        """
        return list(self._workflows.keys())
    
    def unregister(self, workflow_id: str) -> bool:
        """Remove a workflow from the registry.
        
        Parameters
        ----------
        workflow_id : str
            Workflow identifier.
            
        Returns
        -------
        bool
            True if workflow was removed, False if not found.
        """
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False


# Global registry instance
_global_registry = WorkflowRegistry()


def get_workflow_registry() -> WorkflowRegistry:
    """Get the global workflow registry instance.
    
    Returns
    -------
    WorkflowRegistry
        The global registry.
    """
    return _global_registry
