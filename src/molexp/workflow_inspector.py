"""Workflow inspection and visualization utilities."""

from __future__ import annotations

from typing import Any

from .task_graph import TaskGraph
from .workflow_registry import WorkflowRegistry


class WorkflowInspector:
    """Inspect and visualize workflow graphs."""

    def __init__(self, registry: WorkflowRegistry) -> None:
        """Initialize workflow inspector.
        
        Args:
            registry: Workflow registry to inspect
        """
        self.registry = registry

    def get_workflow_info(self, workflow_id: str) -> dict[str, Any]:
        """Get detailed workflow information.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Dictionary with workflow information
            
        Raises:
            ValueError: If workflow not found
        """
        graph = self.registry.get(workflow_id)
        if graph is None:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Get task information
        tasks = []
        for task_id, task in graph.tasks.items():
            task_info = {
                "id": task_id,
                "name": task.name,
                "type": type(task).__name__,
                "dependencies": list(task.dependencies.keys()) if hasattr(task, "dependencies") else [],
            }
            tasks.append(task_info)

        return {
            "workflow_id": workflow_id,
            "num_tasks": len(graph.tasks),
            "tasks": tasks,
        }

    def render_tree(self, workflow_id: str) -> str:
        """Render workflow as ASCII tree.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            ASCII tree representation
            
        Raises:
            ValueError: If workflow not found
        """
        graph = self.registry.get(workflow_id)
        if graph is None:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Build dependency graph
        dependencies: dict[str, list[str]] = {}
        dependents: dict[str, list[str]] = {}
        
        for task_id, task in graph.tasks.items():
            deps = []
            if hasattr(task, "dependencies"):
                deps = list(task.dependencies.keys())
            dependencies[task_id] = deps
            
            for dep in deps:
                if dep not in dependents:
                    dependents[dep] = []
                dependents[dep].append(task_id)

        # Find root tasks (no dependencies)
        roots = [tid for tid, deps in dependencies.items() if not deps]

        # Build tree
        lines = [f"Workflow: {workflow_id}"]
        lines.append("")
        
        visited = set()
        
        def render_node(task_id: str, prefix: str = "", is_last: bool = True) -> None:
            """Render a task node and its dependents."""
            if task_id in visited:
                return
            visited.add(task_id)
            
            # Draw current node
            connector = "└── " if is_last else "├── "
            task = graph.tasks[task_id]
            task_type = type(task).__name__
            lines.append(f"{prefix}{connector}{task.name} ({task_type})")
            
            # Draw dependents
            deps = dependents.get(task_id, [])
            if deps:
                extension = "    " if is_last else "│   "
                for i, dep_id in enumerate(deps):
                    is_last_dep = i == len(deps) - 1
                    render_node(dep_id, prefix + extension, is_last_dep)

        # Render from roots
        for i, root_id in enumerate(roots):
            is_last_root = i == len(roots) - 1
            render_node(root_id, "", is_last_root)

        return "\n".join(lines)

    def export_json_ir(self, workflow_id: str) -> dict[str, Any]:
        """Export workflow to JSON IR format.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            JSON IR representation
            
        Raises:
            ValueError: If workflow not found
        """
        graph = self.registry.get(workflow_id)
        if graph is None:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Use the graph's to_json method if available
        if hasattr(graph, "to_json"):
            return graph.to_json()

        # Fallback: build basic IR
        tasks_ir = []
        for task_id, task in graph.tasks.items():
            task_ir = {
                "id": task_id,
                "name": task.name,
                "type": type(task).__name__,
            }
            
            # Add dependencies if present
            if hasattr(task, "dependencies"):
                task_ir["dependencies"] = list(task.dependencies.keys())
            
            # Add config if present
            if hasattr(task, "cfg") and task.cfg is not None:
                if hasattr(task.cfg, "model_dump"):
                    task_ir["config"] = task.cfg.model_dump()
                else:
                    task_ir["config"] = task.cfg

            tasks_ir.append(task_ir)

        return {
            "workflow_id": workflow_id,
            "version": "1.0",
            "tasks": tasks_ir,
        }

    def list_workflows(self) -> list[dict[str, Any]]:
        """List all registered workflows with basic info.
        
        Returns:
            List of workflow information dictionaries
        """
        workflows = []
        for workflow_id in self.registry.list():
            try:
                info = self.get_workflow_info(workflow_id)
                workflows.append({
                    "id": workflow_id,
                    "num_tasks": info["num_tasks"],
                })
            except Exception:
                # Skip workflows that fail to load
                workflows.append({
                    "id": workflow_id,
                    "num_tasks": 0,
                    "error": "Failed to load",
                })
        
        return workflows
