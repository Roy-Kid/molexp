"""Core compiler classes for workflow compilation.

This module contains the user-facing WorkflowCompiler and CompiledWorkflow classes.
"""

from typing import Any, List

from ..workflow import Workflow


class CompiledWorkflow:
    """Encapsulates a compiled workflow with high-level access methods.
    
    This class provides a user-friendly interface to compiled workflows.
    Users interact with high-level methods like get_topology() and 
    get_execution_plan().
    
    Example:
        >>> compiler = WorkflowCompiler()
        >>> compiled = compiler.compile(workflow)
        >>> topology = compiled.get_topology()
        >>> plan = compiled.get_execution_plan()
    """

    def __init__(self, workflow: Workflow):
        """Initialize with validated Workflow.
        
        Args:
            workflow: Validated workflow
        """
        self._workflow = workflow

    def get_workflow(self) -> Workflow:
        """Get underlying Workflow for debugging or advanced use.
        
        Returns:
            The underlying Workflow object
        """
        return self._workflow

    def get_topology(self) -> dict[str, Any]:
        """Get execution topology as a dictionary.
        
        Returns a dictionary containing nodes, edges, and metadata describing
        the workflow structure in a user-friendly format.
        
        Returns:
            Dictionary with 'nodes', 'edges', and 'metadata' keys
            
        Example:
            >>> topology = compiled.get_topology()
            >>> print(topology['nodes'])  # List of task nodes
            >>> print(topology['edges'])  # List of dependency edges
        """
        # Build nodes from task_configs
        nodes = [
            {
                'id': tc.task_id,
                'type': tc.task_type,
                'config': tc.config,
            }
            for tc in self._workflow.task_configs
        ]
        
        # Build edges from links
        edges = [
            {
                'source': link.source,
                'target': link.target,
            }
            for link in self._workflow.links
        ]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'id': self._workflow.workflow_id,
                'name': self._workflow.name,
            }
        }

    def get_execution_plan(self, targets: List[str] | None = None) -> List[str]:
        """Get execution order for specified targets.
        
        Returns a list of task IDs in topological order (execution order).
        
        Args:
            targets: Optional list of target task IDs to execute.
                    If None, returns execution order for all tasks.
        
        Returns:
            List of task IDs in execution order
            
        Example:
            >>> plan = compiled.get_execution_plan()
            >>> print(plan)  # ['task1', 'task2', 'task3']
        """
        # Build adjacency list and in-degree map
        adj, in_degree = self.get_dependency_graph()
        
        # If no targets specified, use all tasks
        if targets is None:
            targets = list(in_degree.keys())
        else:
            missing = [task_id for task_id in targets if task_id not in in_degree]
            if missing:
                missing_list = ", ".join(missing)
                raise ValueError(f"Unknown target task IDs: {missing_list}")
        
        # Topological sort using Kahn's algorithm
        from collections import deque
        
        # Find all tasks needed for targets (reverse topological traversal)
        needed = set()
        queue = deque(targets)
        while queue:
            task_id = queue.popleft()
            if task_id in needed:
                continue
            needed.add(task_id)
            # Add all dependencies
            for node_id, neighbors in adj.items():
                if task_id in neighbors:
                    queue.append(node_id)
        
        # Topological sort on needed tasks
        in_deg = {k: v for k, v in in_degree.items() if k in needed}
        queue = deque([k for k in needed if in_deg[k] == 0])
        result = []
        
        while queue:
            task_id = queue.popleft()
            result.append(task_id)
            
            for neighbor in adj.get(task_id, []):
                if neighbor in needed:
                    in_deg[neighbor] -= 1
                    if in_deg[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    def get_dependency_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Get dependency graph (adjacency list and in-degree map).
        
        Returns:
            Tuple of (adjacency_list, in_degree_map)
        """
        topology = self.get_topology()
        
        adj: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}
        
        for node in topology['nodes']:
            node_id = node['id']
            adj[node_id] = []
            in_degree[node_id] = 0
        
        for edge in topology['edges']:
            source = edge['source']
            target = edge['target']
            adj[source].append(target)
            in_degree[target] += 1
        
        return adj, in_degree



class WorkflowCompiler:
    """Compiles and validates workflows with a clean OOP API.
    
    This class provides the main entry point for workflow compilation.
    It accepts Workflow objects and returns a CompiledWorkflow with 
    high-level access methods.
    
    Example:
        >>> compiler = WorkflowCompiler()
        >>> compiled = compiler.compile(workflow)
        >>> topology = compiled.get_topology()
        >>> plan = compiled.get_execution_plan()
    """

    def __init__(self, optimize: bool = True):
        """Initialize the compiler.
        
        Args:
            optimize: Whether to apply optimization passes (reserved for future use)
        """
        self.optimize = optimize

    def compile(self, workflow: Workflow) -> CompiledWorkflow:
        """Compile a workflow.
        
        Validates the workflow and returns a CompiledWorkflow with 
        high-level access methods.
        
        Args:
            workflow: Workflow to compile
            
        Returns:
            CompiledWorkflow with high-level API
            
        Raises:
            ValueError: If workflow validation fails
            
        Example:
            >>> workflow = Workflow.from_tasks(tasks, links)
            >>> compiled = compiler.compile(workflow)
        """
        # Basic validation
        self._validate_workflow(workflow)

        # Return compiled workflow
        return CompiledWorkflow(workflow)
    
    def _validate_workflow(self, workflow: Workflow) -> None:
        """Validate workflow structure.
        
        Args:
            workflow: Workflow to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check for duplicate task IDs
        task_ids = [tc.task_id for tc in workflow.task_configs]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs found in workflow")
        
        # Check that all links reference valid tasks
        task_id_set = set(task_ids)
        for link in workflow.links:
            if link.source not in task_id_set:
                raise ValueError(f"Link references unknown source task: {link.source}")
            if link.target not in task_id_set:
                raise ValueError(f"Link references unknown target task: {link.target}")

        # Load tasks for validation and auto-mapping
        task_map = {task.task_id: task for task in workflow.get_tasks()}

        # Auto-generate mappings and validate
        for link in workflow.links:
            source_task = task_map.get(link.source)
            target_task = task_map.get(link.target)
            if source_task is None or target_task is None:
                raise ValueError(
                    f"Link references missing task instances: {link.source} -> {link.target}"
                )
            if not source_task.outputs:
                raise ValueError(
                    f"Task '{link.source}' must declare outputs for link validation"
                )
            if not target_task.inputs:
                raise ValueError(
                    f"Task '{link.target}' must declare inputs for link validation"
                )

            # Auto-generate mapping if not provided
            if link.mapping is None:
                auto_mapping = {}
                for output_name in source_task.outputs.keys():
                    if output_name in target_task.inputs:
                        auto_mapping[output_name] = output_name
                link.mapping = auto_mapping

            # Validate mapping keys against declared inputs/outputs
            for output_name, input_name in link.mapping.items():
                if output_name not in source_task.outputs:
                    raise ValueError(
                        f"Link mapping references unknown output '{output_name}' "
                        f"on task '{link.source}'"
                    )
                if input_name not in target_task.inputs:
                    raise ValueError(
                        f"Link mapping references unknown input '{input_name}' "
                        f"on task '{link.target}'"
                    )
        
        # Check for cycles (simple DFS-based cycle detection)
        adj = {tid: [] for tid in task_ids}
        for link in workflow.links:
            adj[link.source].append(link.target)
        
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_id in task_ids:
            if task_id not in visited:
                if has_cycle(task_id):
                    raise ValueError("Workflow contains cycles")
