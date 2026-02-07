"""Core compiler classes for workflow compilation and type-driven dispatch.

This module implements workflow compilation with automatic task type detection and
channel allocation for actor-based message passing:

1. **WorkflowCompiler**: Main entry point for workflow compilation. Validates workflow
   structure, detects task execution types via return type inspection, and allocates
   message channels for actor communication.

2. **CompiledWorkflow**: Encapsulates compiled workflow with high-level access methods
   for topology, execution plans, task types, and channel configurations.

Type Detection:
    The compiler inspects execute() method return annotations to classify tasks:
    - `-> dict`: BATCH task (executes once)
    - `-> AsyncGenerator[None, dict]`: ACTOR task (runs continuously)

Channel Allocation:
    For links involving actors, the compiler allocates asyncio.Queue configurations
    with configurable buffer sizes. Channel routing uses Link.mapping for logical
    channel name resolution.

Example:
    Basic compilation::

        compiler = WorkflowCompiler()
        compiled = compiler.compile(workflow)
        task_types = compiled.get_task_types()
        channels = compiled.get_channels()

    Accessing execution plan::

        plan = compiled.get_execution_plan()
        topology = compiled.get_topology()
"""

import inspect
from collections.abc import AsyncGenerator
from typing import Any, List, get_origin

from ..execution_type import TaskExecutionType
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

    def __init__(self, workflow: Workflow, task_types: dict[str, TaskExecutionType] | None = None, channels: dict[str, Any] | None = None):
        """Initialize with validated Workflow.

        Args:
            workflow: Validated workflow
            task_types: Optional dict of task_id -> TaskExecutionType
            channels: Optional dict of channel configurations
        """
        self._workflow = workflow
        self._task_types = task_types or {}
        self._channels = channels or {}

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

    def get_task_types(self) -> dict[str, TaskExecutionType]:
        """Get task execution types.

        Returns:
            Dict mapping task_id to TaskExecutionType
        """
        return self._task_types

    def get_channels(self) -> dict[str, Any]:
        """Get channel configurations.

        Returns:
            Dict of channel configurations
        """
        return self._channels

    def has_actors(self) -> bool:
        """Check if workflow contains any Actor tasks.

        Returns:
            True if workflow has at least one Actor task
        """
        return TaskExecutionType.ACTOR in self._task_types.values()



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
        high-level access methods. Detects Actor tasks via return type
        inspection and allocates message channels.

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
        # Get task instances for type detection
        task_map = {task.task_id: task for task in workflow.get_tasks()}

        # Detect task types via return type inspection
        task_types = {}
        for task_id, task in task_map.items():
            task_types[task_id] = self._detect_task_type(task)

        # Basic validation (this auto-generates Link mappings)
        self._validate_workflow(workflow)

        # Allocate channels for actors (after validation populates mappings)
        channels = self._allocate_channels(workflow, task_types)

        # Return compiled workflow with type info
        return CompiledWorkflow(workflow, task_types, channels)
    
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

            # Check if either end is an Actor (Actors use dynamic channels)
            from ..task import Actor
            source_is_actor = isinstance(source_task, Actor)
            target_is_actor = isinstance(target_task, Actor)

            # For Actors, inputs/outputs are optional (channels are dynamic)
            if not source_is_actor and not source_task.outputs:
                raise ValueError(
                    f"Task '{link.source}' must declare outputs for link validation"
                )
            if not target_is_actor and not target_task.inputs:
                raise ValueError(
                    f"Task '{link.target}' must declare inputs for link validation"
                )

            # Auto-generate mapping if not provided
            if link.mapping is None:
                if source_is_actor or target_is_actor:
                    # For Actors, use a default channel name based on link
                    # Format: use last part of source and target IDs
                    source_name = link.source.split('_')[0].lower()
                    target_name = link.target.split('_')[0].lower()
                    channel_name = f"{source_name}_to_{target_name}"
                    link.mapping = {channel_name: channel_name}
                else:
                    # For regular tasks, use input/output intersection
                    auto_mapping = {}
                    for output_name in source_task.outputs.keys():
                        if output_name in target_task.inputs:
                            auto_mapping[output_name] = output_name
                    link.mapping = auto_mapping

            # Validate mapping keys against declared inputs/outputs (only for non-Actors)
            if link.mapping:
                for output_name, input_name in link.mapping.items():
                    if not source_is_actor and source_task.outputs and output_name not in source_task.outputs:
                        raise ValueError(
                            f"Link mapping references unknown output '{output_name}' "
                            f"on task '{link.source}'"
                        )
                    if not target_is_actor and target_task.inputs and input_name not in target_task.inputs:
                        raise ValueError(
                            f"Link mapping references unknown input '{input_name}' "
                            f"on task '{link.target}'"
                        )
        
        # Check for cycles (simple DFS-based cycle detection)
        # Note: Actors can have feedback loops, so only check for cycles in pure batch workflows
        from ..task import Actor
        has_any_actor = any(isinstance(task_map[tid], Actor) for tid in task_ids)

        if not has_any_actor:
            # Pure batch workflow - cycles not allowed
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
                        raise ValueError("Workflow contains cycles (not allowed for batch-only workflows)")

    def _detect_task_type(self, task: Any) -> TaskExecutionType:
        """Detect task execution type via return type annotation.

        Args:
            task: Task instance

        Returns:
            TaskExecutionType (BATCH or ACTOR)

        Raises:
            ValueError: If Actor missing AsyncGenerator return annotation
        """
        # Get execute method signature
        sig = inspect.signature(task.execute)
        return_annotation = sig.return_annotation

        # Check if return type is AsyncGenerator
        if return_annotation != inspect.Signature.empty:
            origin = get_origin(return_annotation)
            if origin is AsyncGenerator:
                return TaskExecutionType.ACTOR

        # Check if task is instance of Actor class
        from ..task import Actor
        if isinstance(task, Actor):
            # Actor without proper return annotation - raise error
            if return_annotation == inspect.Signature.empty or get_origin(return_annotation) is not AsyncGenerator:
                raise ValueError(
                    f"Actor task '{task.task_id}' must annotate execute() return type as "
                    f"AsyncGenerator[None, dict[str, Any]]. Current annotation: {return_annotation}"
                )

        return TaskExecutionType.BATCH

    def _allocate_channels(self, workflow: Workflow, task_types: dict[str, TaskExecutionType]) -> dict[str, Any]:
        """Allocate asyncio.Queue channels for actor links.

        Args:
            workflow: Workflow being compiled
            task_types: Dict of task_id -> TaskExecutionType

        Returns:
            Dict of channel configurations
        """
        channels = {}

        for link in workflow.links:
            source_type = task_types.get(link.source, TaskExecutionType.BATCH)
            target_type = task_types.get(link.target, TaskExecutionType.BATCH)

            # Only create channels if at least one side is an Actor
            if source_type == TaskExecutionType.ACTOR or target_type == TaskExecutionType.ACTOR:
                channel_id = f"{link.source}_to_{link.target}"
                channels[channel_id] = {
                    'source': link.source,
                    'target': link.target,
                    'buffer_size': link.buffer_size,
                    'channel_mapping': link.channels or {},
                    'mapping': link.mapping or {},  # Include Link mapping for channel name routing
                }

        return channels
