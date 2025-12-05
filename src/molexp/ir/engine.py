"""Workflow execution engine with failure propagation and timeout support."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import WorkflowIR, Node, EdgeType
from .registry import registry

logger = logging.getLogger(__name__)


# ============================================================================
# Execution Status
# ============================================================================


class ExecutionStatus:
    """Node execution status constants."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


# ============================================================================
# Execution Hooks
# ============================================================================


@dataclass
class ExecutionHooks:
    """Callbacks for monitoring workflow execution.
    
    These hooks are called at various points during execution to allow
    for monitoring, logging, and custom behavior.
    """
    
    on_node_start: Callable[[str, Node], None] | None = None
    on_node_success: Callable[[str, Node, Any], None] | None = None
    on_node_failure: Callable[[str, Node, Exception], None] | None = None
    on_node_cancelled: Callable[[str, Node], None] | None = None
    on_workflow_start: Callable[[str, WorkflowIR], None] | None = None
    on_workflow_complete: Callable[[str, dict[str, str]], None] | None = None


# ============================================================================
# Execution Context
# ============================================================================


@dataclass
class ExecutionContext:
    """Context for tracking workflow execution state.
    
    Attributes:
        run_id: Unique identifier for this execution run
        artifacts: Mapping of node_id to execution output
        status: Mapping of node_id to execution status
        errors: Mapping of node_id to error information
    """
    
    run_id: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    
    def mark_succeeded(self, node_id: str, result: Any) -> None:
        """Mark a node as successfully completed."""
        self.status[node_id] = ExecutionStatus.SUCCEEDED
        self.artifacts[node_id] = result
    
    def mark_failed(self, node_id: str, error: Exception) -> None:
        """Mark a node as failed."""
        self.status[node_id] = ExecutionStatus.FAILED
        self.errors[node_id] = str(error)
    
    def mark_cancelled(self, node_id: str) -> None:
        """Mark a node as cancelled due to upstream failure."""
        self.status[node_id] = ExecutionStatus.CANCELLED
    
    def is_failed_or_cancelled(self, node_id: str) -> bool:
        """Check if a node has failed or been cancelled."""
        return self.status.get(node_id) in (
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        )


# ============================================================================
# Workflow Engine
# ============================================================================


class WorkflowEngine:
    """Executes workflow graphs with parallel execution and failure propagation.
    
    The engine supports:
    - Parallel execution of independent nodes
    - Failure propagation to dependent nodes
    - Execution hooks for monitoring
    - Configurable timeouts
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        node_timeout: float | None = None,
        hooks: ExecutionHooks | None = None,
    ) -> None:
        """Initialize the workflow engine.
        
        Args:
            max_workers: Maximum number of concurrent node executions
            node_timeout: Timeout in seconds for each node (None = no timeout)
            hooks: Optional execution hooks for monitoring
        """
        self.max_workers = max_workers
        self.node_timeout = node_timeout
        self.hooks = hooks or ExecutionHooks()
        self._executor: ThreadPoolExecutor | None = None
    
    def execute(
        self,
        workflow: WorkflowIR,
        run_id: str,
        node_ids: list[str] | None = None,
    ) -> dict[str, str]:
        """Execute a workflow definition.
        
        Args:
            workflow: The workflow IR model
            run_id: Unique ID for this run
            node_ids: Optional list of node IDs to execute (if None, executes all)
            
        Returns:
            Dict mapping node IDs to their final status
        """
        ctx = ExecutionContext(run_id)
        wf = workflow.workflow
        
        # Call workflow start hook
        if self.hooks.on_workflow_start:
            self.hooks.on_workflow_start(run_id, workflow)
        
        # Filter nodes if specific IDs provided
        nodes_to_run = wf.nodes
        if node_ids is not None:
            nodes_to_run = [n for n in wf.nodes if n.id in node_ids]
        
        # Build dependency graph
        adj, in_degree, node_map = self._build_dependency_graph(nodes_to_run, wf.edges)
        
        # Pre-process data edges for input resolution
        data_edges = [e for e in wf.edges if e.type == EdgeType.DATA]
        
        # Initialize execution
        queue = [n_id for n_id, d in in_degree.items() if d == 0]
        active_futures: dict[Future, str] = {}
        
        logger.info(f"Starting workflow {wf.id} (Run {run_id})")
        
        # Create executor for this execution
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        try:
            while queue or active_futures:
                # Submit ready tasks (that aren't cancelled)
                while queue:
                    node_id = queue.pop(0)
                    
                    # Skip if already cancelled due to upstream failure
                    if ctx.is_failed_or_cancelled(node_id):
                        continue
                    
                    node = node_map[node_id]
                    ctx.status[node_id] = ExecutionStatus.RUNNING
                    
                    # Call node start hook
                    if self.hooks.on_node_start:
                        self.hooks.on_node_start(run_id, node)
                    
                    # Resolve inputs
                    inputs = self._resolve_inputs(node, data_edges, ctx)
                    
                    # Submit for execution
                    future = self._executor.submit(self._run_node, node, inputs)
                    active_futures[future] = node_id
                
                # Wait for any completion
                if not active_futures:
                    break
                
                done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                
                for future in done:
                    node_id = active_futures.pop(future)
                    node = node_map[node_id]
                    
                    try:
                        # Get result with timeout
                        result = future.result(timeout=self.node_timeout)
                        ctx.mark_succeeded(node_id, result)
                        logger.info(f"Node {node_id} succeeded")
                        
                        # Call success hook
                        if self.hooks.on_node_success:
                            self.hooks.on_node_success(run_id, node, result)
                        
                        # Unlock dependents
                        for neighbor in adj[node_id]:
                            in_degree[neighbor] -= 1
                            if in_degree[neighbor] == 0:
                                queue.append(neighbor)
                                
                    except TimeoutError:
                        error = TimeoutError(f"Node execution timed out after {self.node_timeout}s")
                        self._handle_node_failure(ctx, node_id, node, adj, error, run_id)
                        
                    except Exception as e:
                        self._handle_node_failure(ctx, node_id, node, adj, e, run_id)
            
            # Call workflow complete hook
            if self.hooks.on_workflow_complete:
                self.hooks.on_workflow_complete(run_id, ctx.status)
            
            return ctx.status
            
        finally:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def _build_dependency_graph(
        self,
        nodes: list[Node],
        edges: list,
    ) -> tuple[dict[str, list[str]], dict[str, int], dict[str, Node]]:
        """Build dependency graph for execution ordering.
        
        Returns:
            Tuple of (adjacency list, in-degree map, node map)
        """
        adj: dict[str, list[str]] = {n.id: [] for n in nodes}
        in_degree: dict[str, int] = {n.id: 0 for n in nodes}
        node_map = {n.id: n for n in nodes}
        
        for edge in edges:
            if edge.source in adj and edge.target in in_degree:
                adj[edge.source].append(edge.target)
                in_degree[edge.target] += 1
        
        return adj, in_degree, node_map
    
    def _handle_node_failure(
        self,
        ctx: ExecutionContext,
        node_id: str,
        node: Node,
        adj: dict[str, list[str]],
        error: Exception,
        run_id: str,
    ) -> set[str]:
        """Handle a node failure by marking it and propagating to dependents.
        
        Returns:
            Set of all cancelled node IDs
        """
        logger.error(f"Node {node_id} failed: {error}")
        ctx.mark_failed(node_id, error)
        
        # Call failure hook
        if self.hooks.on_node_failure:
            self.hooks.on_node_failure(run_id, node, error)
        
        # Propagate failure to all dependents
        cancelled = self._propagate_failure(ctx, node_id, adj, run_id)
        
        return cancelled
    
    def _propagate_failure(
        self,
        ctx: ExecutionContext,
        failed_node: str,
        adj: dict[str, list[str]],
        run_id: str,
    ) -> set[str]:
        """Cancel all nodes that depend on a failed node.
        
        Uses BFS to find and cancel all transitive dependents.
        
        Returns:
            Set of cancelled node IDs
        """
        cancelled: set[str] = set()
        queue = list(adj.get(failed_node, []))
        
        while queue:
            node_id = queue.pop(0)
            
            if node_id in cancelled:
                continue
            
            # Only cancel if not already completed
            current_status = ctx.status.get(node_id)
            if current_status not in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED):
                cancelled.add(node_id)
                ctx.mark_cancelled(node_id)
                logger.warning(f"Node {node_id} cancelled due to upstream failure")
                
                # Call cancelled hook
                if self.hooks.on_node_cancelled:
                    # We don't have the node object here, but hooks can handle None
                    self.hooks.on_node_cancelled(run_id, None)  # type: ignore
                
                # Add dependents to queue
                queue.extend(adj.get(node_id, []))
        
        return cancelled
    
    def _resolve_inputs(
        self,
        node: Node,
        data_edges: list,
        ctx: ExecutionContext,
    ) -> list[Any]:
        """Collect outputs from upstream nodes connected via DATA edges."""
        inputs = []
        relevant_edges = [e for e in data_edges if e.target == node.id]
        
        for edge in relevant_edges:
            val = ctx.artifacts.get(edge.source)
            inputs.append(val)
        
        return inputs
    
    def _run_node(self, node: Node, inputs: list[Any]) -> Any:
        """Execute a single node with its inputs."""
        op_def = registry.get_operation(node.op)
        
        # Validate config via Pydantic model instantiation
        config = op_def.schema_model(**node.args)
        
        # Execute the node
        if isinstance(op_def.handler, type):
            # It's a class - instantiate and call
            task_instance = op_def.handler(name=node.id)
            return task_instance(*inputs, cfg=config)
        else:
            # It's a function - call directly
            return op_def.handler(*inputs, cfg=config)
