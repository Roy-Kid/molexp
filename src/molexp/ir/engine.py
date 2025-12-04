import logging
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from typing import Any, Dict, List

from .models import WorkflowIR, Node, EdgeType
from .registry import registry

logger = logging.getLogger(__name__)

class ExecutionContext:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.artifacts: Dict[str, Any] = {}  # node_id -> output
        self.status: Dict[str, str] = {}     # node_id -> status

class WorkflowEngine:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute(self, workflow: WorkflowIR, run_id: str, node_ids: List[str] | None = None) -> Dict[str, str]:
        """
        Execute a workflow definition.
        
        Args:
            workflow: The IR model.
            run_id: Unique ID for this run.
            node_ids: Optional list of node IDs to execute (if None, execute all).
            
        Returns:
            Dict mapping node IDs to their final status.
        """
        ctx = ExecutionContext(run_id)
        wf = workflow.workflow
        
        # Filter nodes if specific IDs provided
        nodes_to_run = wf.nodes
        if node_ids is not None:
            nodes_to_run = [n for n in wf.nodes if n.id in node_ids]
            
        # Build dependency graph for the subset
        adj: Dict[str, List[str]] = {n.id: [] for n in nodes_to_run}
        in_degree: Dict[str, int] = {n.id: 0 for n in nodes_to_run}
        node_map = {n.id: n for n in nodes_to_run}
        
        # Pre-process edges
        data_edges = [e for e in wf.edges if e.type == EdgeType.DATA]
        dep_edges = [e for e in wf.edges if e.type == EdgeType.DEPENDENCY]
        
        # Both types of edges contribute to execution order
        all_edges = wf.edges
        
        for edge in all_edges:
            # Only consider edges where both source and target are in the execution set
            if edge.source in adj and edge.target in in_degree:
                adj[edge.source].append(edge.target)
                in_degree[edge.target] += 1
            # else: ignore edges connected to excluded nodes

        # Initial queue
        queue = [n_id for n_id, d in in_degree.items() if d == 0]
        active_futures: Dict[Future, str] = {}
        
        logger.info(f"Starting workflow {wf.id} (Run {run_id})")
        
        while queue or active_futures:
            # Submit ready tasks
            while queue:
                node_id = queue.pop(0)
                node = node_map[node_id]
                ctx.status[node_id] = "RUNNING"
                
                # Resolve inputs
                inputs = self._resolve_inputs(node, data_edges, ctx)
                
                future = self.executor.submit(self._run_node, node, inputs)
                active_futures[future] = node_id
            
            # Wait for any completion
            if not active_futures:
                break
                
            done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            
            for future in done:
                node_id = active_futures.pop(future)
                try:
                    result = future.result()
                    ctx.artifacts[node_id] = result
                    ctx.status[node_id] = "SUCCEEDED"
                    logger.info(f"Node {node_id} succeeded")
                    
                    # Unlock dependents
                    for neighbor in adj[node_id]:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
                            
                except Exception as e:
                    logger.error(f"Node {node_id} failed: {e}")
                    ctx.status[node_id] = "FAILED"
                    # TODO: Handle failure propagation (cancel dependents)
                    
        return ctx.status

    def _resolve_inputs(self, node: Node, data_edges: List, ctx: ExecutionContext) -> List[Any]:
        """Collect outputs from upstream nodes connected via DATA edges."""
        inputs = []
        # Find edges targeting this node
        relevant_edges = [e for e in data_edges if e.target == node.id]
        
        for edge in relevant_edges:
            val = ctx.artifacts.get(edge.source)
            inputs.append(val)
            
        return inputs

    def _run_node(self, node: Node, inputs: List[Any]) -> Any:
        op_def = registry.get_operation(node.op)
        
        # Validate config
        # Pydantic model instantiation validates the args
        config = op_def.schema_model(**node.args)
        
        # Instantiate the task
        # We assume the handler is a Task class that takes *upstreams in init
        # or it's a function.
        # Based on existing codebase, Task classes take *upstreams in __init__
        
        if isinstance(op_def.handler, type):
            # It's a class
            # We instantiate the task. Upstreams in init are for graph building, 
            # but here we are executing with concrete values.
            task_instance = op_def.handler(name=node.id)
            return task_instance(*inputs, cfg=config)
        else:
            # It's a function
            return op_def.handler(*inputs, cfg=config)
