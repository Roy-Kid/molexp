"""Workflow compiler and validator."""

from typing import List, Set

from .models import WorkflowIR, Node

class ValidationError(Exception):
    """Workflow validation error."""
    pass

def validate_workflow(workflow: WorkflowIR) -> None:
    """
    Validate a workflow IR.
    
    Checks:
    1. Presence of at least one Target node.
    2. No cycles in the dependency graph.
    
    Args:
        workflow: The workflow to validate.
        
    Raises:
        ValidationError: If validation fails.
    """
    if not workflow.workflow.targets:
        raise ValidationError("Workflow must have at least one Target node.")
        
    # Cycle detection is implicitly handled by the topological sort in plan_execution
    try:
        plan_execution(workflow)
    except RecursionError:
        raise ValidationError("Workflow contains a cycle.")
    except Exception as e:
        raise ValidationError(f"Validation failed: {e}")

def plan_execution(workflow: WorkflowIR, targets: List[str] | None = None) -> List[str]:
    """
    Plan execution by finding the subgraph required for the targets.
    
    Args:
        workflow: The workflow IR.
        targets: Optional list of target node IDs. If None, uses workflow.targets.
        
    Returns:
        List of node IDs in topological order (execution order).
    """
    effective_targets = targets or workflow.workflow.targets
    if not effective_targets:
        return []
        
    # Build adjacency list (reverse graph: target -> source) to find dependencies
    # and forward graph for topological sort
    node_map = {n.id: n for n in workflow.workflow.nodes}
    
    # Check if targets exist
    for t in effective_targets:
        if t not in node_map:
            raise ValidationError(f"Target node '{t}' not found in workflow.")
            
    # Build dependency graph
    # upstream_deps[u] = [v, w] means u depends on v and w
    upstream_deps: dict[str, set[str]] = {n.id: set() for n in workflow.workflow.nodes}
    for edge in workflow.workflow.edges:
        if edge.target in upstream_deps and edge.source in node_map:
            upstream_deps[edge.target].add(edge.source)
            
    # 1. Identify all required nodes (transitive closure of dependencies from targets)
    required_nodes: Set[str] = set()
    stack = list(effective_targets)
    visited = set()
    
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        required_nodes.add(node_id)
        
        # Add dependencies to stack
        for dep_id in upstream_deps.get(node_id, []):
            if dep_id not in visited:
                stack.append(dep_id)
                
    # 2. Topological sort of the required subgraph
    # Calculate in-degree for the subgraph
    in_degree = {n: 0 for n in required_nodes}
    subgraph_adj: dict[str, list[str]] = {n: [] for n in required_nodes}
    
    for u in required_nodes:
        # u depends on v, so v -> u is the execution direction
        for v in upstream_deps[u]:
            if v in required_nodes:
                subgraph_adj[v].append(u)
                in_degree[u] += 1
                
    # Kahn's algorithm
    queue = [n for n in required_nodes if in_degree[n] == 0]
    execution_order = []
    
    while queue:
        u = queue.pop(0)
        execution_order.append(u)
        
        for v in subgraph_adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(execution_order) != len(required_nodes):
        raise ValidationError("Workflow contains a cycle (circular dependency).")
        
    return execution_order

def compile_workflow(workflow: WorkflowIR) -> WorkflowIR:
    """
    Compile and validate a workflow.
    
    Args:
        workflow: The raw workflow IR.
        
    Returns:
        The compiled (and validated) workflow.
    """
    validate_workflow(workflow)
    return workflow
