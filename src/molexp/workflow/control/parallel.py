"""Parallel map node: Concurrent execution."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, List, Iterable
from pydantic import BaseModel, Field

from ..node import Node
from ..registry import register


class ParallelMapConfig(BaseModel):
    """Configuration for ParallelMapNode.
    
    Attributes
    ----------
    max_workers : int | None
        Maximum number of workers (None = CPU count)
    use_processes : bool
        If True, use processes (CPU-bound); if False, use threads (I/O-bound)
    """
    
    max_workers: int | None = Field(
        None,
        description="Maximum number of workers (None = CPU count)"
    )
    use_processes: bool = Field(
        False,
        description="Use processes (True) or threads (False)"
    )


@register("control.parallel_map")
class ParallelMapNode(Node[ParallelMapConfig, List[Any]]):
    """Apply a node to each element in parallel.
    
    Uses concurrent.futures for parallel execution.
    
    Examples
    --------
    >>> expensive_task = SimulationNode()
    >>> parallel_node = ParallelMapNode(collection, base_task=expensive_task)
    >>> results = parallel_node(max_workers=4, use_processes=True)
    """
    
    config_type = ParallelMapConfig
    
    def __init__(
        self,
        collection: Any,
        base_task: Node,
        id: str | None = None,
    ):
        """Initialize parallel map node.
        
        Parameters
        ----------
        collection : Any
            Upstream node or constant providing the collection
        base_task : Node
            Node to apply to each element
        id : str | None
            Unique identifier
        """
        super().__init__(collection, id=id or f"{base_task.id}__parallel_map")
        self.base_task = base_task
    
    def execute(self, collection: Iterable[Any], config: ParallelMapConfig) -> List[Any]:
        """Apply base_task to each element in parallel.
        
        Parameters
        ----------
        collection : Iterable[Any]
            Collection to iterate over
        config : ParallelMapConfig
            Configuration for parallel execution
            
        Returns
        -------
        List[Any]
            List of results
        """
        items = list(collection)
        
        if not items:
            return []
        
        # Choose executor type
        executor_class = ProcessPoolExecutor if config.use_processes else ThreadPoolExecutor
        
        # Define worker function
        def process_item(item: Any) -> Any:
            if isinstance(item, tuple):
                return self.base_task(*item)
            else:
                return self.base_task(item)
        
        # Execute in parallel
        with executor_class(max_workers=config.max_workers) as executor:
            results = list(executor.map(process_item, items))
        
        return results
