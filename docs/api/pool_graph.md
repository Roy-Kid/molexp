# Task Management - Pool and Graph

This section provides API documentation for task pool and graph management components.

## TaskPool

The TaskPool manages collections of tasks for batch execution.

::: molexp.pool.TaskPool
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - add_task
        - remove_task
        - get_task
        - list_tasks
        - execute_all
        - execute_batch
        - get_status
        - clear

### TaskPool Usage Examples

```python
from molexp import TaskPool, LocalTask

# Create task pool
pool = TaskPool(max_size=100)

# Add tasks
task1 = LocalTask("task1", func=lambda x: x*2, inputs={'x': 5})
task2 = LocalTask("task2", func=lambda x: x+10, inputs={'x': 3})

pool.add_task(task1)
pool.add_task(task2)

# Execute all tasks
results = pool.execute_all()

# Execute specific batch
batch_results = pool.execute_batch(['task1'])

# Get pool status
status = pool.get_status()
print(f"Total tasks: {status['total_tasks']}")
print(f"Completed: {status['completed']}")
print(f"Failed: {status['failed']}")
```

### Batch Processing

```python
# Process tasks in batches
batch_size = 10
total_tasks = len(pool)

for i in range(0, total_tasks, batch_size):
    batch_tasks = list(pool.list_tasks())[i:i+batch_size]
    batch_results = pool.execute_batch(batch_tasks)
    
    # Process batch results
    for task_name, result in batch_results.items():
        if isinstance(result, Exception):
            print(f"Task {task_name} failed: {result}")
        else:
            print(f"Task {task_name} completed: {result}")
```

## TaskGraph

The TaskGraph manages task dependencies and execution ordering.

::: molexp.graph.TaskGraph
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - add_task
        - remove_task
        - add_dependency
        - remove_dependency
        - get_dependencies  
        - get_dependents
        - topological_sort
        - has_cycles
        - execute
        - visualize

### TaskGraph Usage Examples

```python
from molexp import TaskGraph, LocalTask

# Create task graph
graph = TaskGraph()

# Create tasks
def prepare_data():
    return [1, 2, 3, 4, 5]

def process_data(data):
    return [x * 2 for x in data]

def analyze_data(processed_data):
    return sum(processed_data)

prep_task = LocalTask("prepare", func=prepare_data, outputs=['data'])
proc_task = LocalTask("process", func=process_data, outputs=['processed'])
anal_task = LocalTask("analyze", func=analyze_data, outputs=['result'])

# Build dependency graph
graph.add_task(prep_task)
graph.add_task(proc_task, dependencies=[prep_task])
graph.add_task(anal_task, dependencies=[proc_task])

# Execute graph
results = graph.execute()
print(f"Final result: {results['analyze']}")
```

### Complex Dependency Patterns

```python
# Diamond pattern: A -> B,C -> D
graph = TaskGraph()

task_a = LocalTask("A", func=lambda: "data")
task_b = LocalTask("B", func=lambda data: f"processed_{data}_B")
task_c = LocalTask("C", func=lambda data: f"processed_{data}_C")
task_d = LocalTask("D", func=lambda b_result, c_result: f"combined_{b_result}_{c_result}")

graph.add_task(task_a)
graph.add_task(task_b, dependencies=[task_a])
graph.add_task(task_c, dependencies=[task_a])
graph.add_task(task_d, dependencies=[task_b, task_c])

# Get execution order
execution_order = graph.topological_sort()
print("Execution order:", execution_order)

# Check for cycles
if graph.has_cycles():
    print("Warning: Circular dependencies detected!")
```

### Graph Visualization

```python
# Visualize task graph structure
graph.visualize(
    output_file="task_graph.png",
    layout="hierarchical",
    show_task_status=True,
    show_dependencies=True
)

# Get graph statistics
stats = graph.get_statistics()
print(f"Nodes: {stats['num_nodes']}")
print(f"Edges: {stats['num_edges']}")
print(f"Max depth: {stats['max_depth']}")
print(f"Parallelizable tasks: {stats['parallel_tasks']}")
```

## Advanced Pool Operations

### Priority-based Execution

```python
from molexp import PriorityTaskPool

# Create priority pool
priority_pool = PriorityTaskPool()

# Add tasks with priorities
high_priority_task = LocalTask("urgent", func=urgent_computation)
low_priority_task = LocalTask("background", func=background_computation)

priority_pool.add_task(high_priority_task, priority=1)  # High priority
priority_pool.add_task(low_priority_task, priority=10)  # Low priority

# Execute in priority order
results = priority_pool.execute_all()
```

### Resource-aware Pool Management

```python
from molexp import ResourceAwarePool

# Create resource-aware pool
resource_pool = ResourceAwarePool(
    max_cpu_cores=8,
    max_memory_mb=4096,
    max_concurrent_tasks=4
)

# Add tasks with resource requirements
cpu_task = LocalTask("cpu_intensive", func=cpu_computation)
cpu_task.resource_requirements = {'cpu_cores': 4, 'memory_mb': 1024}

memory_task = LocalTask("memory_intensive", func=memory_computation) 
memory_task.resource_requirements = {'cpu_cores': 1, 'memory_mb': 2048}

resource_pool.add_task(cpu_task)
resource_pool.add_task(memory_task)

# Execute with resource management
results = resource_pool.execute_all(
    monitor_resources=True,
    auto_scale=True
)
```

## Advanced Graph Operations

### Dynamic Graph Modification

```python
# Modify graph during execution
class DynamicTaskGraph(TaskGraph):
    def on_task_completed(self, task_name, result):
        """Add new tasks based on results."""
        if task_name == "analysis" and result.get('needs_refinement'):
            # Add refinement task
            refinement_task = LocalTask(
                "refinement", 
                func=refine_analysis,
                inputs={'analysis_result': result}
            )
            self.add_task(refinement_task, dependencies=[task_name])

# Use dynamic graph
dynamic_graph = DynamicTaskGraph()
# Add initial tasks...
results = dynamic_graph.execute(allow_dynamic_modification=True)
```

### Conditional Dependencies

```python
from molexp import ConditionalDependency

# Create conditional dependency
def should_run_validation(context):
    """Determine if validation should run."""
    return context.get('quality_score', 0) < 0.8

validation_dependency = ConditionalDependency(
    condition=should_run_validation,
    true_dependencies=['data_cleaning'],
    false_dependencies=['quality_check']
)

graph.add_conditional_dependency('validation_task', validation_dependency)
```

### Parallel Subgraph Execution

```python
# Execute independent subgraphs in parallel
subgraphs = graph.identify_independent_subgraphs()

from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=len(subgraphs)) as executor:
    subgraph_futures = [
        executor.submit(subgraph.execute) 
        for subgraph in subgraphs
    ]
    
    subgraph_results = [
        future.result() 
        for future in subgraph_futures
    ]

# Combine subgraph results
combined_results = {}
for result_dict in subgraph_results:
    combined_results.update(result_dict)
```

## Error Handling and Recovery

### Pool Error Management

```python
# Configure pool with error handling
error_handling_pool = TaskPool(
    on_task_failure='continue',  # Continue with other tasks
    max_retries=3,
    retry_delay=1.0,
    error_callback=lambda task, error: print(f"Task {task.name} failed: {error}")
)

# Execute with error recovery
results = error_handling_pool.execute_all(
    return_partial_results=True,
    collect_errors=True
)

# Check for errors
if error_handling_pool.has_errors():
    errors = error_handling_pool.get_errors()
    print(f"Found {len(errors)} errors:")
    for task_name, error in errors.items():
        print(f"  {task_name}: {error}")
```

### Graph Error Recovery

```python
# Configure graph with error recovery
recovery_graph = TaskGraph(
    error_strategy='isolate_failures',  # Isolate failed branches
    auto_retry=True,
    max_retry_attempts=2
)

# Add error recovery tasks
def create_recovery_task(failed_task_name):
    """Create recovery task for failed task."""
    def recovery_func():
        return f"recovery_result_for_{failed_task_name}"
    
    return LocalTask(f"{failed_task_name}_recovery", func=recovery_func)

recovery_graph.set_recovery_task_factory(create_recovery_task)

# Execute with automatic recovery
results = recovery_graph.execute(
    enable_recovery=True,
    isolation_mode='branch'
)
```

## Performance Optimization

### Pool Performance Tuning

```python
# Optimize pool for performance
optimized_pool = TaskPool(
    max_size=1000,
    batch_size=50,              # Process in batches
    prefetch_factor=2,          # Prefetch next batch
    memory_limit_mb=2048,       # Memory management
    enable_task_caching=True,   # Cache task results
    compression_level=6         # Compress large results
)

# Monitor performance
with optimized_pool.performance_monitor():
    results = optimized_pool.execute_all()

perf_stats = optimized_pool.get_performance_stats()
print(f"Throughput: {perf_stats['tasks_per_second']:.2f} tasks/sec")
print(f"Memory efficiency: {perf_stats['memory_efficiency']:.2%}")
```

### Graph Performance Optimization

```python
# Optimize graph execution
optimized_graph = TaskGraph(
    execution_strategy='adaptive',   # Adaptive scheduling
    load_balancing=True,            # Balance task distribution
    memory_optimization=True,       # Optimize memory usage
    result_streaming=True           # Stream results
)

# Configure parallel execution
parallel_config = {
    'max_workers': 8,
    'task_batching': True,
    'batch_size': 10,
    'memory_aware_scheduling': True
}

results = optimized_graph.execute(
    parallel=True,
    parallel_config=parallel_config,
    progress_callback=lambda completed, total: print(f"Progress: {completed}/{total}")
)
```

## Integration Patterns

### Pool-Graph Integration

```python
# Use pools within graph nodes
def create_batch_processing_task():
    """Create task that uses internal pool for batch processing."""
    
    def batch_processor(data_list):
        # Create internal pool for batch processing
        internal_pool = TaskPool()
        
        # Create processing tasks for each data item
        for i, data_item in enumerate(data_list):
            task = LocalTask(
                f"process_item_{i}",
                func=lambda item: process_single_item(item),
                inputs={'item': data_item}
            )
            internal_pool.add_task(task)
        
        # Execute batch
        batch_results = internal_pool.execute_all()
        return list(batch_results.values())
    
    return LocalTask("batch_processor", func=batch_processor)

# Use in graph
main_graph = TaskGraph()
batch_task = create_batch_processing_task()
main_graph.add_task(batch_task)
```

### Multi-level Task Management

```python
# Hierarchical task management
class HierarchicalTaskManager:
    def __init__(self):
        self.top_level_graph = TaskGraph()
        self.task_pools = {}
    
    def add_workflow(self, workflow_name, graph):
        """Add a workflow as a single task in top-level graph."""
        
        def execute_workflow():
            return graph.execute()
        
        workflow_task = LocalTask(workflow_name, func=execute_workflow)
        self.top_level_graph.add_task(workflow_task)
    
    def add_batch_job(self, job_name, tasks):
        """Add a batch job as a pool."""
        pool = TaskPool()
        for task in tasks:
            pool.add_task(task)
        
        def execute_batch():
            return pool.execute_all()
        
        batch_task = LocalTask(job_name, func=execute_batch)
        self.top_level_graph.add_task(batch_task)
    
    def execute_all(self):
        """Execute entire hierarchical structure."""
        return self.top_level_graph.execute()

# Usage
manager = HierarchicalTaskManager()
manager.add_workflow("data_processing", data_workflow)
manager.add_batch_job("analysis_batch", analysis_tasks)
results = manager.execute_all()
```

This comprehensive documentation covers the task pool and graph management capabilities of MolExp, providing both basic usage examples and advanced patterns for complex scientific computing workflows.
