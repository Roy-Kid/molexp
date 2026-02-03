# Parallel Execution

`ParallelMapTask` enables parallel execution of tasks over collections, significantly improving performance for independent computations. It uses Python's concurrent execution facilities to run multiple tasks simultaneously.

## What Parallel Execution Is

`ParallelMapTask` is similar to `MapTask`, but executes the base task on each collection element in parallel rather than sequentially. It supports both thread-based and process-based parallelism, allowing you to choose the appropriate execution model based on your tasks' characteristics.

Parallel execution is crucial for performance when dealing with large collections or computationally expensive tasks. By making parallelism explicit in the workflow, the engine can optimize resource usage and provide better progress tracking.

## Why This Design

Explicit parallel tasks have several advantages. First, they make parallelism visible in the workflow graph—you can see which operations run in parallel and understand performance characteristics. Second, they enable better resource management—you can control the number of workers and choose between threads and processes.

Third, they provide better error handling—if one parallel task fails, others can continue, and the engine can report which tasks succeeded or failed. Finally, they make workflows more declarative—the parallelization strategy is part of the workflow definition, not hidden in task implementations.

## How to Use

### Basic Parallel Map

Create a parallel map task to process a collection in parallel:

```python
from molexp.workflow.control.parallel import ParallelMapTask, ParallelMapConfig

class ParallelMapConfig(BaseModel):
    max_workers: int = Field(default=4, ge=1)
    use_processes: bool = Field(default=False)

@registry.register("control.parallel_map", ParallelMapConfig)
class ParallelMapTask(Task[ParallelMapConfig, list]):
    config_type = ParallelMapConfig
    
    def execute(self, collection: list) -> list:
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        
        executor_class = (
            ProcessPoolExecutor if self.config.use_processes 
            else ThreadPoolExecutor
        )
        
        def process_item(item):
            return self.base_task(item)
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(process_item, collection))
        
        return results

# Create parallel map task
base_task = ExpensiveTask(task_id="expensive")
parallel_map = ParallelMapTask(
    collection=[1, 2, 3, 4, 5],
    base_task=base_task,
    task_id="parallel_process",
    max_workers=4,
    use_processes=False,  # Use threads
)

# Execute
results = parallel_map.execute([1, 2, 3, 4, 5])
```

### Threads vs Processes

Choose between threads and processes based on your task characteristics:

```python
# Use threads for I/O-bound tasks
io_task = ParallelMapTask(
    collection=urls,
    base_task=fetch_url_task,
    max_workers=8,
    use_processes=False,  # Threads for I/O
    task_id="fetch_parallel",
)

# Use processes for CPU-bound tasks
cpu_task = ParallelMapTask(
    collection=data_chunks,
    base_task=compute_task,
    max_workers=4,
    use_processes=True,  # Processes for CPU
    task_id="compute_parallel",
)
```

Threads are better for I/O-bound operations (file I/O, network requests) because they can yield during waits. Processes are better for CPU-bound operations because they can utilize multiple CPU cores without Python's GIL limitations.

### Configuring Workers

Control the number of parallel workers:

```python
# Few workers for resource-constrained environments
small_parallel = ParallelMapTask(
    collection=items,
    base_task=task,
    max_workers=2,
    task_id="small_parallel",
)

# Many workers for high-performance scenarios
large_parallel = ParallelMapTask(
    collection=items,
    base_task=task,
    max_workers=16,
    task_id="large_parallel",
)
```

The optimal number of workers depends on your system resources, task characteristics, and collection size. For CPU-bound tasks, typically use the number of CPU cores. For I/O-bound tasks, you can use more workers.

### Error Handling

Parallel tasks handle errors gracefully:

```python
# If one task fails, others continue
# Results list will contain results or exceptions
parallel_map = ParallelMapTask(
    collection=items,
    base_task=risky_task,
    max_workers=4,
    task_id="parallel_risky",
)

try:
    results = parallel_map.execute(items)
    # Check for exceptions in results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Item {i} failed: {result}")
        else:
            print(f"Item {i} succeeded: {result}")
except Exception as e:
    print(f"Parallel execution failed: {e}")
```

### Complete Example

Here's a complete example processing files in parallel:

```python
from molexp.workflow.node import Task
from molexp.workflow.control.parallel import ParallelMapTask
from pydantic import BaseModel
from pathlib import Path

class ProcessFileConfig(BaseModel):
    output_dir: str = "./output"

@registry.register("process_file", ProcessFileConfig)
class ProcessFileTask(Task[ProcessFileConfig, Path]):
    config_type = ProcessFileConfig
    
    def execute(self, file_path: Path) -> Path:
        # Simulate expensive file processing
        output_path = Path(self.config.output_dir) / f"processed_{file_path.name}"
        
        # Process file (I/O operation)
        with open(file_path) as f:
            data = f.read()
        
        processed = data.upper()  # Simple processing
        
        with open(output_path, "w") as f:
            f.write(processed)
        
        return output_path

def main():
    # List of files to process
    files = [
        Path("file1.txt"),
        Path("file2.txt"),
        Path("file3.txt"),
        Path("file4.txt"),
    ]
    
    # Create processing task
    process = ProcessFileTask(task_id="process", output_dir="./output")
    
    # Create parallel map (use threads for I/O)
    parallel = ParallelMapTask(
        collection=files,
        base_task=process,
        max_workers=4,
        use_processes=False,  # Threads for I/O-bound
        task_id="parallel_process",
    )
    
    # Execute
    import time
    start = time.time()
    results = parallel.execute(files)
    elapsed = time.time() - start
    
    print(f"Processed {len(results)} files in {elapsed:.2f} seconds")
    for result in results:
        print(f"Output: {result}")

if __name__ == "__main__":
    main()
```

Parallel execution tasks provide powerful performance improvements for independent computations, making parallelism explicit and controllable in your workflows.



