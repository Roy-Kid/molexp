# Map and Reduce

Map and reduce are fundamental control flow patterns for processing collections. MolExp provides `MapTask` and `ReduceTask` to apply operations to collections and aggregate results.

## What Map and Reduce Are

`MapTask` applies a base task to each element in a collection, returning a list of results. `ReduceTask` aggregates a collection into a single value using operations like sum, mean, max, or min. These tasks enable you to process collections declaratively within your workflow graph.

Map and reduce are essential for batch processing scenarios—when you need to apply the same operation to many items or aggregate results across a collection. By making these operations explicit in the workflow, the engine can optimize execution and provide better monitoring.

## Why This Design

Making map and reduce explicit tasks rather than imperative loops has several benefits. First, it makes the workflow structure clear—you can see at a glance that a collection is being processed, without digging into task implementation code. Second, it enables parallel execution opportunities—the engine can identify that map operations can run in parallel.

Third, it provides better observability—you can monitor how many items were processed, track progress, and see results for each element. Finally, it makes workflows more composable—you can easily chain map and reduce operations, or combine them with other control flow tasks.

## How to Use

### Using Map Task

Map task applies a base task to each element in a collection:

```python
from molexp.workflow.control.map import MapTask, MapConfig
from molexp.ir.registry import registry

# Define a base task to apply
@registry.register("square", BaseModel)
class SquareTask(Task[BaseModel, int]):
    config_type = BaseModel
    
    def execute(self, value: int) -> int:
        return value * value

# Map task is already registered as "control.map"
# In workflow IR:
workflow_ir = WorkflowIR(
    version="1.0",
    workflow=Workflow(
        id="map_example",
        tasks=[
            IRTask(id="load", task_id="load_list", args={}),
            IRTask(id="map", task_id="control.map", args={}),
            # Map task needs base_task reference in metadata or separate config
        ],
        links=[
            Link(source="load", target="map", type=LinkType.DATA),
        ],
        targets=["map"],
    ),
)
```

In practice, map tasks are typically created programmatically when building workflows:

```python
from molexp.workflow.control.map import MapTask

# Create base task
square = SquareTask(task_id="square")

# Create map task
numbers = [1, 2, 3, 4, 5]
map_task = MapTask(numbers, base_task=square, task_id="map_squares")

# Execute map
results = map_task.execute(numbers)
# Results: [1, 4, 9, 16, 25]
```

### Using Reduce Task

Reduce task aggregates a collection into a single value:

```python
from molexp.workflow.control.reduce import ReduceTask, ReduceConfig
from pydantic import BaseModel

class ReduceConfig(BaseModel):
    method: str = "sum"  # "sum", "mean", "max", "min"

@registry.register("control.reduce", ReduceConfig)
class ReduceTask(Task[ReduceConfig, float]):
    config_type = ReduceConfig
    
    def execute(self, collection: list[float]) -> float:
        if self.config.method == "sum":
            return sum(collection)
        elif self.config.method == "mean":
            return sum(collection) / len(collection)
        elif self.config.method == "max":
            return max(collection)
        elif self.config.method == "min":
            return min(collection)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

# Use reduce task
numbers = [1, 2, 3, 4, 5]
reduce_task = ReduceTask(task_id="reduce_sum", method="sum")
total = reduce_task.execute(numbers)
# Result: 15
```

### Combining Map and Reduce

You can chain map and reduce operations:

```python
# Map: square each number
squares = MapTask(numbers, base_task=square, task_id="map")

# Reduce: sum the squares
total = ReduceTask(squares, task_id="reduce", method="sum")

# Or in workflow IR:
workflow_ir = WorkflowIR(
    version="1.0",
    workflow=Workflow(
        id="map_reduce_example",
        tasks=[
            IRTask(id="load", task_id="load_data", args={}),
            IRTask(id="map", task_id="control.map", args={}),
            IRTask(id="reduce", task_id="control.reduce", args={"method": "sum"}),
        ],
        links=[
            Link(source="load", target="map", type=LinkType.DATA),
            Link(source="map", target="reduce", type=LinkType.DATA),
        ],
        targets=["reduce"],
    ),
)
```

### Complete Example

Here's a complete example processing a dataset:

```python
from molexp.workflow.node import Task
from molexp.workflow.control.map import MapTask
from molexp.workflow.control.reduce import ReduceTask
from pydantic import BaseModel

class ProcessItemConfig(BaseModel):
    scale: float = 1.0

@registry.register("process_item", ProcessItemConfig)
class ProcessItemTask(Task[ProcessItemConfig, float]):
    config_type = ProcessItemConfig
    
    def execute(self, item: float) -> float:
        return item * self.config.scale

def main():
    # Input data
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Create processing task
    process = ProcessItemTask(task_id="process", scale=2.0)
    
    # Map: process each item
    map_task = MapTask(data, base_task=process, task_id="map_process")
    processed = map_task.execute(data)
    print(f"Processed: {processed}")  # [2.0, 4.0, 6.0, 8.0, 10.0]
    
    # Reduce: compute mean
    reduce_task = ReduceTask(task_id="reduce_mean", method="mean")
    mean = reduce_task.execute(processed)
    print(f"Mean: {mean}")  # 6.0

if __name__ == "__main__":
    main()
```

Map and reduce tasks provide powerful, declarative ways to process collections in your workflows, making batch operations explicit and optimizable.



