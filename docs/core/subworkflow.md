# SubWorkflow

SubWorkflow allows groups of tasks to be treated as a single composable unit in workflows. This enables better organization, reuse, and composition of complex workflows.

## What SubWorkflow Is

A SubWorkflow is a composable group of tasks that can be treated as a single task in a workflow. It implements the `SubWorkflowProtocol`, which requires:
- `tasks`: List of tasks in the subworkflow
- `links`: List of links between tasks
- `task_id`: Unique identifier
- `config_type`: Pydantic configuration model
- `execute`: Method to execute the subworkflow

SubWorkflows enable you to encapsulate complex processing pipelines and reuse them across different workflows. They can be used as regular tasks, or expanded to reveal their internal structure for detailed execution.

## Why This Design

SubWorkflows provide several important benefits. First, they enable composition—you can build complex workflows by combining simpler subworkflows. Second, they promote reuse—a well-designed subworkflow can be used in multiple contexts without duplication.

Third, they improve organization—complex workflows become more manageable when broken into logical subworkflows. Finally, they support protocol-based design—components from other frameworks (like MolNex's `DataPipeline`) can be used as subworkflows without modification, as long as they implement the required protocol.

## How to Use

### Protocol-Based Design

SubWorkflow uses protocol-based design, meaning you don't need to inherit from any base class. As long as your component implements the required attributes and methods, it can be used as a subworkflow.

The required protocol is:

```python
@runtime_checkable
class SubWorkflowProtocol(Protocol):
    @property
    def tasks(self) -> list[Any]: ...
    
    @property
    def links(self) -> list[dict[str, str]]: ...
    
    @property
    def task_id(self) -> str: ...
    
    @property
    def config_type(self) -> type[BaseModel]: ...
    
    def execute(self, *inputs: Any) -> Any: ...
```

### Using DataPipeline as SubWorkflow

MolNex's `DataPipeline` is a perfect example of a protocol-compatible subworkflow. It already implements all required attributes and methods:

```python
from molix.data.pipeline import DataPipeline
from molix.data.node import DataNode
from molexp.workflow.subworkflow import is_subworkflow
# No import needed - use OOP method: pipeline.to_ir()

# Create a DataPipeline
pipeline = DataPipeline(
    dataset,
    AtomicDress(elements=[1, 6, 7, 8]),
    ComputeNeighborList(cutoff=5.0),
    task_id="data_pipeline",
)

# Check if it's a subworkflow (it is!)
assert is_subworkflow(pipeline)

# Access subworkflow properties
print(f"Tasks: {pipeline.tasks}")
print(f"Links: {pipeline.links}")
print(f"Task ID: {pipeline.task_id}")

# Convert to WorkflowIR
workflow_ir = pipeline.to_ir(workflow_id="pipeline_workflow")
```

### Creating Custom SubWorkflows

You can create custom subworkflows by implementing the protocol:

```python
from pydantic import BaseModel
from molexp.workflow.node import Task

class MySubWorkflowConfig(BaseModel):
    param: float = 1.0

class MySubWorkflow:
    """Custom subworkflow implementing the protocol."""
    
    config_type = MySubWorkflowConfig
    
    def __init__(self, *upstreams, task_id: str | None = None, **config_kwargs):
        self.task_id = task_id or self.__class__.__name__
        self.upstreams = list(upstreams)
        self.config = self.config_type(**config_kwargs)
        
        # Create internal tasks
        self._tasks = [
            Task1(param=self.config.param),
            Task2(),
        ]
    
    @property
    def tasks(self):
        return self._tasks
    
    @property
    def links(self):
        return [
            {'source': self._tasks[0].task_id, 'target': self._tasks[1].task_id, 'type': 'data'}
        ]
    
    def execute(self, *inputs):
        result = self._tasks[0].execute(*inputs)
        return self._tasks[1].execute(result)
```

### Converting SubWorkflow to WorkflowIR

You can convert a subworkflow to WorkflowIR for compilation and execution:

```python
# No import needed - use OOP method: pipeline.to_ir()
from molexp.compiler import compile_workflow

# Convert subworkflow to IR using OOP method
workflow_ir = pipeline.to_ir(
    workflow_id="my_workflow",
    prefix="pipeline_",  # Optional prefix for task IDs
)

# Compile
compiled = WorkflowCompiler().compile(workflow_ir)
```

### Expanding SubWorkflows in Workflows

You can expand a subworkflow within a larger workflow, replacing the subworkflow task with its internal tasks:

```python
# Expand subworkflow in existing workflow using OOP method
expanded_ir = main_workflow_ir.expand_subworkflow(
    subworkflow_task_id="data_pipeline",
    subworkflow=pipeline,
)
```

This replaces the subworkflow task with its internal tasks and updates all links accordingly.

### Complete Example

Here's a complete example using DataPipeline as a subworkflow:

```python
from molix.data.pipeline import DataPipeline
from molix.data.node import DataNode, EmptyConfig
from molexp.workflow.subworkflow import is_subworkflow
# No import needed - use OOP method: pipeline.to_ir()
from molexp.compiler import compile_workflow

# Define data processing nodes
class NormalizeNode(DataNode):
    config_type = EmptyConfig
    
    def execute(self, data: float) -> float:
        return data / 100.0

class FilterNode(DataNode):
    config_type = EmptyConfig
    
    def execute(self, data: float) -> float:
        return data if data > 0.5 else 0.0

# Create dataset
class SimpleDataset:
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        return float(idx)

# Create pipeline (subworkflow)
dataset = SimpleDataset()
pipeline = DataPipeline(
    dataset,
    NormalizeNode(task_id="normalize"),
    FilterNode(task_id="filter"),
    task_id="data_pipeline",
)

# Verify it's a subworkflow
assert is_subworkflow(pipeline)

# Convert to WorkflowIR using OOP method
workflow_ir = pipeline.to_ir()

# Compile and use
compiled = WorkflowCompiler().compile(workflow_ir)

# Execute directly
result = pipeline.execute(75.0)  # Normalize: 0.75, Filter: 0.75
print(f"Result: {result}")
```

SubWorkflows provide a powerful way to compose and reuse complex processing pipelines, making workflows more modular and maintainable.

