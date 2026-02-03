# Task Abstraction

Task is the most fundamental and important abstraction in MolExp. Understanding Task's design and usage is key to mastering MolExp.

## What Task Is

Task is a generic abstract class that represents an executable unit in a workflow. Each Task instance encapsulates a computation logic that accepts outputs from upstream tasks as inputs, performs computation, and produces outputs. Task uses Pydantic models for configuration, ensuring type safety and configuration validation.

Task's design follows pure functional principles: given the same inputs and configuration, a Task always produces the same output. This determinism is crucial for reproducibility in scientific computing.

## Why This Design

Task's design has several key considerations. First, configuration is determined and validated at construction time, allowing us to discover configuration errors before execution rather than at runtime. Second, Task is type-safe. Through generic parameters `Task[ConfigType, OutputType]`, we can explicitly know each task's configuration type and output type, greatly reducing type-related errors.

Additionally, Task's configuration is static, meaning once created, the configuration cannot change. This design ensures task behavior is predictable and enables complete workflow serialization. Finally, Task supports declarative definition of upstream dependencies through constructor parameters, making dependency relationships clear at a glance.

## How to Use

Let's learn how to define and use Tasks through several examples.

### Basic Task Definition

The simplest task only needs to inherit from `Task`, define a configuration type, and implement the `execute` method:

```python
from molexp.workflow.node import Task
from pydantic import BaseModel

class MultiplyConfig(BaseModel):
    factor: float = 2.0

class MultiplyTask(Task[MultiplyConfig, float]):
    config_type = MultiplyConfig
    
    def execute(self, value: float) -> float:
        return value * self.config.factor
```

Here we define a multiplication task that receives a float, multiplies it by the factor in the configuration, and returns the result. Note that `self.config` contains the validated configuration object, and you can safely access its fields.

### Multi-Input Tasks

Some tasks require multiple inputs, which you can receive through the `execute` method's parameters:

```python
class AddConfig(BaseModel):
    pass

class AddTask(Task[AddConfig, float]):
    config_type = AddConfig
    
    def execute(self, a: float, b: float) -> float:
        return a + b
```

When connecting this task in a workflow, ensure the number and order of upstream tasks match the `execute` method's parameters.

### Tasks Without Configuration

If a task doesn't need configuration, you can use `BaseModel` as the configuration type:

```python
from pydantic import BaseModel

class IdentityTask(Task[BaseModel, any]):
    config_type = BaseModel
    
    def execute(self, value: any) -> any:
        return value
```

### Declaring Dependencies

Dependencies are declared explicitly via `Link` objects with **output-to-input mappings**:

```python
from molexp.workflow import Workflow, Link

load = LoadTask(file_path="data.txt")
transform = TransformTask(scale=2.0)

link = Link(
    source=load.task_id,
    target=transform.task_id,
    mapping={"result": "input_value"},
)

workflow = Workflow.from_tasks([load, transform], links=[link])
```

### Task Registration

To make a task persistable and replayable, register it with a deterministic task type ID:

```python
from molexp.workflow import register_task

register_task(MultiplyTask)  # ID defaults to module.path.ClassName
```

Registration is required for workflow serialization and replay.

### Accessing Run Context

During task execution, you may need to access the current run context, such as registering assets, logging, etc.:

```python
from molexp.workflow.context import get_current_context
from molexp.assets import register_asset

class ProcessTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, data: list) -> str:
        # Process data
        result = process(data)
        
        # Save result and register as asset
        output_path = "result.txt"
        with open(output_path, "w") as f:
            f.write(result)
        
        # Register asset to current run
        register_asset(output_path, label="processed_data")
        
        return output_path
```

Through `get_current_context()`, you can access the current run context, accessing workspace, run metadata, and other information. The `register_asset` function automatically registers assets to the current run's context.

### Complete Example

Here's a complete task definition example showcasing all key features:

```python
from molexp.workflow.node import Task
from molexp.ir.registry import registry
from molexp.workflow.context import get_current_context
from molexp.assets import register_asset
from pydantic import BaseModel, Field
from pathlib import Path

class DataProcessConfig(BaseModel):
    output_dir: str = Field(default="./output")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

@registry.register("data_process", DataProcessConfig)
class DataProcessTask(Task[DataProcessConfig, Path]):
    """Process data and save results"""
    config_type = DataProcessConfig
    
    def execute(self, input_data: list[float]) -> Path:
        # Use configuration parameters
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process data
        filtered = [x for x in input_data if x > self.config.threshold]
        
        # Save results
        output_path = output_dir / "filtered_data.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(map(str, filtered)))
        
        # Register asset
        register_asset(output_path, label="filtered_data", meta={"count": len(filtered)})
        
        return output_path
```

This example shows how to define configuration models with validation, how to use configuration parameters, how to access context and register assets. With these, you can define any complex task.

## Protocol-Based Design

MolExp doesn't require you to inherit from `Task` to create compatible components. Instead, it uses protocol-based design (duck typing), meaning any class that implements the required interface can be used as a task in MolExp workflows.

### What Protocol-Based Design Is

Protocol-based design means MolExp checks for the presence of required methods and attributes rather than requiring inheritance from a specific base class. If your class has an `execute` method, a `config_type` attribute, and follows the configuration pattern, it can be used as a task—even if it doesn't inherit from `Task`.

This design enables seamless integration with other frameworks and packages. For example, components from MolNex (like `DataNode` and `DataPipeline`) can be used directly in MolExp workflows without modification, as long as they implement the required protocol.

### Why This Design

Protocol-based design provides several important benefits. First, it enables interoperability—you can use components from other packages without wrapping or adapting them. Second, it reduces coupling—your code doesn't need to depend on MolExp's base classes, making it more flexible and testable.

Third, it allows for gradual adoption—you can start using MolExp with existing code by simply ensuring it follows the protocol, without requiring a complete rewrite. Finally, it promotes composition over inheritance, making code more modular and easier to maintain.

### How to Implement the Protocol

To make your class compatible with MolExp without inheriting from `Task`, you need to implement the following:

1. **`config_type` class attribute**: A Pydantic model class for configuration
2. **`execute` method**: The computation logic that takes inputs and returns output
3. **`task_id` attribute**: A unique identifier (optional, can default to class name)
4. **Constructor pattern**: Accept `*upstreams` and `**config_kwargs`, create `self.config` from `config_type`

Here's an example of a protocol-compatible task:

```python
from pydantic import BaseModel
from typing import Any

class MyConfig(BaseModel):
    multiplier: float = 2.0

class ProtocolCompatibleTask:
    """A task that follows MolExp protocol without inheriting from Task."""
    
    config_type = MyConfig  # Required: Pydantic config model
    
    def __init__(self, *upstreams: Any, task_id: str | None = None, **config_kwargs: Any):
        """Initialize with upstreams and configuration."""
        self.task_id = task_id or self.__class__.__name__
        self.upstreams = list(upstreams)
        
        # Create and validate configuration
        self.config = self.config_type(**config_kwargs)
    
    def execute(self, value: float) -> float:
        """Execute the task (required method)."""
        return value * self.config.multiplier
    
    def __call__(self, *inputs: Any) -> Any:
        """Optional: make it callable like Task."""
        return self.execute(*inputs)
```

This class can be used in MolExp workflows just like a class that inherits from `Task`.

### Real-World Example: MolNex Integration

MolNex's `DataNode` and `DataPipeline` are perfect examples of protocol-compatible components. They don't inherit from MolExp's `Task`, but they implement the required protocol and can be used directly in MolExp workflows.

Here's how `DataNode` from MolNex implements the protocol:

```python
# From molnex: molix/data/node.py
class DataNode:
    """A protocol-compatible node that works with both MolNex and MolExp."""
    
    config_type = SomeConfig  # Pydantic config model
    
    def __init__(self, *upstreams, **config_kwargs):
        self.upstreams = list(upstreams)
        self.config = self.config_type(**config_kwargs)
        # ... other initialization
    
    def execute(self, frame):
        """Process a single frame."""
        # ... processing logic
        return processed_frame
```

When used in a workflow, `DataNode` instances can be composed with MolExp tasks:

```python
from molix.data.node import DataNode
from molix.data.pipeline import DataPipeline
from molexp.workflow.node import Task

# MolNex DataNode (protocol-compatible)
atomic_dress = AtomicDress(elements=[1, 6, 7, 8, 9], target_key="U0")

# MolExp Task
save_task = SaveTask(task_id="save")

# They can be composed together
pipeline = DataPipeline(
    dataset,
    atomic_dress,  # MolNex component
    # ... other nodes
)

# Pipeline itself is also protocol-compatible
# It can be used in MolExp workflows
```

### Benefits of Protocol-Based Design

The protocol-based approach enables several powerful patterns:

**Framework Interoperability**: Components from different frameworks can work together seamlessly. For example, you can use MolNex's data processing nodes alongside MolExp's workflow tasks.

**No Forced Dependencies**: Your code doesn't need to import MolExp to be compatible. As long as it follows the protocol, it can be used in MolExp workflows.

**Flexible Composition**: You can mix and match components from different sources, creating workflows that leverage the best parts of each framework.

**Easy Testing**: Since components don't require MolExp base classes, you can test them independently without MolExp dependencies.

### Complete Example

Here's a complete example showing protocol-compatible components from different sources working together:

```python
from pydantic import BaseModel
from molix.data.node import DataNode, DataPipeline
from molix.datasets.qm9 import QM9Dataset

# MolNex component (protocol-compatible)
class AtomicDress(DataNode):
    config_type = AtomicDressConfig
    
    def __init__(self, *upstreams, **config_kwargs):
        super().__init__(*upstreams, **config_kwargs)
        # ... initialization
    
    def execute(self, frame):
        # ... processing
        return frame

# Create protocol-compatible pipeline
dataset = QM9Dataset(root="./data", split="train")
pipeline = DataPipeline(
    dataset,
    AtomicDress(elements=[1, 6, 7, 8, 9], target_key="U0"),
)

# Pipeline can be used in MolExp workflows
# It implements the protocol: has execute(), config_type, etc.
# No inheritance from MolExp's Task required!
```

This design makes MolExp highly extensible and interoperable, allowing you to leverage existing components from other frameworks without modification.
