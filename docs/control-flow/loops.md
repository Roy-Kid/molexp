# Loops

Loop tasks enable iterative execution in workflows. MolExp provides several loop types: fixed-iteration loops, while loops, and for loops, each suited for different iteration patterns.

## What Loops Are

Loop tasks execute a body task multiple times, with the output of each iteration potentially feeding into the next. MolExp provides `LoopTask` for fixed iterations, `WhileLoopTask` for condition-based iteration, and `ForLoopTask` for iterating over collections with index tracking.

Loops are essential for iterative algorithms, optimization procedures, or any scenario requiring repeated computation with state accumulation. By making loops explicit tasks, the workflow structure shows iteration clearly, and the engine can monitor progress and handle failures appropriately.

## Why This Design

Explicit loop tasks have several advantages over imperative loops in task code. First, they make iteration visible in the workflow graph—you can see that a task runs in a loop and how many iterations are planned. Second, they enable better monitoring—you can track iteration count, see intermediate results, and understand loop termination conditions.

Third, they provide better error handling—if a loop iteration fails, the engine can decide whether to continue, retry, or abort the loop. Finally, they make workflows more declarative—the iteration strategy is part of the workflow definition, not hidden in task implementations.

## How to Use

### Fixed Iteration Loop

`LoopTask` executes a body task a fixed number of times:

```python
from molexp.workflow.control.loop import LoopTask, LoopConfig

class LoopConfig(BaseModel):
    iterations: int = Field(..., ge=1)

@registry.register("control.loop", LoopConfig)
class LoopTask(Task[LoopConfig, Any]):
    config_type = LoopConfig
    
    def execute(self, initial_value: Any) -> Any:
        value = initial_value
        for i in range(self.config.iterations):
            value = self.body_task(value)
        return value

# Create loop task
body_task = SomeTask(task_id="body")
loop = LoopTask(
    body_task=body_task,
    task_id="loop",
    iterations=10,
)

# Execute
result = loop.execute(initial_value)
```

### While Loop

`WhileLoopTask` executes while a condition is true:

```python
from molexp.workflow.control.loop import WhileLoopTask, WhileLoopConfig

class WhileLoopConfig(BaseModel):
    max_iterations: int = Field(default=1000, ge=1)
    condition_fn: Callable[[Any], bool]

@registry.register("control.while_loop", WhileLoopConfig)
class WhileLoopTask(Task[WhileLoopConfig, Any]):
    config_type = WhileLoopConfig
    
    def execute(self, initial_value: Any) -> Any:
        value = initial_value
        iteration = 0
        while (iteration < self.config.max_iterations and 
               self.config.condition_fn(value)):
            value = self.body_task(value)
            iteration += 1
        return value

# Create while loop
def not_converged(state: dict) -> bool:
    return state.get("converged", False) == False

body_task = OptimizationStepTask(task_id="optimize")
while_loop = WhileLoopTask(
    body_task=body_task,
    condition_fn=not_converged,
    max_iterations=100,
    task_id="optimize_loop",
)

# Execute
result = while_loop.execute(initial_state)
```

### For Loop

`ForLoopTask` iterates over a collection with index tracking:

```python
from molexp.workflow.control.loop import ForLoopTask, ForLoopConfig

class ForLoopConfig(BaseModel):
    iterations: int = Field(..., ge=1)

@registry.register("control.for_loop", ForLoopConfig)
class ForLoopTask(Task[ForLoopConfig, Any]):
    config_type = ForLoopConfig
    
    def execute(self, collection: list) -> list:
        results = []
        for i in range(self.config.iterations):
            if i < len(collection):
                result = self.body_task(collection[i], i)
                results.append(result)
        return results

# Create for loop
body_task = ProcessItemTask(task_id="process")
for_loop = ForLoopTask(
    body_task=body_task,
    iterations=10,
    task_id="process_loop",
)

# Execute
items = [1, 2, 3, 4, 5]
results = for_loop.execute(items)
```

### Complete Example

Here's a complete example using loops for iterative optimization:

```python
from molexp.workflow.node import Task
from molexp.workflow.control.loop import WhileLoopTask
from pydantic import BaseModel

class OptimizationState(BaseModel):
    value: float
    converged: bool = False
    iteration: int = 0

class OptimizeStepConfig(BaseModel):
    learning_rate: float = 0.01

@registry.register("optimize_step", OptimizeStepConfig)
class OptimizeStepTask(Task[OptimizeStepConfig, OptimizationState]):
    config_type = OptimizeStepConfig
    
    def execute(self, state: OptimizationState) -> OptimizationState:
        # Simple gradient descent step
        new_value = state.value - self.config.learning_rate * state.value
        converged = abs(new_value - state.value) < 1e-6
        
        return OptimizationState(
            value=new_value,
            converged=converged,
            iteration=state.iteration + 1,
        )

def main():
    # Create optimization step
    step = OptimizeStepTask(task_id="step", learning_rate=0.01)
    
    # Create while loop
    def not_converged(state: OptimizationState) -> bool:
        return not state.converged and state.iteration < 100
    
    loop = WhileLoopTask(
        body_task=step,
        condition_fn=not_converged,
        max_iterations=100,
        task_id="optimize",
    )
    
    # Initial state
    initial = OptimizationState(value=10.0)
    
    # Execute loop
    result = loop.execute(initial)
    print(f"Optimized value: {result.value}")
    print(f"Iterations: {result.iteration}")
    print(f"Converged: {result.converged}")

if __name__ == "__main__":
    main()
```

Loop tasks provide powerful iteration capabilities for workflows, making iterative algorithms explicit and monitorable in the workflow graph.



