# Conditional Execution

Conditional tasks enable branching in workflows based on runtime conditions. MolExp's `ConditionalTask` evaluates a condition and executes either the "then" or "else" branch, but not both.

## What Conditional Tasks Are

`ConditionalTask` is a control flow task that evaluates a condition function on its input and executes one of two branches based on the result. The condition is evaluated lazily—only the selected branch is executed, making it efficient for expensive operations.

Conditional tasks enable workflows to make decisions at runtime, adapting behavior based on data or intermediate results. This is essential for scenarios like error handling, data validation, or adaptive processing strategies.

## Why This Design

Making conditionals explicit tasks rather than if-statements in task code has several advantages. First, it makes branching explicit in the workflow graph—you can see the decision points and branches visually. Second, it enables better monitoring—you can track which branch executed and why.

Third, it makes workflows more declarative—the conditional logic is part of the workflow definition, not hidden in task implementations. Finally, it enables optimization opportunities—the engine can potentially optimize branch selection or parallelize independent branches.

## How to Use

### Basic Conditional

Create a conditional task with a condition function and two branch tasks:

```python
from molexp.workflow.control.conditional import ConditionalTask, ConditionalConfig

# Define condition function
def is_positive(value: float) -> bool:
    return value > 0

# Define branch tasks
@registry.register("process_positive", BaseModel)
class ProcessPositiveTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, value: float) -> str:
        return f"Positive: {value}"

@registry.register("process_negative", BaseModel)
class ProcessNegativeTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, value: float) -> str:
        return f"Negative: {value}"

# Create conditional task
positive_task = ProcessPositiveTask(task_id="positive")
negative_task = ProcessNegativeTask(task_id="negative")

conditional = ConditionalTask(
    condition_input=some_value,
    condition_fn=is_positive,
    then_task=positive_task,
    else_task=negative_task,
    task_id="conditional",
)

# Execute
result = conditional.execute(some_value)
```

### Using in Workflow IR

In workflow IR, conditionals require special handling since they involve function references. Typically, you'd define the conditional structure and use metadata or configuration to specify the condition:

```python
# Note: Conditional tasks in IR may require special serialization
# of condition functions. In practice, you might use task IDs
# that encode the condition logic, or use a separate condition task.

workflow_ir = WorkflowIR(
    version="1.0",
    workflow=Workflow(
        id="conditional_example",
        tasks=[
            IRTask(id="load", task_id="load_data", args={}),
            IRTask(id="check", task_id="check_condition", args={}),
            IRTask(id="then_branch", task_id="process_positive", args={}),
            IRTask(id="else_branch", task_id="process_negative", args={}),
            IRTask(id="conditional", task_id="control.conditional", args={}),
        ],
        links=[
            Link(source="load", target="check", type=LinkType.DATA),
            Link(source="check", target="conditional", type=LinkType.DATA),
            Link(source="conditional", target="then_branch", type=LinkType.DEPENDENCY),
            Link(source="conditional", target="else_branch", type=LinkType.DEPENDENCY),
        ],
        targets=["conditional"],
    ),
)
```

### Complex Conditions

You can use more complex condition functions:

```python
def is_valid_molecule(data: dict) -> bool:
    """Check if molecule data is valid"""
    return (
        "atoms" in data
        and len(data["atoms"]) > 0
        and "bonds" in data
    )

def should_optimize(structure: dict) -> bool:
    """Decide if structure needs optimization"""
    energy = structure.get("energy", float("inf"))
    return energy > threshold

# Use in conditional
conditional = ConditionalTask(
    condition_input=molecule_data,
    condition_fn=is_valid_molecule,
    then_task=process_valid_task,
    else_task=handle_invalid_task,
    task_id="validate",
)
```

### Nested Conditionals

You can nest conditionals for complex decision trees:

```python
# Outer conditional
outer = ConditionalTask(
    condition_input=data,
    condition_fn=is_positive,
    then_task=positive_task,
    else_task=inner_conditional,  # Inner conditional as else branch
    task_id="outer",
)

# Inner conditional
inner_conditional = ConditionalTask(
    condition_input=data,
    condition_fn=lambda x: abs(x) > 10,
    then_task=large_negative_task,
    else_task=small_negative_task,
    task_id="inner",
)
```

### Complete Example

Here's a complete example with error handling:

```python
from molexp.workflow.node import Task
from molexp.workflow.control.conditional import ConditionalTask
from pydantic import BaseModel

class ProcessConfig(BaseModel):
    threshold: float = 0.5

@registry.register("process", ProcessConfig)
class ProcessTask(Task[ProcessConfig, dict]):
    config_type = ProcessConfig
    
    def execute(self, data: dict) -> dict:
        value = data.get("value", 0.0)
        if value > self.config.threshold:
            return {"status": "high", "value": value}
        else:
            return {"status": "low", "value": value}

@registry.register("handle_high", BaseModel)
class HandleHighTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, result: dict) -> str:
        return f"High value detected: {result['value']}"

@registry.register("handle_low", BaseModel)
class HandleLowTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, result: dict) -> str:
        return f"Low value: {result['value']}"

def main():
    # Process data
    process = ProcessTask(task_id="process", threshold=0.5)
    result = process.execute({"value": 0.7})
    
    # Conditional handling
    def is_high(result: dict) -> bool:
        return result["status"] == "high"
    
    handle_high = HandleHighTask(task_id="handle_high")
    handle_low = HandleLowTask(task_id="handle_low")
    
    conditional = ConditionalTask(
        condition_input=result,
        condition_fn=is_high,
        then_task=handle_high,
        else_task=handle_low,
        task_id="conditional",
    )
    
    output = conditional.execute(result)
    print(output)

if __name__ == "__main__":
    main()
```

Conditional tasks provide a declarative way to add branching logic to your workflows, making decision points explicit and monitorable.



