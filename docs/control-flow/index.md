# Control Flow

MolExp provides built-in control flow tasks for common workflow patterns like iteration, conditionals, and parallel execution. These tasks enable you to build complex workflows declaratively.

## What Control Flow Is

Control flow tasks are special tasks that manage the execution flow of your workflow. They include tasks for mapping over collections, conditional execution, loops, parallel execution, and reduction operations.

**Note**: Control flow tasks carry runtime callables or nested tasks and are **not replayable**. Persisted workflows must avoid them.

Control flow tasks abstract common patterns, making workflows more expressive and easier to understand. Instead of writing imperative loops or conditionals in your task code, you can use declarative control flow tasks that are part of the workflow graph itself.

## Why This Design

Control flow as tasks has several advantages. First, it makes control flow explicit in the workflow graph, allowing the compiler and engine to reason about execution order and parallelization opportunities. Second, it enables better monitoring and debugging—you can see exactly which branch executed or how many iterations ran.

Third, control flow tasks are composable—you can nest them, combine them, and use them with any other tasks. This provides great flexibility for building complex workflows in memory.

## How to Use

Control flow tasks are registered in the task registry and can be used in workflow IR:

```python
from molexp.ir.models import WorkflowIR, Workflow, Task as IRTask, Link, LinkType

# Example: Using map task in workflow
workflow_ir = WorkflowIR(
    version="1.0",
    workflow=Workflow(
        id="control_flow_example",
        tasks=[
            IRTask(id="load", task_id="load_data", args={}),
            IRTask(id="map", task_id="control.map", args={}),
            IRTask(id="save", task_id="save_data", args={}),
        ],
        links=[
            Link(source="load", target="map", type=LinkType.DATA),
            Link(source="map", target="save", type=LinkType.DATA),
        ],
        targets=["save"],
    ),
)
```

The following sections cover each control flow task in detail, showing how to use them in your workflows.


