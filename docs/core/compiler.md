# IR and Compiler

The compiler is a key component in MolExp's architecture, responsible for converting task graphs into executable intermediate representation (IR) and generating deterministic execution plans.

## What IR and Compiler Are

IR (Intermediate Representation) is a standardized data structure for workflows, completely independent of Python code. It can be serialized to JSON and passed between different environments. The compiler receives workflow IR, validates its correctness, and generates execution plans.

The compiler's main responsibilities include: validating workflow structure (checking for target nodes, circular dependencies), computing execution order (topological sort), and identifying tasks that can execute in parallel. All of this happens before execution, allowing us to discover potential issues early.

## Why This Design

Separating compilation and execution has several important advantages. First, static validation allows us to discover errors before execution, such as circular dependencies, missing dependencies, invalid configurations, etc. This is much more efficient than discovering errors at runtime, especially when dealing with large workflows.

Second, deterministic execution order ensures result reproducibility. The compiler uses topological sort algorithms to generate execution order, which is deterministic and doesn't depend on task creation order or execution environment. This is crucial for reproducibility in scientific computing.

Finally, IR's standardized format allows workflows to be shared and version-controlled across different environments. You can save workflows as JSON files, execute them on different machines, or pass workflow definitions through APIs.

## How to Use

### Building Workflow IR

Workflow IR is represented by the `WorkflowIR` model, which contains version information and workflow definition:

```python
from molexp.ir.models import WorkflowIR, Workflow, Task as IRTask, Link, LinkType

workflow_ir = WorkflowIR(
    version="1.0",
    workflow=Workflow(
        id="my_workflow",
        name="My Workflow",
        tasks=[
            IRTask(
                id="task_1",
                task_id="multiply",  # Task identifier in registry
                args={"factor": 2.0},  # Task configuration parameters
            ),
            IRTask(
                id="task_2",
                task_id="add",
                args={"offset": 10},
            ),
        ],
        links=[
            Link(
                source="task_1",
                target="task_2",
                type=LinkType.DATA,  # Data flow link
            ),
        ],
        targets=["task_2"],  # Target tasks to execute
    ),
)
```

The `Workflow` model contains several key fields: `tasks` is a list of tasks, each with a unique `id`, `task_id` (for finding task definitions), and `args` (configuration parameters). `links` define connections between tasks, where `type` can be `DATA` (data flow) or `DEPENDENCY` (dependency relationship). `targets` specifies the list of task IDs to execute.

### Compiling Workflows

Use the `compile_workflow` function to compile workflows:

```python
from molexp.compiler import compile_workflow, ValidationError

try:
    compiled = WorkflowCompiler().compile(workflow_ir)
    print("Compilation successful!")
except ValidationError as e:
    print(f"Compilation failed: {e}")
```

The compiler performs the following validations:

1. **Check target nodes**: Ensure at least one target node exists
2. **Check node existence**: Ensure all nodes referenced by links exist
3. **Detect circular dependencies**: Use topological sort to detect cycles

If validation passes, the compiler returns the compiled workflow IR (currently the same as input, but may add optimizations in the future).

### Generating Execution Plans

Use the `plan_execution` function to generate execution plans:

```python
from molexp.compiler import plan_execution

# Generate complete execution plan
execution_order = plan_execution(compiled)
print(f"Execution order: {execution_order}")

# Execute only specific targets
partial_order = plan_execution(compiled, targets=["task_2"])
print(f"Partial execution order: {partial_order}")
```

The execution plan is a list of task IDs in topological sort order. This means tasks in the list can be executed in this order, with all dependencies of each task executed before it.

### Loading Workflows from Files

You can save workflows as JSON files and load them from files:

```python
from molexp.ir.loader import load_workflow_from_file, load_workflow_from_json

# Load from file
workflow_ir = load_workflow_from_file("workflow.json")

# Load from JSON string
json_str = '{"version": "1.0", "workflow": {...}}'
workflow_ir = load_workflow_from_json(json_str)
```

This allows you to separate workflow definitions from execution code, achieving better maintainability and reusability.

### Handling Complex Dependencies

When workflows have multiple branches and merge points, the compiler handles them automatically:

```python
workflow_ir = WorkflowIR(
    version="1.0",
    workflow=Workflow(
        id="complex_workflow",
        tasks=[
            IRTask(id="start", task_id="load", args={}),
            IRTask(id="branch_a", task_id="process_a", args={}),
            IRTask(id="branch_b", task_id="process_b", args={}),
            IRTask(id="merge", task_id="combine", args={}),
        ],
        links=[
            Link(source="start", target="branch_a", type=LinkType.DATA),
            Link(source="start", target="branch_b", type=LinkType.DATA),
            Link(source="branch_a", target="merge", type=LinkType.DATA),
            Link(source="branch_b", target="merge", type=LinkType.DATA),
        ],
        targets=["merge"],
    ),
)

compiled = WorkflowCompiler().compile(workflow_ir)
execution_order = plan_execution(compiled)
# Possible execution order: ["start", "branch_a", "branch_b", "merge"]
# Note: branch_a and branch_b can execute in parallel
```

The compiler identifies that `branch_a` and `branch_b` can execute in parallel since they both only depend on `start` and are independent of each other.

### Complete Example

Here's a complete compilation and execution example:

```python
from molexp.ir.models import WorkflowIR, Workflow, Task as IRTask, Link, LinkType
from molexp.compiler import compile_workflow, plan_execution, ValidationError
from molexp.ir.loader import load_workflow_from_file
import json

def build_workflow():
    """Build workflow IR"""
    return WorkflowIR(
        version="1.0",
        workflow=Workflow(
            id="example_workflow",
            tasks=[
                IRTask(id="load", task_id="load_data", args={"file": "input.txt"}),
                IRTask(id="transform", task_id="transform_data", args={"scale": 2.0}),
                IRTask(id="save", task_id="save_data", args={"output": "output.txt"}),
            ],
            links=[
                Link(source="load", target="transform", type=LinkType.DATA),
                Link(source="transform", target="save", type=LinkType.DATA),
            ],
            targets=["save"],
        ),
    )

def main():
    # Build workflow
    workflow_ir = build_workflow()
    
    # Compile
    try:
        compiled = WorkflowCompiler().compile(workflow_ir)
        print("✓ Compilation successful")
    except ValidationError as e:
        print(f"✗ Compilation failed: {e}")
        return
    
    # Generate execution plan
    execution_order = plan_execution(compiled)
    print(f"Execution order: {' -> '.join(execution_order)}")
    
    # Can save as JSON
    with open("workflow.json", "w") as f:
        json.dump(compiled.model_dump(), f, indent=2)
    print("✓ Workflow saved to workflow.json")

if __name__ == "__main__":
    main()
```

Through the compiler, we can validate workflow correctness in advance, generate execution plans, and serialize workflows for saving. This lays the foundation for subsequent execution and reproducibility.
