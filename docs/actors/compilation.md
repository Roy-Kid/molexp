# Workflow Compilation

Workflow compilation transforms a high-level workflow definition into an executable representation with type information, channel allocations, and execution plans. The compilation process validates workflow structure, detects task types through return type inspection, allocates communication channels, and prepares metadata needed for execution.

The workflow compiler serves as the bridge between workflow definition and execution. When you create a workflow by specifying tasks and links, you describe what you want to happen at a high level. The compiler processes this description to determine how to execute it, checking for errors and computing necessary runtime structures. This separation allows workflow definitions to remain simple and declarative while execution mechanisms handle complex coordination automatically.

Compilation begins when you pass a workflow to the WorkflowEngine constructor. The engine creates a WorkflowCompiler instance and calls its compile method with the workflow object. The compile method orchestrates several analysis and transformation passes, each building on previous results.

```python
from molexp.workflow import Workflow, WorkflowEngine
from molexp.workflow.compiler import WorkflowCompiler

# Define workflow
workflow = Workflow.from_tasks(
    tasks=[actor1, actor2, actor3],
    links=[
        Link(source=actor1, target=actor2, mapping={'out': 'in'}),
        Link(source=actor2, target=actor3, mapping={'out': 'in'})
    ],
    name="pipeline"
)

# Compilation happens automatically
compiler = WorkflowCompiler()
compiled = compiler.compile(workflow)

# Or through engine construction
engine = WorkflowEngine(workflow)
# Engine internally compiles and stores result
```

Type detection forms the foundation of compilation. The compiler examines each task's execute method to determine whether it's a batch task or an actor. This detection uses Python's inspect module to extract the method signature and its return type annotation. If the return type is AsyncGenerator, the compiler classifies the task as an actor. Otherwise, it assumes batch execution.

The detection process handles several edge cases. If a task lacks return type annotations, the compiler defaults to batch classification. If a task explicitly inherits from the Actor class but lacks proper AsyncGenerator annotation, the compiler raises an error because this likely indicates a mistake. The error messages guide you toward fixing annotation issues rather than silently producing incorrect behavior.

```python
import inspect
from typing import get_origin
from collections.abc import AsyncGenerator

def detect_task_type(task):
    sig = inspect.signature(task.execute)
    return_annotation = sig.return_annotation

    if return_annotation != inspect.Signature.empty:
        origin = get_origin(return_annotation)
        if origin is AsyncGenerator:
            return TaskExecutionType.ACTOR

    return TaskExecutionType.BATCH
```

Validation follows type detection and performs several structural checks. The compiler verifies that all task IDs are unique, preventing ambiguous references. It checks that links reference valid source and target tasks, catching configuration errors early. For batch tasks, it validates that source tasks declare outputs and target tasks declare inputs, ensuring data can flow through the link. For actors, this input/output validation is skipped because actors use dynamic channels not declared in advance.

Link validation includes automatic mapping generation for actor connections. When you create a link between actors without specifying a mapping, the compiler generates a default mapping based on task names. This convenience reduces boilerplate while remaining explicit enough to understand the data flow. You can always provide an explicit mapping if the default doesn't match your needs.

Channel allocation happens after validation and only for links involving actors. The compiler creates channel configuration entries specifying the source task, target task, buffer size, and mapping information. These configurations serve as blueprints for creating actual asyncio.Queue instances at runtime. The compiler doesn't create queues itself because compilation happens before the asyncio event loop exists.

```python
def allocate_channels(workflow, task_types):
    channels = {}

    for link in workflow.links:
        source_type = task_types.get(link.source, TaskExecutionType.BATCH)
        target_type = task_types.get(link.target, TaskExecutionType.BATCH)

        # Create channel if either end is an actor
        if source_type == TaskExecutionType.ACTOR or target_type == TaskExecutionType.ACTOR:
            channel_id = f"{link.source}_to_{link.target}"
            channels[channel_id] = {
                'source': link.source,
                'target': link.target,
                'buffer_size': link.buffer_size,
                'mapping': link.mapping or {}
            }

    return channels
```

The compilation order matters for correctness. Type detection must happen first because subsequent passes need to know which tasks are actors. Validation comes next, including the step that auto-generates mappings for links without explicit mappings. Channel allocation must happen last because it depends on both type information and validated mappings. This ordering ensures each pass has the information it needs and that later passes see complete data from earlier passes.

Cycle detection provides safety for batch workflows while allowing cycles for actor workflows. Batch workflows must form directed acyclic graphs because task execution order follows dependency relationships. A cycle in dependencies creates an impossible situation where task A waits for B, B waits for C, and C waits for A. The compiler detects such cycles and rejects pure batch workflows containing them.

Actor workflows, by contrast, can contain cycles because actors don't wait for each other to complete before starting. Message passing through buffered channels allows actors to form feedback loops where A sends to B, B sends to C, and C sends back to A. These loops are not only legal but useful for implementing iterative algorithms and control systems. The compiler allows cycles when any task in the cycle is an actor.

The compiled workflow object returned by compilation provides high-level access to compilation results. It exposes methods like get_task_types returning the task type classifications, get_channels returning channel configurations, and get_dependency_graph returning the task dependency structure. These methods let the engine and other tools query compilation results without coupling to internal compilation details.

Compilation errors include detailed messages identifying the problem and suggesting fixes. If a link references a nonexistent task, the error message names the missing task and lists available tasks. If a batch task lacks required input declarations, the error explains which task needs inputs declared and why. These helpful errors accelerate development by making problems obvious and solutions clear.

The compiler design emphasizes simplicity and correctness over performance. Compilation happens once per workflow before execution, so compilation time doesn't affect runtime performance. The compiler performs thorough validation and generates complete metadata, trading some compilation cost for runtime efficiency and reliability. This tradeoff makes sense because workflows typically execute much longer than they take to compile.

Understanding compilation clarifies how high-level workflow definitions become executable systems. The compiler acts as a translator that understands both the declarative workflow language and the requirements of the execution engine. It fills in gaps like default mappings, performs safety checks like cycle detection, and organizes information like channel configurations. The result is an executable workflow representation that the engine can run efficiently and correctly.

The compilation model integrates with molexp's philosophy of automatic behavior based on type information. Rather than requiring explicit mode declarations or manual configuration, the system examines your code and infers what execution model to use. This inference reduces the conceptual burden of workflow development while maintaining precise semantics and predictable behavior. You write task definitions and link specifications, and the compiler handles the translation to executable form.
