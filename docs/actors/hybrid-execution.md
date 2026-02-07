# Hybrid Execution Model

The hybrid execution model allows batch tasks and actors to coexist in the same workflow, with the workflow engine automatically selecting the appropriate execution strategy based on task types. This automatic mode selection means you don't manually configure execution modes or worry about compatibility between different task types.

Molexp workflows originally executed batch tasks sequentially using dependency ordering and parallel execution via ThreadPoolExecutor. This model works excellently for discrete computation steps with clear dependencies. Each task runs once, produces outputs, and those outputs feed into downstream tasks. The execution engine schedules tasks as their dependencies complete, maximizing parallelism while respecting ordering constraints.

Actors introduce different execution requirements. They run indefinitely rather than executing once. They communicate through asynchronous message passing rather than dependency resolution. They need an event loop to coordinate concurrent execution. These differences could have led to separate execution systems for batch and actor workflows, but maintaining two parallel systems creates complexity and limits flexibility.

The hybrid execution model unifies these approaches by detecting which execution strategy each workflow requires and applying it automatically. When a workflow contains only batch tasks, the engine uses the traditional fast path with ThreadPoolExecutor. When a workflow contains any actors, the engine switches to hybrid mode using asyncio to coordinate both batch and actor execution. This automatic selection happens during workflow compilation based on task type detection.

Type detection drives the automatic mode selection. The workflow compiler inspects each task's execute method signature to determine its return type. A return type annotation of dict indicates a batch task. A return type annotation of AsyncGenerator[None, dict] indicates an actor. The compiler records these classifications in the compiled workflow and uses them to choose the execution mode.

```python
# Batch task - returns dict
class DataLoader(Task[LoadConfig, dict]):
    config_type = LoadConfig

    def execute(self, ctx, **inputs) -> dict:
        data = load_data(self.config.path)
        return {'data': data}

# Actor - returns AsyncGenerator
class StreamProcessor(Actor[ProcessConfig, dict]):
    config_type = ProcessConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        while True:
            item = await ctx.receive('input')
            result = process(item)
            await ctx.emit('output', result)
            yield
```

The pure batch fast path provides optimal performance for traditional workflows. When the compiler detects no actors, execution proceeds through the existing batch execution logic. Tasks run in topologically sorted order with parallelism limited by dependencies and the ThreadPoolExecutor pool size. This path avoids asyncio overhead and uses straightforward threading for independent task execution.

Hybrid mode activates when any actor appears in the workflow. The engine creates an asyncio event loop and executes all tasks within that loop. Actors run as native coroutines, allowing natural use of async/await for message passing. Batch tasks, which are synchronous, execute via asyncio.to_thread to prevent blocking the event loop. This threading integration maintains backward compatibility while enabling actors to coexist with traditional tasks.

Channel management distinguishes hybrid execution from pure batch execution. During compilation, the compiler allocates channel configurations for links between actors. At runtime, the engine creates asyncio.Queue instances for these channels and registers them with the run context using the channel names from link mappings. Actors access channels through the context without needing to know about queue implementations or routing details.

```python
# Workflow with both batch and actor tasks
loader = DataLoader(path="/data/input.csv")
processor = StreamProcessor(transform="normalize")

workflow = Workflow.from_tasks(
    tasks=[loader, processor],
    links=[
        Link(source=loader, target=processor, mapping={'data': 'input'})
    ],
    name="hybrid_workflow"
)

# Engine automatically detects hybrid mode needed
engine = WorkflowEngine(workflow)
with run.start() as run_ctx:
    results = engine.execute(run_context=run_ctx)
```

Concurrent execution of multiple actors happens naturally in hybrid mode. The asyncio event loop schedules all actor coroutines concurrently, switching between them at await points. Actors blocked waiting for channel input don't prevent other actors from making progress. The event loop ensures fair scheduling and efficient use of available CPU time.

The execution model handles actor completion by waiting for all actor coroutines to finish before returning results. An actor signals completion by returning from its execute method. The engine collects return values via ctx.set_result calls made before the actor returns. Once all actors complete and all batch tasks finish, the engine returns the complete results dictionary.

Failure handling in hybrid mode propagates actor exceptions similarly to batch task failures. If an actor raises an exception, the engine catches it, records the failure in the run context, and continues executing other actors. This isolation prevents one failing actor from immediately crashing the entire workflow. Whether you want fail-fast or continued execution depends on your use case and can be controlled through error handling policies.

The automatic mode selection has implications for workflow design. You can freely mix batch and actor tasks without worrying about execution compatibility. A workflow might load data with a batch task, process it with multiple concurrent actors, and aggregate results with another batch task. The execution engine handles the mode transitions and ensures correct execution semantics.

Performance characteristics differ between pure batch and hybrid modes. Pure batch mode has lower overhead and more predictable performance because it uses simpler threading primitives. Hybrid mode adds asyncio overhead and requires careful management of blocking operations. For workflows dominated by batch tasks with only a few actors, the hybrid mode overhead is minimal. For workflows with many actors and complex message passing, the asyncio model provides better scalability than threading would offer.

Understanding the execution model helps in designing efficient workflows. Place computationally intensive work in batch tasks when it doesn't need continuous execution or message passing. Use actors for streaming data processing, long-running monitors, or components that need asynchronous communication. The hybrid model lets you choose the right abstraction for each component while maintaining a unified workflow definition.

The execution model integrates with molexp's broader architecture through clean interfaces. The workflow compiler produces execution plans annotating each task with its execution type. The workflow engine consumes these plans and dispatches work appropriately. The run context provides a uniform interface that works in both batch and hybrid modes, with message passing methods available in hybrid mode and ignored in pure batch mode.

Backward compatibility receives careful attention in the hybrid execution model. Existing pure batch workflows continue to execute through the fast path with unchanged behavior and performance. The addition of hybrid mode doesn't affect these workflows at all. Only workflows that explicitly include actor tasks engage the hybrid execution path, and even then, batch tasks within those workflows execute with their traditional semantics.

The design philosophy behind hybrid execution emphasizes automatic over explicit configuration. Rather than requiring users to declare execution modes or manually configure runtime environments, the system infers requirements from task definitions and handles the details transparently. This automation reduces cognitive load and prevents configuration errors while providing flexibility to evolve workflows from simple batch pipelines to complex actor systems without architectural changes.
