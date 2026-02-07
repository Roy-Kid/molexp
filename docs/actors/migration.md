# Migration Guide

Understanding when and how to adopt actors in existing workflows requires recognizing the patterns that benefit from the actor model versus those better served by traditional batch tasks. This guide helps you identify migration opportunities and execute the transition smoothly.

## When to Use Actors vs Batch Tasks

Batch tasks excel at discrete computation steps with clear inputs and outputs. A task that loads data from a file, transforms it, and returns results works perfectly as a batch task. The execution happens once, all inputs are available at start, and outputs appear at completion. Dependencies between batch tasks create natural sequencing, and the workflow engine handles scheduling automatically.

Actors suit continuous processing where data arrives over time or components need ongoing coordination. Consider actors when you need streaming data processing, when components run at different speeds and need buffering, when you want feedback loops or iterative refinement, or when you need to adjust parameters without stopping the workflow. The actor model makes these patterns explicit and manageable.

A molecular dynamics simulation illustrates the distinction. Batch tasks handle initial system setup, loading force fields, and final analysis. Actors handle the continuous simulation loop, streaming conformation sampling, and adaptive parameter adjustment based on observed statistics. Mixed workflows using both patterns leverage each model's strengths appropriately.

## Identifying Candidates for Migration

Existing batch tasks that run in loops or need periodic execution often translate well to actors. If you find yourself writing tasks that never truly complete or that artificially segment continuous processes into discrete chunks, actors provide a more natural expression.

Look for workflows where you chain batch tasks through artificial staging. When task A produces intermediate results that task B consumes to produce more intermediate results that task C needs, and this happens many times, an actor pipeline might better capture the streaming nature of the computation. The batch version requires checkpointing between stages and complex resumption logic. The actor version processes data continuously without artificial boundaries.

Components that need runtime parameter adjustment benefit from actor migration. Batch tasks receive configuration at construction and cannot change during execution. Actors with config-as-state pattern support hot reconfiguration, enabling adaptive algorithms and interactive tuning during long-running computations.

## Migration Process

Converting a batch task to an actor follows a systematic pattern. Start by examining the batch task's execute method to understand its computation structure. Identify what represents configuration versus what represents runtime state. Extract runtime state into config fields, applying the config-as-state pattern.

```python
# Original batch task
class DataProcessor(Task[ProcessorConfig, dict]):
    config_type = ProcessorConfig

    def execute(self, ctx, **inputs) -> dict:
        results = []
        for item in inputs['data']:
            if item['value'] > self.config.threshold:
                results.append(self.process(item))
        return {'results': results}


# Migrated to actor
class ProcessorConfig(BaseModel):
    threshold: float = 0.5
    items_processed: int = 0  # Runtime state in config

class DataProcessorActor(Actor[ProcessorConfig, dict]):
    config_type = ProcessorConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        while True:
            try:
                item = await asyncio.wait_for(ctx.receive('input'), timeout=1.0)
            except asyncio.TimeoutError:
                break

            if item['value'] > self.config.threshold:
                result = self.process(item)
                await ctx.emit('output', result)

            self.config.items_processed += 1
            yield

        ctx.set_result(self.task_id, {
            'items_processed': self.config.items_processed
        })
        return
```

The migration transforms the method signature from synchronous returning dict to async returning AsyncGenerator. Input comes from ctx.receive instead of function parameters. Output goes through ctx.emit instead of return values. The processing loop continues until no more input arrives rather than processing a predetermined batch.

Workflow link definitions require updating when migrating tasks to actors. Batch task links use input/output name mappings based on declared inputs and outputs. Actor links use channel name mappings that connect emit calls to receive calls. The channel names can be any strings that make sense for your data flow.

```python
# Batch workflow links
Link(source=loader, target=processor, mapping={'data': 'data'})

# Actor workflow links
Link(
    source=loader_actor,
    target=processor_actor,
    mapping={'output': 'input'},
    buffer_size=100
)
```

Adding buffer_size configuration helps tune the backpressure behavior for actor connections. Start with the default of 100 and adjust based on observed behavior. Increase buffer_size if you see frequent blocking on emit. Decrease it if you want tighter coupling between producer and consumer speeds.

## Handling Common Migration Challenges

State management presents the primary migration challenge. Batch tasks often accumulate results in local variables and return them at completion. Actors need to stream results incrementally rather than accumulating everything. This shift requires rethinking how data flows through the workflow.

Consider a batch task that processes a thousand items and returns all results at once. The actor version processes items continuously, emitting results as they complete. Downstream consumers receive results incrementally, enabling pipelining and better resource utilization. This change requires downstream tasks to also handle streaming data or accumulate results themselves.

Termination conditions need explicit handling in actors. Batch tasks complete when their code finishes executing. Actors run indefinitely until they explicitly return. You need to decide what signals completion: processing a fixed number of items, receiving a sentinel value indicating end of data, or reaching a convergence criterion. Make this logic explicit in the actor's loop condition.

Error handling differs between batch and actor execution. Batch tasks that raise exceptions fail the task and stop the workflow. Actors that raise exceptions fail but other actors continue executing. This isolation provides robustness but requires careful thought about error propagation and recovery strategies.

Testing strategies shift when migrating to actors. Batch tasks can be tested by constructing them with test config and calling execute with test inputs. Actor testing requires mocking the run context to provide test channels and verify emit/receive behavior. The async nature adds complexity but tools like pytest-asyncio help manage this.

## Incremental Migration Strategy

You don't need to migrate entire workflows at once. The hybrid execution model supports mixing batch and actor tasks in the same workflow, enabling incremental migration where you convert tasks one at a time while maintaining a working system.

Start by identifying the highest-value migration candidate. Perhaps one task has awkward batch semantics or would benefit from streaming processing. Convert just that task to an actor, adjust its links, and test thoroughly. Once working, move to the next candidate. This incremental approach reduces risk and allows learning from each migration before proceeding.

Some workflows may never fully migrate to actors and that's acceptable. A workflow might use actors for streaming data processing but retain batch tasks for initialization and finalization. The hybrid model accommodates these mixed workflows naturally without forcing uniform abstraction across all components.

## Performance Considerations

Actors add overhead compared to batch tasks through asyncio event loop management and message passing through queues. For very small, fast tasks, batch execution may perform better. For longer-running tasks or those that benefit from concurrent execution, actors provide comparable or better performance while offering more flexibility.

Measure performance before and after migration to understand the impact on your specific workload. Use profiling to identify bottlenecks. The actor model shines for I/O-bound or coordination-heavy workflows but may not help computationally bound tasks that don't benefit from concurrency.

Memory usage patterns change with actors. Batch tasks hold data for their execution duration then release it. Actors hold data in channel buffers, creating different memory pressure. Monitor queue depths and adjust buffer sizes to balance memory use with throughput requirements.

## Backward Compatibility

The actor model implementation maintains backward compatibility with existing batch workflows. Pure batch workflows continue executing through the fast path with unchanged behavior. Existing batch tasks don't require modification unless you actively choose to convert them to actors.

Dependencies on workflow serialization, run context interfaces, and result structures remain compatible. The hybrid execution model extends the system without breaking existing patterns. This compatibility allows exploration of actors without committing to wholesale migration.

Understanding migration patterns and challenges prepares you to adopt actors where they provide value while retaining batch tasks where they work well. The goal is not to convert everything to actors but to use each abstraction where it fits best, creating workflows that express your computation naturally and execute efficiently.
