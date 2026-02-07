# Complete Examples

Understanding the actor model through concrete examples illuminates how the pieces fit together in practice. These examples progress from simple single actors to complex multi-actor systems with feedback loops and hot reconfiguration.

## Simple Data Generator

The simplest actor generates data and sends it downstream without receiving input. This pattern appears frequently as workflow entry points where actors produce data from external sources or synthetic generation.

```python
from pydantic import BaseModel
from molexp.workflow.task import Actor
from molexp.workflow import Workflow, Link
from molexp.workflow.engine import WorkflowEngine
from molexp.workspace import Workspace
from collections.abc import AsyncGenerator
import asyncio

class GeneratorConfig(BaseModel):
    num_items: int = 10
    rate_hz: float = 1.0
    multiplier: float = 1.0

class GeneratorActor(Actor[GeneratorConfig, dict]):
    config_type = GeneratorConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        for i in range(self.config.num_items):
            data = {
                'id': i,
                'value': i * self.config.multiplier,
                'timestamp': asyncio.get_event_loop().time()
            }

            await ctx.emit('output', data)

            # Rate limiting
            await asyncio.sleep(1.0 / self.config.rate_hz)
            yield

        ctx.set_result(self.task_id, {'items_generated': self.config.num_items})
        return

# Create and run workflow
workspace = Workspace(root="/tmp/examples", name="Generator Example")
workspace.materialize()
project = workspace.create_project(name="actors", exist_ok=True)
experiment = project.create_experiment(name="simple", exist_ok=True)
run = experiment.create_run(parameters={'example': 'generator'})

generator = GeneratorActor(num_items=5, rate_hz=2.0, multiplier=10.0)
workflow = Workflow.from_tasks(tasks=[generator], links=[], name="generator")
engine = WorkflowEngine(workflow)

with run.start() as run_ctx:
    results = engine.execute(run_context=run_ctx)
    print(f"Generated {results[generator.task_id]['items_generated']} items")
```

This generator demonstrates several actor patterns. It maintains state through config fields (num_items, rate_hz). It controls its execution rate using asyncio.sleep. It yields after each item to allow other actors to run. It reports results via ctx.set_result before returning.

## Three-Actor Pipeline

A pipeline connects multiple actors in sequence, with data flowing from producers through transformers to consumers. This pattern handles streaming data processing where each stage operates independently.

```python
class ProducerConfig(BaseModel):
    items_to_produce: int = 20
    production_rate: float = 5.0

class ProducerActor(Actor[ProducerConfig, dict]):
    config_type = ProducerConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        for i in range(self.config.items_to_produce):
            await ctx.emit('output', {'id': i, 'raw_value': i})
            await asyncio.sleep(1.0 / self.config.production_rate)
            yield

        ctx.set_result(self.task_id, {'produced': self.config.items_to_produce})
        return


class ProcessorConfig(BaseModel):
    items_to_process: int = 20
    processing_factor: float = 2.0

class ProcessorActor(Actor[ProcessorConfig, dict]):
    config_type = ProcessorConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        for _ in range(self.config.items_to_process):
            item = await ctx.receive('input')
            processed = {
                'id': item['id'],
                'processed_value': item['raw_value'] * self.config.processing_factor
            }
            await ctx.emit('output', processed)
            yield

        ctx.set_result(self.task_id, {'processed': self.config.items_to_process})
        return


class CollectorConfig(BaseModel):
    items_to_collect: int = 20

class CollectorActor(Actor[CollectorConfig, dict]):
    config_type = CollectorConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        collected = []

        for _ in range(self.config.items_to_collect):
            item = await ctx.receive('input')
            collected.append(item)
            yield

        ctx.set_result(self.task_id, {
            'collected_count': len(collected),
            'items': collected
        })
        return


# Build and run pipeline
producer = ProducerActor(items_to_produce=10, production_rate=10.0)
processor = ProcessorActor(items_to_process=10, processing_factor=3.0)
collector = CollectorActor(items_to_collect=10)

workflow = Workflow.from_tasks(
    tasks=[producer, processor, collector],
    links=[
        Link(source=producer, target=processor, mapping={'output': 'input'}),
        Link(source=processor, target=collector, mapping={'output': 'input'})
    ],
    name="pipeline"
)

engine = WorkflowEngine(workflow)
with run.start() as run_ctx:
    results = engine.execute(run_context=run_ctx)

    collected_items = results[collector.task_id]['items']
    print(f"Collected {len(collected_items)} items")
    print(f"First item: {collected_items[0]}")
```

The pipeline demonstrates message passing between actors with different processing rates. The producer generates items at a configured rate. The processor transforms them. The collector accumulates results. Buffer sizes in links (defaulting to 100) absorb rate mismatches without blocking.

## Feedback Loop

Feedback loops connect actors in cycles, allowing iterative refinement or control systems. The actor model permits these cycles because actors communicate through buffered channels rather than blocking on completion.

```python
class IteratorConfig(BaseModel):
    max_iterations: int = 10
    current_iteration: int = 0
    converged: bool = False

class IteratorActor(Actor[IteratorConfig, dict]):
    config_type = IteratorConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        while not self.config.converged and self.config.current_iteration < self.config.max_iterations:
            # Try to receive feedback
            try:
                feedback = await asyncio.wait_for(
                    ctx.receive('feedback'),
                    timeout=0.1
                )
                value = feedback['value']
            except asyncio.TimeoutError:
                value = 0.0

            # Compute new value
            new_value = value + 1.0
            self.config.current_iteration += 1

            # Check convergence
            if new_value > 8.0:
                self.config.converged = True

            # Send forward
            await ctx.emit('output', {
                'iteration': self.config.current_iteration,
                'value': new_value
            })

            yield

        ctx.set_result(self.task_id, {
            'iterations': self.config.current_iteration,
            'converged': self.config.converged
        })
        return


# Create feedback loop: A → B → A
actor_a = IteratorActor(max_iterations=5)
actor_b = IteratorActor(max_iterations=5)

workflow = Workflow.from_tasks(
    tasks=[actor_a, actor_b],
    links=[
        Link(source=actor_a, target=actor_b, mapping={'output': 'feedback'}),
        Link(source=actor_b, target=actor_a, mapping={'output': 'feedback'})
    ],
    name="feedback"
)

engine = WorkflowEngine(workflow)
with run.start() as run_ctx:
    results = engine.execute(run_context=run_ctx)
```

The feedback loop uses timeout on receive to handle the bootstrap problem where no initial feedback exists. Actors track iteration count and convergence in their config. The cycle continues until convergence or max iterations.

## Hot Reconfiguration

Hot reconfiguration enables parameter adjustment during execution, useful for adaptive algorithms and interactive tuning.

```python
class AdaptiveSamplerConfig(BaseModel):
    threshold: float = 0.5
    samples_collected: int = 0
    samples_rejected: int = 0
    items_processed: int = 0

class AdaptiveSamplerActor(Actor[AdaptiveSamplerConfig, dict]):
    config_type = AdaptiveSamplerConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        max_items = 100

        for i in range(max_items):
            # Simulate scoring
            score = (i % 10) / 10.0

            # Check current threshold (enables hot reconfig)
            current_threshold = self.config.threshold

            self.config.items_processed += 1

            if score > current_threshold:
                self.config.samples_collected += 1
                print(f"[{self.config.items_processed}] ACCEPT: score={score:.2f} > threshold={current_threshold:.2f}")
            else:
                self.config.samples_rejected += 1
                print(f"[{self.config.items_processed}] REJECT: score={score:.2f} <= threshold={current_threshold:.2f}")

            await asyncio.sleep(0.1)
            yield

        ctx.set_result(self.task_id, {
            'items_processed': self.config.items_processed,
            'samples_collected': self.config.samples_collected,
            'samples_rejected': self.config.samples_rejected
        })
        return


# Create workflow
sampler = AdaptiveSamplerActor(threshold=0.5)
workflow = Workflow.from_tasks(tasks=[sampler], links=[], name="adaptive")
engine = WorkflowEngine(workflow)

# Run in background and update config
import threading

def run_workflow():
    with run.start() as run_ctx:
        engine.execute(run_context=run_ctx)

workflow_thread = threading.Thread(target=run_workflow, daemon=True)
workflow_thread.start()

# Update threshold during execution
import time
time.sleep(2)
print("\n=== Updating threshold to 0.3 ===\n")
engine.update_actor_config(sampler.task_id, {'threshold': 0.3})

time.sleep(2)
print("\n=== Updating threshold to 0.7 ===\n")
engine.update_actor_config(sampler.task_id, {'threshold': 0.7})

workflow_thread.join()
```

The adaptive sampler checks self.config.threshold on each iteration, allowing threshold updates to take effect immediately. The actor prints acceptance decisions showing how threshold changes affect behavior in real time.

These examples demonstrate core actor patterns: data generation, pipeline processing, feedback loops, and hot reconfiguration. Each example builds on config-as-state design, message passing through channels, and yielding for concurrency. Combining these patterns lets you build sophisticated concurrent workflows that adapt to changing conditions and process data streams efficiently.
