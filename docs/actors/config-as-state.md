# Config-as-State Pattern

The config-as-state pattern represents a design choice that simplifies actor implementation by storing all runtime state in the actor's configuration object. Rather than maintaining separate instance variables for state and configuration parameters, actors keep everything in a single Pydantic model that serves both purposes.

This pattern emerged from experience with long-running computational workflows where the ability to inspect, persist, and modify actor state becomes essential. Traditional object-oriented design encourages separating configuration from state, but this separation creates challenges for distributed or long-running systems. Configuration parameters might live in one data structure, runtime state in another, and the relationship between them becomes implicit. When you need to checkpoint an actor's state or understand why it behaves a certain way, you must gather information from multiple sources.

By placing all state in the config, molexp makes the actor's complete state visible and serializable through a single object. This visibility simplifies debugging because you can inspect actor.config at any point and see everything that influences the actor's behavior. The config object provides a complete snapshot of the actor's state without hidden dependencies or implicit state stored in closures or private variables.

The pattern also enables automatic persistence without explicit checkpoint code. Molexp's workflow system already serializes task configurations to save workflow metadata. Since the config contains all state, serializing the config automatically captures the actor's complete state. When you resume a workflow, reconstructing actors from their saved configs restores them to their previous state. This automatic persistence eliminates the need for separate checkpoint and restore mechanisms that might drift out of sync with the actual state representation.

Implementing the config-as-state pattern requires thinking differently about state management. Instead of creating instance variables in the actor class, you add fields to the config model. Instead of updating instance variables, you update config fields. This shift feels natural once you recognize the config as the single source of truth for all actor state.

```python
from pydantic import BaseModel
from molexp.workflow.task import Actor
from collections.abc import AsyncGenerator

class ProcessorConfig(BaseModel):
    # Configuration parameters
    threshold: float = 0.5
    batch_size: int = 10

    # Runtime state
    items_processed: int = 0
    batches_completed: int = 0
    current_batch: list = []

class ProcessorActor(Actor[ProcessorConfig, dict]):
    config_type = ProcessorConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        while self.config.items_processed < 1000:
            item = await ctx.receive('input')

            # Update state in config
            self.config.items_processed += 1
            self.config.current_batch.append(item)

            # Process batch when full
            if len(self.config.current_batch) >= self.config.batch_size:
                self.process_batch(self.config.current_batch)
                self.config.batches_completed += 1
                self.config.current_batch = []

            yield

        ctx.set_result(self.task_id, {
            'items_processed': self.config.items_processed,
            'batches_completed': self.config.batches_completed
        })
        return

    def process_batch(self, batch):
        # Processing logic that reads from config
        filtered = [item for item in batch if item['score'] > self.config.threshold]
        return filtered
```

The config model mixes configuration parameters like threshold and batch_size with runtime state like items_processed and batches_completed. This mixing might seem to violate separation of concerns, but it serves a practical purpose: it creates a single, coherent representation of the actor's complete state that combines both what the actor was configured to do and what it has done so far.

Pydantic validation applies to all config fields, including state fields. This means you can enforce constraints on state values just as you would for configuration parameters. For example, you might require that items_processed is non-negative or that current_batch doesn't exceed a maximum length. These constraints help catch bugs where state updates violate invariants.

The stateless design principle guides how actors interact with their config. An actor should not cache values from the config in local variables or depend on state that isn't reflected in the config. Every piece of information the actor needs should live in the config and be accessed directly from there. This discipline ensures that the config truly represents the complete state and that changes to the config immediately affect actor behavior.

```python
# Good: Read from config each iteration
async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
    while self.config.count < self.config.max_count:
        # Check current config value
        if self.config.enabled:
            data = await ctx.receive('input')
            self.process(data)
            self.config.count += 1
        yield

# Avoid: Caching config values
async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
    max_count = self.config.max_count  # Cached value won't see updates
    while self.config.count < max_count:
        # ...
```

This reading-from-config-every-time pattern enables hot reconfiguration, which we'll explore in detail in the next document. When you update an actor's config while it runs, the actor sees the new values on its next iteration. If the actor had cached config values, those updates wouldn't take effect until the actor restarted.

Choosing what to put in the config versus what to compute on demand requires some judgment. State that persists across iterations belongs in the config: counters, accumulators, flags, recently processed items. Temporary variables used within a single iteration don't belong in the config: loop counters, intermediate results, local computations. The distinguishing factor is whether the information needs to survive across yield points.

Some state naturally lives outside the config, particularly state managed by the asyncio runtime like channel queues or tasks. The actor doesn't own these resources, so it shouldn't try to include them in the config. The workflow engine manages channels and provides access to them through the context. Similarly, any file handles, network connections, or other external resources should be opened when needed and closed promptly rather than stored in the config.

The config-as-state pattern integrates naturally with Python's type system and Pydantic's validation. Type hints on config fields document what kind of state the actor maintains. Pydantic's validators catch type errors early. The model's json schema provides a machine-readable description of the actor's state structure, useful for tooling or documentation generation.

Consider the implications for testing. When testing an actor, you can construct it with specific state by passing that state as config fields. You don't need separate setup methods or state injection mechanisms. The config provides a natural way to initialize actors in any state you want to test.

```python
# Test actor starting from non-zero state
def test_actor_resume_from_checkpoint():
    actor = ProcessorActor(
        items_processed=500,
        batches_completed=50,
        current_batch=[]
    )

    # Actor resumes from this state
    # ...
```

The pattern does introduce some constraints. Config fields must be serializable, which rules out certain types like open file handles or lambda functions. Complex nested structures work but can become unwieldy. In practice, keeping state relatively flat and simple leads to more maintainable actors. If you find yourself building deep nested structures in the config, consider whether some of that complexity could move into helper methods or separate components.

Comparing this approach to alternatives clarifies its benefits. Traditional object-oriented design might use instance variables for state and a separate config object for parameters. This separation requires explicit checkpoint and restore methods that must be kept in sync with the actual state representation. Functional approaches might avoid mutable state entirely, passing state explicitly between function calls. This purity has theoretical appeal but adds verbosity and makes state evolution harder to track. The config-as-state pattern strikes a pragmatic balance: state is mutable for implementation simplicity but contained in a single, visible, serializable object for system simplicity.

Understanding this pattern helps in designing effective actors. Think of the config as the actor's complete memory. Everything the actor needs to remember goes there. Configuration parameters and runtime state coexist in the same structure because they're both part of what defines the actor's current situation. This mental model leads to cleaner implementations and systems that are easier to reason about, debug, and extend.
