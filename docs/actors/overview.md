# Actor Model in Molexp

The actor model provides a foundation for building concurrent, long-running computational workflows that process streams of data through message passing. Unlike traditional batch tasks that execute once and return results, actors represent continuous processing units that run indefinitely until they complete their work or are explicitly stopped.

An actor in molexp is a specialized task that returns an asynchronous generator instead of a dictionary. This fundamental difference in return type signals to the workflow compiler that the task requires different execution semantics. Where a batch task might load data, transform it, and return results in a single execution, an actor continuously processes incoming messages, maintains state across iterations, and communicates with other actors through explicit channels.

The design emerged from the need to model multi-scale computational workflows where different components operate at different time scales and need to exchange information asynchronously. Traditional workflow systems excel at orchestrating discrete computation steps with clear dependencies, but struggle when components need to run concurrently and communicate dynamically. The actor model addresses this by treating each computational unit as an independent entity with its own execution loop and communication channels.

Consider a molecular dynamics simulation workflow where one component generates conformations at a fast rate while another component analyzes those conformations using expensive quantum chemistry calculations. These components naturally operate at different speeds and need to communicate through buffered channels that handle backpressure. The actor model makes this pattern explicit and manageable.

Creating an actor begins by defining a configuration class using Pydantic. This configuration serves a dual purpose in molexp's design: it specifies the actor's parameters and also holds all runtime state. This config-as-state pattern simplifies persistence and enables hot reconfiguration, as we'll explore in detail later.

```python
from pydantic import BaseModel
from molexp.workflow.task import Actor
from collections.abc import AsyncGenerator

class SamplerConfig(BaseModel):
    threshold: float = 0.5
    samples_collected: int = 0
    max_samples: int = 100

class SamplerActor(Actor[SamplerConfig, dict]):
    config_type = SamplerConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        while self.config.samples_collected < self.config.max_samples:
            data = await ctx.receive('input')

            if data['score'] > self.config.threshold:
                self.config.samples_collected += 1
                await ctx.emit('output', data)

            yield

        ctx.set_result(self.task_id, {
            'samples_collected': self.config.samples_collected
        })
        return
```

The execute method's signature reveals several important characteristics. The return type annotation `AsyncGenerator[None, dict]` tells the compiler this is an actor. The method is declared async, allowing it to use await expressions for asynchronous operations. The context parameter provides access to communication channels and result storage.

Inside the execute method, actors follow a common pattern: loop until a termination condition is met, receive or process data, optionally emit results to other actors, and yield control back to the asyncio event loop. The yield statement is critical because it allows the event loop to switch between actors, preventing any single actor from monopolizing execution time.

Actors communicate through named channels established by workflow links. The await ctx.receive('input') call blocks until a message arrives on the input channel. This blocking behavior implements natural backpressure: if an actor receives data faster than it can process, the sending actor will eventually block when the channel buffer fills. Similarly, ctx.emit('output', data) sends a message to the output channel, blocking if the channel is full.

The stateless design principle guides actor implementation. Rather than storing state in instance variables, all state lives in the config object. This might seem unusual at first, but it provides several benefits. The actor's complete state is always visible and serializable through the config. Updates to config fields automatically persist when the workflow saves metadata. Most importantly for long-running computations, this design enables hot reconfiguration where parameter values can change while the actor runs.

When an actor completes its work, it uses ctx.set_result to store final results. Unlike batch tasks that return results directly, actors cannot use return values because Python's async generator protocol doesn't support returning values during iteration. The set_result method provides an alternative that integrates naturally with the workflow execution context.

Actors differ from batch tasks in several fundamental ways. Batch tasks execute once in dependency order, while actors run concurrently for as long as needed. Batch tasks communicate through input-output mappings resolved at compile time, while actors use dynamic message passing through channels. Batch tasks can safely share data structures, while actors must maintain isolation and communicate explicitly. These differences reflect the distinct computational patterns each model serves.

The workflow compiler detects actors automatically by inspecting the execute method's return type annotation. When it sees AsyncGenerator, it marks the task as an actor and allocates communication channels for any links involving that actor. This automatic detection means you don't need to manually register actors or configure execution modes. The type system itself carries the semantic information needed for correct compilation and execution.

Understanding the actor model's place in molexp's architecture helps clarify when to use it. Actors excel at continuous processing, stream transformation, filtering, monitoring, and any scenario where components run at different rates or need dynamic communication. They are less suitable for simple sequential computations where batch tasks provide clearer semantics and better performance. The hybrid execution model allows both patterns to coexist in the same workflow, letting you choose the right abstraction for each component.
