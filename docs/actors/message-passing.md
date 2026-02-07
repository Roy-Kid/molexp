# Message Passing and Channels

Message passing provides the communication mechanism that allows actors to exchange data while maintaining independence and isolation. Rather than sharing memory or calling each other's methods directly, actors send messages through named channels that provide buffering, backpressure control, and explicit communication patterns.

A channel in molexp is implemented as an asyncio.Queue with a configurable maximum size. When an actor emits a message to a channel, the message enters the queue. Another actor receiving from that channel retrieves messages in first-in-first-out order. The queue sits between actors as an independent entity that neither actor controls directly, creating a clear separation of concerns and enabling the actors to operate at different speeds.

The design philosophy behind explicit channels reflects lessons learned from building complex computational workflows. Implicit communication through shared state leads to subtle bugs and tight coupling between components. When one component modifies shared data, other components see those changes immediately, making it difficult to reason about system behavior or debug issues. By contrast, channels make communication explicit and visible. You can inspect what messages flow through a channel, monitor queue depths, and understand the data flow by examining the workflow graph.

Channels also provide natural backpressure handling. When a fast producer generates data faster than a slow consumer can process it, the channel buffer fills up. Once full, the producer blocks on emit until the consumer drains messages from the queue. This automatic throttling prevents memory exhaustion and creates a self-regulating system where components naturally balance their processing rates. Without this mechanism, a fast producer could overwhelm system memory or force the consumer to drop messages.

Message passing in molexp uses two primary operations: emit and receive. Both are asynchronous operations that may block depending on channel state. The emit operation sends a message to a named channel, and the receive operation retrieves a message from a named channel.

```python
class ProducerActor(Actor[ProducerConfig, dict]):
    config_type = ProducerConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        for i in range(self.config.num_items):
            data = {
                'id': i,
                'value': self.generate_value(i),
                'timestamp': time.time()
            }

            await ctx.emit('output', data)
            yield

        ctx.set_result(self.task_id, {'items_sent': self.config.num_items})
        return
```

The emit operation takes two arguments: the channel name and the message to send. Channel names are strings that connect to workflow links through the link's mapping configuration. The message can be any Python object, though in practice, dictionaries work well because they serialize naturally and provide clear structure. The await keyword indicates this operation may block if the channel buffer is full.

Receiving messages follows a similar pattern but introduces the possibility of waiting indefinitely for data. An actor that receives from a channel must be prepared to block until a message arrives. This blocking behavior is intentional and desirable: it means actors don't need to poll or busy-wait for data. The asyncio event loop handles the blocking efficiently, allowing other actors to make progress while one waits for input.

```python
class ConsumerActor(Actor[ConsumerConfig, dict]):
    config_type = ConsumerConfig

    async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        processed = []

        for _ in range(self.config.num_expected):
            data = await ctx.receive('input')

            result = self.process(data)
            processed.append(result)

            if len(processed) % 10 == 0:
                await ctx.emit('status', {'processed': len(processed)})

            yield

        ctx.set_result(self.task_id, {
            'processed_count': len(processed),
            'results': processed
        })
        return
```

Connecting actors through channels requires defining links in the workflow. A link specifies the source actor, target actor, and a mapping that connects output channel names to input channel names. The mapping dictionary's keys represent channel names used by the source actor in emit calls, while values represent channel names used by the target actor in receive calls.

```python
from molexp.workflow import Workflow, Link

producer = ProducerActor(num_items=100)
consumer = ConsumerActor(num_expected=100)

workflow = Workflow.from_tasks(
    tasks=[producer, consumer],
    links=[
        Link(
            source=producer,
            target=consumer,
            mapping={'output': 'input'},
            buffer_size=50
        )
    ],
    name="producer_consumer"
)
```

The link configuration includes a buffer_size parameter that sets the maximum number of messages the channel can hold. Larger buffers allow more decoupling between producer and consumer speeds but use more memory. Smaller buffers enforce tighter coupling and faster backpressure responses. Choosing an appropriate buffer size depends on the expected rate mismatch between actors and the memory constraints of your system. A buffer size of 100 works well as a default, allowing moderate speed differences without excessive memory use.

Channel routing in molexp follows a logical naming convention rather than physical connections. Actors use human-readable channel names like 'output', 'input', 'results', or 'errors'. The workflow compiler translates these logical names to physical asyncio.Queue instances using the link mappings. This indirection provides flexibility: you can connect the same actor to different downstream consumers by changing link mappings without modifying the actor's code.

Multiple actors can send to the same channel, and multiple actors can receive from the same channel, though in practice these patterns are less common than one-to-one connections. When multiple actors receive from a channel, each message goes to only one receiver in a round-robin fashion. When multiple actors send to a channel, messages interleave in the order they arrive. These semantics match asyncio.Queue behavior and provide reasonable defaults for most scenarios.

Error handling in message passing deserves special attention. If an actor attempts to emit to or receive from a channel that doesn't exist, molexp raises a KeyError with a helpful message listing the available channels. This fail-fast behavior catches configuration errors during testing rather than allowing them to manifest as silent failures or subtle bugs. The error messages include enough context to quickly identify and fix connection issues.

Monitoring channel health provides insights into workflow behavior and helps diagnose performance issues. The get_channel_depths method returns a dictionary mapping channel names to their current queue sizes. High queue depths indicate a consumer can't keep up with its producer. Zero depths might indicate a consumer is starving for data. Monitoring these metrics during execution helps identify bottlenecks and tune buffer sizes or actor implementations.

```python
# Inside an actor or monitoring component
depths = ctx.get_channel_depths()
for channel_name, depth in depths.items():
    if depth > buffer_size * 0.9:
        logger.warning(f"Channel {channel_name} nearly full: {depth} messages")
```

The channel abstraction integrates with molexp's broader workflow execution model. During compilation, the workflow compiler detects actor tasks and allocates channel configurations for links involving those actors. At runtime, the workflow engine creates asyncio.Queue instances according to these configurations and registers them with the run context using the mapped channel names. Actors then access channels through the context without needing to know about the underlying queue implementations or routing details.

Understanding message passing mechanics illuminates the actor model's strengths and limitations. Channels provide explicit, traceable communication with automatic backpressure and clear data flow. They add some overhead compared to direct method calls or shared memory, but this overhead buys significant benefits in terms of modularity, testability, and system robustness. For workflows involving concurrent processing of streaming data, these trade-offs strongly favor the channel-based approach.
