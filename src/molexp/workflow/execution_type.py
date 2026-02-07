"""Task execution type classification for hybrid workflow execution.

This module defines the TaskExecutionType enum used by WorkflowCompiler to classify
tasks and select appropriate execution strategies:

- **BATCH**: Traditional tasks that execute once and return results directly.
  Execute in dependency order using ThreadPoolExecutor.

- **ACTOR**: Continuous tasks that return AsyncGenerator and run indefinitely.
  Execute concurrently in asyncio event loop with message passing support.

The compiler detects execution types automatically by inspecting execute() method
return annotations:
    - `-> dict` → BATCH
    - `-> AsyncGenerator[None, dict]` → ACTOR

Example:
    Defining task types::

        # BATCH task
        class DataLoader(Task[LoadConfig, dict]):
            def execute(self, ctx, **inputs) -> dict:
                return {"data": load_data()}

        # ACTOR task
        class StreamProcessor(Actor[ProcessConfig, dict]):
            async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
                while True:
                    item = await ctx.receive('input')
                    yield
"""

from enum import Enum


class TaskExecutionType(Enum):
    """Classification of task execution modes.

    BATCH: Traditional task that executes once and returns a dict
    ACTOR: Continuous actor that returns AsyncGenerator and runs indefinitely
    """

    BATCH = "batch"
    ACTOR = "actor"
