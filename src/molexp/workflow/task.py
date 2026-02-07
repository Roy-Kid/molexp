"""Base Task abstraction for all executable units.

This module provides the fundamental building blocks for workflow execution:

1. **Task**: Base class for batch-style tasks that execute once and return results.
   Tasks support local execution via execute() and external submission via submit().

2. **Actor**: Specialized task for continuous concurrent execution using the actor model.
   Actors run indefinitely, communicate via message passing, and store all state in config.

The module implements a config-as-state pattern where task configuration doubles as
runtime state, enabling hot reconfiguration and automatic persistence.

Example:
    Batch task::

        class MyTask(Task[MyConfig, dict]):
            config_type = MyConfig

            def execute(self, ctx, **inputs) -> dict:
                return {"result": self.config.value * 2}

    Actor task::

        class MyActor(Actor[MyConfig, dict]):
            config_type = MyConfig

            async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
                while self.config.running:
                    data = await ctx.receive('input')
                    await ctx.emit('output', data * 2)
                    yield
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Any, Generic, Type, TypeVar, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from molq.resources import JobSpec
    # JobHandle type - will be defined based on molq's actual return type
    JobHandle = Any

# Type variables for configuration and output
CfgT = TypeVar("CfgT")  # Removed BaseModel bound for flexibility
OutT = TypeVar("OutT")


class Task(Generic[CfgT, OutT], ABC):
    """Base abstraction for all executable units in workflows.

    A Task represents a single unit of computation that:
    - Has a unique identifier (auto-generated)
    - Accepts typed inputs via execute()
    - Produces typed outputs
    - Is configured via a Pydantic BaseModel config
    - Can declare inputs and outputs for validation
    - Can be tagged with a phase for workflow filtering
    - Can be submitted to external execution backends via submit()

    Tasks support two execution paths:
    1. Local execution: execute() returns dict directly
    2. External submission: submit() yields JobSpec, receives JobHandle, returns dict

    The execution path is determined by the `submittable` attribute.

    Attributes:
        config_type: Configuration class (Pydantic BaseModel)
        inputs: Optional dict mapping input names to types
        outputs: Optional dict mapping output names to types
        replayable: Whether task results can be replayed (default True)
        phase: Optional phase tag for workflow filtering (default None).
               Subclasses can set this to enable phase-based execution control.
        submittable: Optional backend name for external submission (default None).
                     If None, task executes locally via execute().
                     If set (e.g., "molq"), task submits via submit() method.
                     The backend name must match a registered submitor in WorkflowEngine.
        task_id: Unique identifier (auto-generated)
        config: Configuration instance (set at construction)
    """

    # Class attributes - must be overridden by subclasses
    config_type: Type[CfgT]
    inputs: dict[str, type] = {}
    outputs: dict[str, type] = {}
    replayable: bool = True
    phase: str | None = None
    submittable: str | None = None

    def __init__(self, **config_kwargs: Any) -> None:
        """Initialize task with static configuration.

        Args:
            **config_kwargs: Configuration parameters

        Raises:
            ValueError: If config_kwargs don't match the config schema
        """
        # Auto-generate unique task ID
        self.task_id = f"{self.__class__.__name__}_{uuid4().hex[:8]}"

        # Validate config type
        if not hasattr(self, "config_type"):
            raise ValueError(
                f"{self.__class__.__name__} must define a Pydantic config_type"
            )
        if not isinstance(self.config_type, type) or not issubclass(
            self.config_type, BaseModel
        ):
            raise ValueError(
                f"{self.__class__.__name__}.config_type must be a Pydantic BaseModel"
            )

        self.config: CfgT = self.config_type(**config_kwargs)

    @abstractmethod
    def execute(self, ctx: Any | None = None, **inputs: Any) -> dict[str, Any]:
        """Execute the task locally with given named inputs.

        This is the default execution method for tasks with submittable=None.
        Configuration is available via self.config.

        Args:
            ctx: Optional runtime context (e.g., RunContext) for accessing
                 directories, logging, asset registration, etc.
            **inputs: Named input values matching the task's input schema

        Returns:
            Dictionary of named outputs matching the task's output schema.
        """
        raise NotImplementedError

    def submit(self, ctx: Any | None = None, **inputs: Any) -> Generator["JobSpec", int, dict[str, Any]]:
        """Submit task to external execution backend via generator protocol.

        This method is called for tasks with submittable != None.
        It should yield JobSpec objects which will be submitted via the configured
        submitor backend. The generator receives job IDs (int) back and
        eventually returns task outputs.

        Generator Protocol:
            - Yield: JobSpec objects to submit
            - Receive: int (job ID) for submitted jobs
            - Return: dict of task outputs (same structure as execute())

        Args:
            ctx: Optional runtime context
            **inputs: Named input values matching the task's input schema

        Yields:
            JobSpec: Job specification to submit

        Receives:
            int: Job ID assigned by the scheduler

        Returns:
            Dictionary of named outputs matching the task's output schema.

        Raises:
            NotImplementedError: If task has submittable set but doesn't implement this method

        Example:
            >>> def submit(self, ctx, **inputs):
            ...     job_spec = JobSpec(
            ...         execution=ExecutionSpec(command="python train.py"),
            ...         resources=ResourceSpec(gpu_count=1)
            ...     )
            ...     job_id = yield job_spec
            ...     return {"job_id": job_id}
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has submittable='{self.submittable}' "
            f"but does not implement submit() method. Either implement submit() "
            f"or set submittable=None to use execute() instead."
        )

    def __call__(self, ctx: Any | None = None, **inputs: Any) -> dict[str, Any]:
        """Callable interface - executes with stored configuration.

        This allows tasks to be called directly from Python code:

            task = MyTask(param1="value", param2=42)
            result = task(input_value=10, ctx=ctx)

        Args:
            ctx: Optional runtime context
            **inputs: Named input values

        Returns:
            Dictionary of named outputs
        """
        return self.execute(ctx=ctx, **inputs)

    def dump(self) -> dict[str, Any]:
        """Serialize task configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return self.config.model_dump()
    
    def compute(self, ctx: Any | None = None, **inputs: Any) -> dict[str, Any]:
        """Alias for execute() for API consistency with molpy.Compute.
        
        This method delegates to execute() and is provided for API consistency.
        
        Args:
            ctx: Optional runtime context
            **inputs: Named input values
            
        Returns:
            Dictionary of named outputs
        """
        return self.execute(ctx=ctx, **inputs)

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get JSON schema for this task's configuration.

        Returns:
            JSON schema dictionary, or empty dict if no schema available
        """
        if hasattr(cls, 'config_type') and issubclass(cls.config_type, BaseModel):
            return cls.config_type.model_json_schema()
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.task_id!r}, config={self.config!r})"


class Actor(Task[CfgT, OutT], ABC):
    """Base class for continuous concurrent execution using the Actor model.

    Actor semantics:
    - Independent execution unit with isolated state
    - Communicates via explicit message passing (emit/receive)
    - Runs indefinitely until stopped or completes naturally
    - No shared state between actors
    - **Stateless design**: All state stored in `config` (config = state)

    Framework provides ONLY this base class; users must implement domain-specific actors.
    Actors use async generators for execution and can dynamically emit/receive messages
    on any channel without pre-declaration.

    **Config-as-State Pattern:**
    Store all actor state in the config field. This enables:
    - Automatic persistence via workflow metadata serialization
    - Hot reconfiguration by updating config
    - Easy state inspection and debugging

    Example:
        >>> class SamplerConfig(BaseModel):
        ...     threshold: float = 0.1
        ...     samples_collected: int = 0  # State in config
        ...
        >>> class SamplerActor(Actor[SamplerConfig, dict]):
        ...     config_type = SamplerConfig
        ...
        ...     async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
        ...         while True:
        ...             data = await ctx.receive('input')
        ...             if data['score'] > self.config.threshold:
        ...                 self.config.samples_collected += 1
        ...                 await ctx.emit('output', data)
        ...             yield

    Attributes:
        Same as Task. State should be stored in config, not as instance variables.
    """

    @abstractmethod
    def execute(self, ctx: Any | None = None, **inputs: Any) -> AsyncGenerator[None, dict[str, Any]]:
        """Execute actor continuously.

        Actors override this method to return an AsyncGenerator instead of a dict.
        The generator runs indefinitely, using ctx.receive() to consume messages
        and ctx.emit() to send messages. Must yield periodically to allow asyncio
        task switching.

        **Stateless Pattern:** Store all state in self.config, check config on each
        iteration for hot reconfiguration.

        Args:
            ctx: RunContext providing receive() and emit() for message passing
            **inputs: Initial inputs (typically empty for actors)

        Yields:
            None (control flow, allows asyncio switching)

        Returns:
            Use ctx.set_result(task_id, result) to return final results

        Example:
            >>> async def execute(self, ctx, **inputs) -> AsyncGenerator[None, dict]:
            ...     while self.config.samples_collected < self.config.max_samples:
            ...         data = await ctx.receive('data')
            ...         if data['score'] > self.config.threshold:  # Check config
            ...             self.config.samples_collected += 1
            ...             await ctx.emit('results', data)
            ...         yield  # Required for asyncio
            ...
            ...     ctx.set_result(self.task_id, {'count': self.config.samples_collected})
            ...     return
        """
        raise NotImplementedError



# Serialization model
class TaskConfig(BaseModel):
    """Serializable task configuration.

    This model represents a task's configuration for serialization purposes.
    It stores the task ID, task type (class name), and the configuration data.
    """

    task_id: str
    task_type: str
    config: dict[str, Any]
    status: str = "pending"
    phase: str | None = None
