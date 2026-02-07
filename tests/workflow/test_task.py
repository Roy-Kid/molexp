"""Tests for Actor base class and Task abstraction."""

import asyncio
import pytest
from collections.abc import AsyncGenerator
from pydantic import BaseModel, ValidationError

from molexp.workflow.task import Task, Actor, TaskConfig


# Test configurations
class SimpleConfig(BaseModel):
    """Simple config for testing."""
    value: int = 10
    name: str = "test"


class StatefulConfig(BaseModel):
    """Config with state fields for config-as-state pattern."""
    threshold: float = 0.5
    count: int = 0
    items_processed: int = 0


# Test tasks
class SimpleTask(Task[SimpleConfig, dict]):
    """Simple batch task for testing."""
    config_type = SimpleConfig

    def execute(self, ctx=None, **inputs) -> dict:
        return {"result": self.config.value * 2}


class SimpleActor(Actor[StatefulConfig, dict]):
    """Simple actor for testing."""
    config_type = StatefulConfig

    async def execute(self, ctx=None, **inputs) -> AsyncGenerator[None, dict]:
        """Execute actor with config-as-state pattern."""
        for i in range(5):
            self.config.items_processed += 1
            if i > self.config.threshold:
                self.config.count += 1
            yield

        if ctx:
            ctx.set_result(self.task_id, {
                'count': self.config.count,
                'items_processed': self.config.items_processed
            })
        return


# Tests for Task base class
class TestTask:
    """Tests for Task base class."""

    def test_task_instantiation_with_valid_config(self):
        """Test Task instantiation with valid config."""
        task = SimpleTask(value=42, name="test_task")

        assert task.config.value == 42
        assert task.config.name == "test_task"
        assert task.task_id.startswith("SimpleTask_")

    def test_task_instantiation_with_invalid_config(self):
        """Test Task instantiation fails with invalid config."""
        with pytest.raises(ValidationError):
            SimpleTask(value="not_an_int")

    def test_task_auto_generates_unique_id(self):
        """Test each task gets unique auto-generated ID."""
        task1 = SimpleTask()
        task2 = SimpleTask()

        assert task1.task_id != task2.task_id
        assert task1.task_id.startswith("SimpleTask_")
        assert task2.task_id.startswith("SimpleTask_")

    def test_task_execute_returns_dict(self):
        """Test Task execute() returns dict."""
        task = SimpleTask(value=10)
        result = task.execute()

        assert isinstance(result, dict)
        assert result["result"] == 20

    def test_task_callable_interface(self):
        """Test Task __call__ interface."""
        task = SimpleTask(value=15)
        result = task(input_value=42)

        assert isinstance(result, dict)
        assert result["result"] == 30

    def test_task_dump_serialization(self):
        """Test Task.dump() serializes config."""
        task = SimpleTask(value=99, name="serialize_test")
        dumped = task.dump()

        assert isinstance(dumped, dict)
        assert dumped["value"] == 99
        assert dumped["name"] == "serialize_test"

    def test_task_get_config_schema(self):
        """Test Task.get_config_schema() returns JSON schema."""
        schema = SimpleTask.get_config_schema()

        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "value" in schema["properties"]


# Tests for Actor base class
class TestActor:
    """Tests for Actor base class."""

    def test_actor_instantiation_with_config(self):
        """Test Actor instantiation with config."""
        actor = SimpleActor(threshold=0.3, count=0, items_processed=0)

        assert actor.config.threshold == 0.3
        assert actor.config.count == 0
        assert actor.config.items_processed == 0
        assert actor.task_id.startswith("SimpleActor_")

    def test_actor_is_task_subclass(self):
        """Test Actor is a Task subclass."""
        actor = SimpleActor()

        assert isinstance(actor, Task)
        assert isinstance(actor, Actor)

    @pytest.mark.asyncio
    async def test_actor_returns_async_generator(self):
        """Test Actor.execute() returns AsyncGenerator."""
        actor = SimpleActor(threshold=2.0)

        gen = actor.execute()

        assert isinstance(gen, AsyncGenerator)

    @pytest.mark.asyncio
    async def test_actor_async_generator_protocol(self):
        """Test Actor AsyncGenerator can be iterated."""
        actor = SimpleActor(threshold=2.0)

        gen = actor.execute()
        iterations = 0

        async for _ in gen:
            iterations += 1

        assert iterations == 5

    @pytest.mark.asyncio
    async def test_actor_config_as_state_pattern(self):
        """Test Actor config-as-state pattern."""
        actor = SimpleActor(threshold=2.0, count=0, items_processed=0)

        gen = actor.execute()
        async for _ in gen:
            pass

        # State stored in config
        assert actor.config.items_processed == 5
        assert actor.config.count == 2  # items 3, 4 are > 2.0 (0-indexed: 3.0, 4.0)

    @pytest.mark.asyncio
    async def test_actor_config_changes_during_execution(self):
        """Test Actor reflects config changes during execution."""
        actor = SimpleActor(threshold=5.0)  # High threshold, no items pass

        gen = actor.execute()

        # Process 2 iterations
        await gen.__anext__()
        await gen.__anext__()

        # Update config (simulating hot reconfiguration)
        actor.config.threshold = 0.0  # Now all items should pass

        # Continue execution
        async for _ in gen:
            pass

        # Count should reflect new threshold
        assert actor.config.count > 0

    @pytest.mark.asyncio
    async def test_actor_set_result_via_context(self):
        """Test Actor uses ctx.set_result() for final results."""
        class MockContext:
            def __init__(self):
                self.results = {}

            def set_result(self, task_id, result):
                self.results[task_id] = result

        actor = SimpleActor(threshold=2.0)
        ctx = MockContext()

        gen = actor.execute(ctx=ctx)
        async for _ in gen:
            pass

        # Result should be set via context
        assert actor.task_id in ctx.results
        assert ctx.results[actor.task_id]['items_processed'] == 5
        assert ctx.results[actor.task_id]['count'] == 2


# Tests for Task vs Actor differentiation
class TestTaskActorDifferentiation:
    """Tests for distinguishing Task from Actor."""

    def test_task_and_actor_have_different_execute_signatures(self):
        """Test Task and Actor have different execute() signatures."""
        import inspect

        task = SimpleTask()
        actor = SimpleActor()

        task_sig = inspect.signature(task.execute)
        actor_sig = inspect.signature(actor.execute)

        # Both should have signatures, but return annotations differ
        assert task_sig.return_annotation == dict
        # Actor return annotation should be AsyncGenerator
        from typing import get_origin
        assert get_origin(actor_sig.return_annotation) is AsyncGenerator

    def test_actor_instance_check(self):
        """Test isinstance() distinguishes Actor from Task."""
        task = SimpleTask()
        actor = SimpleActor()

        assert isinstance(task, Task)
        assert not isinstance(task, Actor)

        assert isinstance(actor, Task)
        assert isinstance(actor, Actor)


# Removed TestActorDeprecatedMethods - methods removed in refactoring


# Tests for TaskConfig serialization model
class TestTaskConfig:
    """Tests for TaskConfig serialization model."""

    def test_task_config_instantiation(self):
        """Test TaskConfig can be instantiated."""
        config = TaskConfig(
            task_id="task_123",
            task_type="SimpleTask",
            config={"value": 42, "name": "test"},
            status="pending"
        )

        assert config.task_id == "task_123"
        assert config.task_type == "SimpleTask"
        assert config.config["value"] == 42
        assert config.status == "pending"

    def test_task_config_default_values(self):
        """Test TaskConfig default values."""
        config = TaskConfig(
            task_id="task_123",
            task_type="SimpleTask",
            config={}
        )

        assert config.status == "pending"
        assert config.phase is None
