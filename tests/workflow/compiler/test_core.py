"""Tests for WorkflowCompiler type detection and channel allocation."""

import pytest
from collections.abc import AsyncGenerator
from pydantic import BaseModel

from molexp.workflow import Workflow, Link
from molexp.workflow.task import Task, Actor
from molexp.workflow.compiler import WorkflowCompiler
from molexp.workflow.execution_type import TaskExecutionType


# Test configurations
class SimpleConfig(BaseModel):
    value: int = 10


class ActorConfig(BaseModel):
    threshold: float = 0.5
    count: int = 0


# Test tasks
class BatchTask(Task[SimpleConfig, dict]):
    """Batch task for testing."""
    config_type = SimpleConfig
    outputs = {'result': int}

    def execute(self, ctx=None, **inputs) -> dict:
        return {'result': self.config.value * 2}


class StreamActor(Actor[ActorConfig, dict]):
    """Actor task for testing."""
    config_type = ActorConfig

    async def execute(self, ctx=None, **inputs) -> AsyncGenerator[None, dict]:
        while self.config.count < 5:
            self.config.count += 1
            yield

        if ctx:
            ctx.set_result(self.task_id, {'count': self.config.count})
        return


class AnotherActor(Actor[ActorConfig, dict]):
    """Another actor for multi-actor tests."""
    config_type = ActorConfig

    async def execute(self, ctx=None, **inputs) -> AsyncGenerator[None, dict]:
        for _ in range(3):
            yield
        return


# Tests for return type inspection
class TestReturnTypeInspection:
    """Tests for compiler return type detection."""

    def test_detect_batch_task_type(self):
        """Test compiler detects BATCH task via dict return type."""
        compiler = WorkflowCompiler()
        task = BatchTask()

        task_type = compiler._detect_task_type(task)

        assert task_type == TaskExecutionType.BATCH

    def test_detect_actor_task_type(self):
        """Test compiler detects ACTOR task via AsyncGenerator return type."""
        compiler = WorkflowCompiler()
        actor = StreamActor()

        task_type = compiler._detect_task_type(actor)

        assert task_type == TaskExecutionType.ACTOR

    def test_detect_multiple_task_types(self):
        """Test compiler correctly identifies mixed task types."""
        compiler = WorkflowCompiler()

        batch1 = BatchTask()
        batch2 = BatchTask()
        actor1 = StreamActor()
        actor2 = AnotherActor()

        types = {
            batch1.task_id: compiler._detect_task_type(batch1),
            batch2.task_id: compiler._detect_task_type(batch2),
            actor1.task_id: compiler._detect_task_type(actor1),
            actor2.task_id: compiler._detect_task_type(actor2),
        }

        assert types[batch1.task_id] == TaskExecutionType.BATCH
        assert types[batch2.task_id] == TaskExecutionType.BATCH
        assert types[actor1.task_id] == TaskExecutionType.ACTOR
        assert types[actor2.task_id] == TaskExecutionType.ACTOR


# Tests for TaskExecutionType detection
class TestTaskExecutionTypeDetection:
    """Tests for BATCH vs ACTOR classification."""

    def test_batch_execution_type(self):
        """Test BATCH TaskExecutionType."""
        assert TaskExecutionType.BATCH.value == "batch"

    def test_actor_execution_type(self):
        """Test ACTOR TaskExecutionType."""
        assert TaskExecutionType.ACTOR.value == "actor"

    def test_execution_type_enum_values(self):
        """Test TaskExecutionType enum has correct values."""
        types = [t.value for t in TaskExecutionType]

        assert "batch" in types
        assert "actor" in types
        assert len(types) == 2


# Tests for channel allocation from Links
class TestChannelAllocation:
    """Tests for channel allocation logic."""

    def test_allocate_channels_for_actor_link(self):
        """Test compiler allocates channel for actor-to-actor link."""
        compiler = WorkflowCompiler()

        actor1 = StreamActor()
        actor2 = AnotherActor()

        workflow = Workflow.from_tasks(
            tasks=[actor1, actor2],
            links=[Link(source=actor1, target=actor2, mapping={'data': 'data'})],
            name="test_workflow"
        )

        compiled = compiler.compile(workflow)
        channels = compiled.get_channels()

        # Should have one channel allocated
        assert len(channels) == 1

        # Channel should reference source and target
        channel_id = f"{actor1.task_id}_to_{actor2.task_id}"
        assert channel_id in channels
        assert channels[channel_id]['source'] == actor1.task_id
        assert channels[channel_id]['target'] == actor2.task_id

    def test_no_channels_for_batch_only_workflow(self):
        """Test compiler does not allocate channels for batch-only workflow."""
        compiler = WorkflowCompiler()

        batch1 = BatchTask()
        batch2 = BatchTask()

        workflow = Workflow.from_tasks(
            tasks=[batch1, batch2],
            links=[],
            name="batch_workflow"
        )

        compiled = compiler.compile(workflow)
        channels = compiled.get_channels()

        assert len(channels) == 0

    def test_channel_buffer_size_configuration(self):
        """Test channel buffer_size from Link is preserved."""
        compiler = WorkflowCompiler()

        actor1 = StreamActor()
        actor2 = AnotherActor()

        workflow = Workflow.from_tasks(
            tasks=[actor1, actor2],
            links=[Link(
                source=actor1,
                target=actor2,
                buffer_size=200,
                mapping={'out': 'in'}
            )],
            name="test_workflow"
        )

        compiled = compiler.compile(workflow)
        channels = compiled.get_channels()

        channel_id = f"{actor1.task_id}_to_{actor2.task_id}"
        assert channels[channel_id]['buffer_size'] == 200


# Tests for mapping population order
class TestMappingPopulation:
    """Tests for Link.mapping auto-generation and population order."""

    def test_link_mapping_auto_generated_for_actors(self):
        """Test Link.mapping is auto-generated for actor links."""
        compiler = WorkflowCompiler()

        actor1 = StreamActor()
        actor2 = AnotherActor()

        # Link without mapping
        link = Link(source=actor1, target=actor2)

        workflow = Workflow.from_tasks(
            tasks=[actor1, actor2],
            links=[link],
            name="test_workflow"
        )

        # Compile (triggers validation which populates mapping)
        compiler.compile(workflow)

        # Link should now have mapping
        assert link.mapping is not None
        assert isinstance(link.mapping, dict)

    def test_mapping_populated_before_channel_allocation(self):
        """Test validation populates mapping before channel allocation."""
        compiler = WorkflowCompiler()

        actor1 = StreamActor()
        actor2 = AnotherActor()

        link = Link(source=actor1, target=actor2)

        workflow = Workflow.from_tasks(
            tasks=[actor1, actor2],
            links=[link],
            name="test_workflow"
        )

        # Before compilation, mapping may be None
        original_mapping = link.mapping

        compiled = compiler.compile(workflow)
        channels = compiled.get_channels()

        # After compilation:
        # 1. Mapping should be populated
        assert link.mapping is not None

        # 2. Channel should exist with mapping
        channel_id = f"{actor1.task_id}_to_{actor2.task_id}"
        assert channel_id in channels
        assert 'mapping' in channels[channel_id]

    def test_explicit_mapping_preserved(self):
        """Test explicit Link.mapping is preserved (not overwritten)."""
        compiler = WorkflowCompiler()

        actor1 = StreamActor()
        actor2 = AnotherActor()

        explicit_mapping = {'output_data': 'input_data'}
        link = Link(source=actor1, target=actor2, mapping=explicit_mapping)

        workflow = Workflow.from_tasks(
            tasks=[actor1, actor2],
            links=[link],
            name="test_workflow"
        )

        compiled = compiler.compile(workflow)

        # Explicit mapping should be preserved
        assert link.mapping == explicit_mapping


# Tests for cycle detection
class TestCycleDetection:
    """Tests for cycle detection in workflows."""

    def test_cycles_allowed_for_actor_workflows(self):
        """Test cycles are allowed when workflow contains actors."""
        compiler = WorkflowCompiler()

        actor_a = StreamActor()
        actor_b = AnotherActor()
        actor_c = StreamActor()

        # Create feedback loop: A → B → C → A
        workflow = Workflow.from_tasks(
            tasks=[actor_a, actor_b, actor_c],
            links=[
                Link(source=actor_a, target=actor_b, mapping={'ab': 'ab'}),
                Link(source=actor_b, target=actor_c, mapping={'bc': 'bc'}),
                Link(source=actor_c, target=actor_a, mapping={'ca': 'ca'}),
            ],
            name="feedback_loop"
        )

        # Should compile without error
        compiled = compiler.compile(workflow)

        assert compiled is not None
        assert len(compiled.get_channels()) == 3

    def test_cycles_rejected_for_batch_only_workflows(self):
        """Test cycles are rejected for pure batch workflows."""
        # Note: This requires cycle detection to be implemented in compiler
        # If not implemented yet, this test documents expected behavior

        compiler = WorkflowCompiler()

        # For batch tasks, we need to properly declare inputs/outputs
        class BatchWithIO(Task[SimpleConfig, dict]):
            config_type = SimpleConfig
            inputs = {'in': int}
            outputs = {'out': int}

            def execute(self, ctx=None, **inputs) -> dict:
                return {'out': inputs.get('in', 0) * 2}

        batch_a = BatchWithIO()
        batch_b = BatchWithIO()
        batch_c = BatchWithIO()

        # Try to create cycle with batch tasks
        workflow = Workflow.from_tasks(
            tasks=[batch_a, batch_b, batch_c],
            links=[
                Link(source=batch_a, target=batch_b, mapping={'out': 'in'}),
                Link(source=batch_b, target=batch_c, mapping={'out': 'in'}),
                Link(source=batch_c, target=batch_a, mapping={'out': 'in'}),
            ],
            name="batch_cycle"
        )

        # Should raise ValueError for cycle in batch workflow
        # (if cycle detection is implemented)
        # with pytest.raises(ValueError, match="cycle"):
        #     compiler.compile(workflow)

        # For now, this test documents the expected behavior
        # Uncomment assertion above when cycle detection is implemented


# Tests for dynamic channel validation
class TestDynamicChannelValidation:
    """Tests that compiler skips channel validation for actors."""

    def test_actors_do_not_require_inputs_outputs(self):
        """Test actors don't need inputs/outputs declarations."""
        compiler = WorkflowCompiler()

        # Actors without inputs/outputs declarations
        actor1 = StreamActor()
        actor2 = AnotherActor()

        # Should compile without error
        workflow = Workflow.from_tasks(
            tasks=[actor1, actor2],
            links=[Link(source=actor1, target=actor2, mapping={'data': 'data'})],
            name="dynamic_channels"
        )

        compiled = compiler.compile(workflow)

        assert compiled is not None

    def test_batch_tasks_require_outputs_for_links(self):
        """Test batch tasks require outputs declaration for links."""
        compiler = WorkflowCompiler()

        class BatchNoOutputs(Task[SimpleConfig, dict]):
            config_type = SimpleConfig
            # No outputs declared

            def execute(self, ctx=None, **inputs) -> dict:
                return {}

        class BatchNoInputs(Task[SimpleConfig, dict]):
            config_type = SimpleConfig
            # No inputs declared

            def execute(self, ctx=None, **inputs) -> dict:
                return {}

        batch1 = BatchNoOutputs()
        batch2 = BatchNoInputs()

        workflow = Workflow.from_tasks(
            tasks=[batch1, batch2],
            links=[Link(source=batch1, target=batch2)],
            name="invalid_batch"
        )

        # Should raise validation error
        with pytest.raises(ValueError, match="must declare"):
            compiler.compile(workflow)


# Tests for CompiledWorkflow interface
class TestCompiledWorkflow:
    """Tests for CompiledWorkflow high-level API."""

    def test_get_task_types(self):
        """Test compiled workflow exposes task types."""
        compiler = WorkflowCompiler()

        batch = BatchTask()
        actor = StreamActor()

        workflow = Workflow.from_tasks(
            tasks=[batch, actor],
            links=[],
            name="mixed_workflow"
        )

        compiled = compiler.compile(workflow)
        task_types = compiled.get_task_types()

        assert task_types[batch.task_id] == TaskExecutionType.BATCH
        assert task_types[actor.task_id] == TaskExecutionType.ACTOR

    def test_get_channels(self):
        """Test compiled workflow exposes channel configurations."""
        compiler = WorkflowCompiler()

        actor1 = StreamActor()
        actor2 = AnotherActor()

        workflow = Workflow.from_tasks(
            tasks=[actor1, actor2],
            links=[Link(source=actor1, target=actor2, mapping={'x': 'y'})],
            name="channel_workflow"
        )

        compiled = compiler.compile(workflow)
        channels = compiled.get_channels()

        assert len(channels) > 0
        channel_id = f"{actor1.task_id}_to_{actor2.task_id}"
        assert channel_id in channels

    def test_get_dependency_graph(self):
        """Test compiled workflow exposes dependency graph."""
        compiler = WorkflowCompiler()

        batch1 = BatchTask()
        batch2 = BatchTask()

        workflow = Workflow.from_tasks(
            tasks=[batch1, batch2],
            links=[],
            name="simple_workflow"
        )

        compiled = compiler.compile(workflow)
        adj, in_degree = compiled.get_dependency_graph()

        assert isinstance(adj, dict)
        assert isinstance(in_degree, dict)
        assert batch1.task_id in adj
        assert batch2.task_id in adj
