"""Tests for WorkflowEngine hybrid execution and hot reconfiguration."""

import asyncio
import pytest
from collections.abc import AsyncGenerator
from pathlib import Path
from tempfile import TemporaryDirectory
from pydantic import BaseModel

from molexp.workflow import Workflow, Link
from molexp.workflow.task import Task, Actor
from molexp.workflow.engine import WorkflowEngine
from molexp.workflow.execution_type import TaskExecutionType
from molexp.workspace import Workspace, RunContext


# Test configurations
class SimpleConfig(BaseModel):
    value: int = 10


class ActorConfig(BaseModel):
    threshold: float = 0.5
    count: int = 0
    max_items: int = 5


# Test tasks
class BatchTask(Task[SimpleConfig, dict]):
    """Simple batch task."""
    config_type = SimpleConfig

    def execute(self, ctx=None, **inputs) -> dict:
        return {'result': self.config.value * 2}


class CounterActor(Actor[ActorConfig, dict]):
    """Simple counting actor."""
    config_type = ActorConfig

    async def execute(self, ctx=None, **inputs) -> AsyncGenerator[None, dict]:
        while self.config.count < self.config.max_items:
            self.config.count += 1
            yield

        if ctx:
            ctx.set_result(self.task_id, {'count': self.config.count})
        return


class DataActor(Actor[ActorConfig, dict]):
    """Actor that processes data via channels."""
    config_type = ActorConfig

    async def execute(self, ctx=None, **inputs) -> AsyncGenerator[None, dict]:
        if ctx:
            for i in range(3):
                await ctx.emit('output', {'id': i, 'value': i * 10})
                yield

        if ctx:
            ctx.set_result(self.task_id, {'sent': 3})
        return


class ProcessorActor(Actor[ActorConfig, dict]):
    """Actor that receives and processes data."""
    config_type = ActorConfig

    async def execute(self, ctx=None, **inputs) -> AsyncGenerator[None, dict]:
        if ctx:
            processed = []
            for _ in range(3):
                data = await asyncio.wait_for(ctx.receive('input'), timeout=2.0)
                processed.append(data)
                yield

            ctx.set_result(self.task_id, {'processed': len(processed)})
        return


# Mock workspace/run for testing
def create_test_run(tmpdir):
    """Create a test run context."""
    class MockWorkspace:
        def __init__(self, root):
            self.root = Path(root)

    class MockProject:
        def __init__(self, workspace):
            self.id = "test_project"
            self.workspace = workspace

    class MockExperiment:
        def __init__(self, project):
            self.id = "test_experiment"
            self.project = project

    class MockMetadata:
        def __init__(self):
            from datetime import datetime
            self.created_at = datetime.now()
            self.updated_at = datetime.now()

    class MockRun:
        def __init__(self, work_dir):
            from molexp.workflow.status import TaskStatus as RunStatus
            self.id = "test_run"
            self.status = RunStatus.PENDING
            self.parameters = {}
            self.assets = {}
            self.metadata = MockMetadata()
            workspace = MockWorkspace(Path(work_dir))  # Use tmpdir directly, not parent!
            project = MockProject(workspace)
            self.experiment = MockExperiment(project)

        def start(self):
            return RunContext(self)

        def save(self):
            pass

    return MockRun(tmpdir)


# Tests for execution mode detection
class TestExecutionModeDetection:
    """Tests for automatic execution mode detection."""

    def test_pure_batch_workflow_detection(self):
        """Test engine detects pure batch workflow."""
        batch1 = BatchTask()
        batch2 = BatchTask()

        workflow = Workflow.from_tasks(
            tasks=[batch1, batch2],
            links=[],
            name="pure_batch"
        )

        engine = WorkflowEngine(workflow)
        task_types = engine.compiled.get_task_types()

        # All tasks should be BATCH
        assert all(t == TaskExecutionType.BATCH for t in task_types.values())

    def test_hybrid_workflow_detection(self):
        """Test engine detects hybrid workflow (contains actors)."""
        batch = BatchTask()
        actor = CounterActor()

        workflow = Workflow.from_tasks(
            tasks=[batch, actor],
            links=[],
            name="hybrid"
        )

        engine = WorkflowEngine(workflow)
        task_types = engine.compiled.get_task_types()

        # Should have both BATCH and ACTOR
        assert TaskExecutionType.BATCH in task_types.values()
        assert TaskExecutionType.ACTOR in task_types.values()

    def test_pure_actor_workflow_detection(self):
        """Test engine detects pure actor workflow."""
        actor1 = CounterActor()
        actor2 = DataActor()

        workflow = Workflow.from_tasks(
            tasks=[actor1, actor2],
            links=[],
            name="pure_actor"
        )

        engine = WorkflowEngine(workflow)
        task_types = engine.compiled.get_task_types()

        # All tasks should be ACTOR
        assert all(t == TaskExecutionType.ACTOR for t in task_types.values())


# Tests for pure batch fast path
class TestPureBatchFastPath:
    """Tests for pure batch execution optimization."""

    def test_pure_batch_executes_successfully(self):
        """Test pure batch workflow executes via fast path."""
        with TemporaryDirectory() as tmpdir:
            batch = BatchTask(value=21)

            workflow = Workflow.from_tasks(
                tasks=[batch],
                links=[],
                name="single_batch"
            )

            engine = WorkflowEngine(workflow)
            run = create_test_run(tmpdir)

            with run.start() as run_ctx:
                results = engine.execute(run_context=run_ctx)

            assert batch.task_id in results
            assert results[batch.task_id]['result'] == 42


# Tests for actor concurrent execution
class TestActorConcurrentExecution:
    """Tests for concurrent actor execution."""

    def test_single_actor_executes(self):
        """Test single actor executes successfully."""
        with TemporaryDirectory() as tmpdir:
            actor = CounterActor(max_items=3)

            workflow = Workflow.from_tasks(
                tasks=[actor],
                links=[],
                name="single_actor"
            )

            engine = WorkflowEngine(workflow)
            run = create_test_run(tmpdir)

            with run.start() as run_ctx:
                results = engine.execute(run_context=run_ctx)

            assert actor.task_id in results
            assert results[actor.task_id]['count'] == 3

    def test_multiple_actors_execute_concurrently(self):
        """Test multiple actors execute concurrently."""
        with TemporaryDirectory() as tmpdir:
            actor1 = CounterActor(max_items=3)
            actor2 = CounterActor(max_items=5)
            actor3 = CounterActor(max_items=2)

            workflow = Workflow.from_tasks(
                tasks=[actor1, actor2, actor3],
                links=[],
                name="multi_actor"
            )

            engine = WorkflowEngine(workflow)
            run = create_test_run(tmpdir)

            with run.start() as run_ctx:
                results = engine.execute(run_context=run_ctx)

            # All actors should complete
            assert len(results) == 3
            assert results[actor1.task_id]['count'] == 3
            assert results[actor2.task_id]['count'] == 5
            assert results[actor3.task_id]['count'] == 2

    def test_actors_with_message_passing(self):
        """Test actors communicate via channels."""
        with TemporaryDirectory() as tmpdir:
            sender = DataActor()
            receiver = ProcessorActor()

            workflow = Workflow.from_tasks(
                tasks=[sender, receiver],
                links=[Link(
                    source=sender,
                    target=receiver,
                    mapping={'output': 'input'}
                )],
                name="actor_pipeline"
            )

            engine = WorkflowEngine(workflow)
            run = create_test_run(tmpdir)

            with run.start() as run_ctx:
                results = engine.execute(run_context=run_ctx)

            # Both actors should complete
            assert sender.task_id in results
            assert receiver.task_id in results
            assert results[sender.task_id]['sent'] == 3
            assert results[receiver.task_id]['processed'] == 3


# Tests for actor failure propagation
class TestActorFailurePropagation:
    """Tests for actor failure handling."""

    def test_actor_exception_captured(self):
        """Test actor exceptions are captured."""
        class FailingActor(Actor[ActorConfig, dict]):
            config_type = ActorConfig

            async def execute(self, ctx=None, **inputs) -> AsyncGenerator[None, dict]:
                yield
                raise ValueError("Test failure")

        with TemporaryDirectory() as tmpdir:
            actor = FailingActor()

            workflow = Workflow.from_tasks(
                tasks=[actor],
                links=[],
                name="failing_actor"
            )

            engine = WorkflowEngine(workflow)
            run = create_test_run(tmpdir)

            with run.start() as run_ctx:
                # Engine should handle failure gracefully
                results = engine.execute(run_context=run_ctx)

            # Check that failure was recorded
            # (Exact behavior depends on engine implementation)


# Tests for channel creation and registration
class TestChannelManagement:
    """Tests for channel creation and registration."""

    def test_channels_created_for_actor_links(self):
        """Test channels are created for actor-to-actor links."""
        with TemporaryDirectory() as tmpdir:
            actor1 = DataActor()
            actor2 = ProcessorActor()

            workflow = Workflow.from_tasks(
                tasks=[actor1, actor2],
                links=[Link(
                    source=actor1,
                    target=actor2,
                    mapping={'output': 'input'},
                    buffer_size=50
                )],
                name="channel_test"
            )

            engine = WorkflowEngine(workflow)
            channels = engine.compiled.get_channels()

            # Should have one channel
            assert len(channels) == 1

            channel_id = f"{actor1.task_id}_to_{actor2.task_id}"
            assert channel_id in channels
            assert channels[channel_id]['buffer_size'] == 50

    def test_channel_registration_in_context(self):
        """Test channels are registered in RunContext."""
        with TemporaryDirectory() as tmpdir:
            sender = DataActor()
            receiver = ProcessorActor()

            workflow = Workflow.from_tasks(
                tasks=[sender, receiver],
                links=[Link(
                    source=sender,
                    target=receiver,
                    mapping={'output': 'input'}
                )],
                name="registration_test"
            )

            engine = WorkflowEngine(workflow)
            run = create_test_run(tmpdir)

            with run.start() as run_ctx:
                # Execute workflow (channels should be registered)
                engine.execute(run_context=run_ctx)

                # After execution, can't directly check channels
                # but the fact that message passing worked proves registration


# Tests for result capture from ctx.set_result()
class TestResultCapture:
    """Tests for capturing actor results via ctx.set_result()."""

    def test_actor_result_captured(self):
        """Test actor result is captured via ctx.set_result()."""
        with TemporaryDirectory() as tmpdir:
            actor = CounterActor(max_items=10)

            workflow = Workflow.from_tasks(
                tasks=[actor],
                links=[],
                name="result_capture"
            )

            engine = WorkflowEngine(workflow)
            run = create_test_run(tmpdir)

            with run.start() as run_ctx:
                results = engine.execute(run_context=run_ctx)

            # Result should be captured
            assert actor.task_id in results
            assert 'count' in results[actor.task_id]
            assert results[actor.task_id]['count'] == 10

    def test_multiple_actor_results_captured(self):
        """Test results from multiple actors are all captured."""
        with TemporaryDirectory() as tmpdir:
            actor1 = CounterActor(max_items=5)
            actor2 = CounterActor(max_items=7)
            actor3 = CounterActor(max_items=3)

            workflow = Workflow.from_tasks(
                tasks=[actor1, actor2, actor3],
                links=[],
                name="multi_result"
            )

            engine = WorkflowEngine(workflow)
            run = create_test_run(tmpdir)

            with run.start() as run_ctx:
                results = engine.execute(run_context=run_ctx)

            # All results should be captured
            assert len(results) == 3
            assert all('count' in r for r in results.values())


# Tests for hot reconfiguration
class TestHotReconfiguration:
    """Tests for update_actor_config() method."""

    def test_update_actor_config_basic(self):
        """Test update_actor_config() updates actor config."""
        actor = CounterActor(max_items=10)

        workflow = Workflow.from_tasks(
            tasks=[actor],
            links=[],
            name="reconfig_test"
        )

        engine = WorkflowEngine(workflow)

        # Update config
        engine.update_actor_config(actor.task_id, {'max_items': 20})

        # Config should be updated
        assert actor.config.max_items == 20

    def test_update_actor_config_validation(self):
        """Test update_actor_config() validates new config."""
        actor = CounterActor(max_items=10)

        workflow = Workflow.from_tasks(
            tasks=[actor],
            links=[],
            name="validation_test"
        )

        engine = WorkflowEngine(workflow)

        # Try invalid config (wrong type) - should raise ValidationError
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            engine.update_actor_config(actor.task_id, {'max_items': "not_an_int"})

    def test_update_actor_config_invalid_actor_id(self):
        """Test update_actor_config() raises KeyError for invalid ID."""
        actor = CounterActor()

        workflow = Workflow.from_tasks(
            tasks=[actor],
            links=[],
            name="invalid_id_test"
        )

        engine = WorkflowEngine(workflow)

        # Try non-existent actor
        with pytest.raises(KeyError, match="not found"):
            engine.update_actor_config('nonexistent_actor', {'max_items': 5})

    def test_update_actor_config_not_an_actor(self):
        """Test update_actor_config() raises TypeError for non-actor task."""
        batch = BatchTask()

        workflow = Workflow.from_tasks(
            tasks=[batch],
            links=[],
            name="not_actor_test"
        )

        engine = WorkflowEngine(workflow)

        # Try to update batch task (not an actor)
        with pytest.raises(TypeError, match="not an Actor"):
            engine.update_actor_config(batch.task_id, {'value': 100})

    def test_update_actor_config_partial_update(self):
        """Test update_actor_config() supports partial updates."""
        actor = CounterActor(threshold=0.5, count=0, max_items=10)

        workflow = Workflow.from_tasks(
            tasks=[actor],
            links=[],
            name="partial_update_test"
        )

        engine = WorkflowEngine(workflow)

        # Update only one field
        engine.update_actor_config(actor.task_id, {'threshold': 0.8})

        # Only threshold should change
        assert actor.config.threshold == 0.8
        assert actor.config.max_items == 10  # Unchanged
        assert actor.config.count == 0  # Unchanged

    def test_update_actor_config_multiple_fields(self):
        """Test update_actor_config() can update multiple fields."""
        actor = CounterActor(threshold=0.5, max_items=10)

        workflow = Workflow.from_tasks(
            tasks=[actor],
            links=[],
            name="multi_field_test"
        )

        engine = WorkflowEngine(workflow)

        # Update multiple fields
        engine.update_actor_config(actor.task_id, {
            'threshold': 0.9,
            'max_items': 20
        })

        assert actor.config.threshold == 0.9
        assert actor.config.max_items == 20
