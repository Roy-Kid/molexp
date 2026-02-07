"""Tests for RunContext message passing in hybrid mode."""

import asyncio
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from molexp.workspace import Workspace, RunContext, Context


# Mock classes for testing
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
    """Mock Run for testing."""
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

    def save(self):
        pass


# Tests for emit() and receive()
class TestMessagePassing:
    """Tests for ctx.emit() and ctx.receive()."""

    @pytest.mark.asyncio
    async def test_emit_and_receive_basic(self):
        """Test basic emit and receive functionality."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            # Create a channel
            queue = asyncio.Queue(maxsize=10)
            ctx._register_channel('test_channel', queue)

            # Emit message
            await ctx.emit('test_channel', {'data': 42})

            # Receive message
            received = await ctx.receive('test_channel')

            assert received == {'data': 42}

    @pytest.mark.asyncio
    async def test_emit_multiple_messages(self):
        """Test emitting multiple messages."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            queue = asyncio.Queue(maxsize=10)
            ctx._register_channel('data', queue)

            # Emit multiple messages
            messages = [{'id': i, 'value': i * 10} for i in range(5)]
            for msg in messages:
                await ctx.emit('data', msg)

            # Receive all messages
            received = []
            for _ in range(5):
                msg = await ctx.receive('data')
                received.append(msg)

            assert received == messages

    @pytest.mark.asyncio
    async def test_receive_blocks_when_empty(self):
        """Test receive() blocks when channel is empty."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            queue = asyncio.Queue(maxsize=10)
            ctx._register_channel('empty_channel', queue)

            # Try to receive with timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    ctx.receive('empty_channel'),
                    timeout=0.1
                )

    @pytest.mark.asyncio
    async def test_emit_blocks_when_full(self):
        """Test emit() blocks when channel is full (backpressure)."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            # Create channel with small buffer
            queue = asyncio.Queue(maxsize=2)
            ctx._register_channel('small_channel', queue)

            # Fill the channel
            await ctx.emit('small_channel', 'msg1')
            await ctx.emit('small_channel', 'msg2')

            # Next emit should block
            emit_task = asyncio.create_task(
                ctx.emit('small_channel', 'msg3')
            )

            # Give it time to try emitting
            await asyncio.sleep(0.1)

            # Task should not be done (blocked)
            assert not emit_task.done()

            # Receive one message to make space
            await ctx.receive('small_channel')

            # Now emit should complete
            await asyncio.wait_for(emit_task, timeout=0.5)
            assert emit_task.done()


# Tests for channel not found errors
class TestChannelErrors:
    """Tests for channel error handling."""

    @pytest.mark.asyncio
    async def test_emit_channel_not_found(self):
        """Test emit() silently drops messages to non-existent channels."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            # Emit to non-existent channel should not raise (message is dropped)
            await ctx.emit('nonexistent', {'data': 42})
            # No exception raised - message was silently dropped

    @pytest.mark.asyncio
    async def test_receive_channel_not_found(self):
        """Test receive() raises KeyError for non-existent channel."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            with pytest.raises(KeyError, match="Channel 'nonexistent' not found"):
                await ctx.receive('nonexistent')

    @pytest.mark.asyncio
    async def test_error_message_shows_available_channels(self):
        """Test emit to non-existent channel works when others exist."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            # Register some channels
            ctx._register_channel('channel1', asyncio.Queue())
            ctx._register_channel('channel2', asyncio.Queue())

            # Emit to non-existent channel should not raise (message dropped)
            await ctx.emit('nonexistent', {'data': 42})
            # No exception - emit to unconnected channels is allowed


# Tests for get_channel_depths()
class TestChannelMonitoring:
    """Tests for ctx.get_channel_depths()."""

    @pytest.mark.asyncio
    async def test_get_channel_depths_empty(self):
        """Test get_channel_depths() with empty channels."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            queue1 = asyncio.Queue(maxsize=10)
            queue2 = asyncio.Queue(maxsize=20)
            ctx._register_channel('ch1', queue1)
            ctx._register_channel('ch2', queue2)

            depths = ctx.get_channel_depths()

            assert depths == {'ch1': 0, 'ch2': 0}

    @pytest.mark.asyncio
    async def test_get_channel_depths_with_messages(self):
        """Test get_channel_depths() reports correct queue sizes."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            queue1 = asyncio.Queue(maxsize=10)
            queue2 = asyncio.Queue(maxsize=20)
            ctx._register_channel('data', queue1)
            ctx._register_channel('results', queue2)

            # Add messages to channels
            await ctx.emit('data', 'msg1')
            await ctx.emit('data', 'msg2')
            await ctx.emit('data', 'msg3')
            await ctx.emit('results', 'res1')

            depths = ctx.get_channel_depths()

            assert depths['data'] == 3
            assert depths['results'] == 1

    @pytest.mark.asyncio
    async def test_get_channel_depths_updates_dynamically(self):
        """Test get_channel_depths() reflects dynamic changes."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            queue = asyncio.Queue(maxsize=10)
            ctx._register_channel('dynamic', queue)

            # Initially empty
            assert ctx.get_channel_depths()['dynamic'] == 0

            # Add messages
            await ctx.emit('dynamic', 'msg1')
            await ctx.emit('dynamic', 'msg2')
            assert ctx.get_channel_depths()['dynamic'] == 2

            # Consume message
            await ctx.receive('dynamic')
            assert ctx.get_channel_depths()['dynamic'] == 1


# Tests for set_result()
class TestActorResults:
    """Tests for ctx.set_result() for actor results."""

    def test_set_result_basic(self):
        """Test set_result() stores result in context."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            ctx.set_result('actor_123', {'count': 42, 'status': 'complete'})

            result = ctx.get_result('actor_123')
            assert result == {'count': 42, 'status': 'complete'}

    def test_set_result_overwrites_existing(self):
        """Test set_result() overwrites existing result."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            ctx.set_result('task_1', {'value': 10})
            ctx.set_result('task_1', {'value': 20})

            result = ctx.get_result('task_1')
            assert result == {'value': 20}

    def test_get_result_returns_none_for_nonexistent(self):
        """Test get_result() returns None for non-existent key."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            result = ctx.get_result('nonexistent')
            assert result is None

    @pytest.mark.asyncio
    async def test_set_result_used_by_actor(self):
        """Test set_result() works in actor execution context."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            # Simulate actor calling set_result
            actor_id = 'actor_456'
            final_result = {
                'items_processed': 100,
                'errors': 0,
                'success_rate': 1.0
            }

            ctx.set_result(actor_id, final_result)

            # Verify result is stored
            retrieved = ctx.get_result(actor_id)
            assert retrieved == final_result


# Tests for backward compatibility
class TestBackwardCompatibility:
    """Tests for backward compatibility with batch-only mode."""

    def test_context_works_without_channels(self):
        """Test RunContext works without any channels (batch mode)."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            # Should be able to use set_result and get_result
            ctx.set_result('batch_task', {'output': 42})
            result = ctx.get_result('batch_task')

            assert result == {'output': 42}

    def test_get_channel_depths_with_no_channels(self):
        """Test get_channel_depths() returns empty dict when no channels."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            depths = ctx.get_channel_depths()

            assert depths == {}

    def test_batch_task_does_not_need_message_passing(self):
        """Test batch tasks don't need emit/receive methods."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            # Batch task execution (no message passing)
            # Just uses set_result
            ctx.set_result('batch_1', {'data': [1, 2, 3]})
            ctx.set_result('batch_2', {'processed': True})

            # Verify both results stored
            assert ctx.get_result('batch_1') == {'data': [1, 2, 3]}
            assert ctx.get_result('batch_2') == {'processed': True}


# Tests for channel registration
class TestChannelRegistration:
    """Tests for _register_channel() internal method."""

    def test_register_channel_creates_entry(self):
        """Test _register_channel() adds channel to internal map."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            queue = asyncio.Queue(maxsize=10)
            ctx._register_channel('my_channel', queue)

            # Channel should be accessible
            assert 'my_channel' in ctx._channels
            assert ctx._channels['my_channel'] is queue

    def test_register_multiple_channels(self):
        """Test registering multiple channels."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            queue1 = asyncio.Queue(maxsize=10)
            queue2 = asyncio.Queue(maxsize=20)
            queue3 = asyncio.Queue(maxsize=30)

            ctx._register_channel('ch1', queue1)
            ctx._register_channel('ch2', queue2)
            ctx._register_channel('ch3', queue3)

            assert len(ctx._channels) == 3
            assert ctx._channels['ch1'] is queue1
            assert ctx._channels['ch2'] is queue2
            assert ctx._channels['ch3'] is queue3

    def test_register_channel_overwrites_existing(self):
        """Test re-registering channel overwrites previous."""
        with TemporaryDirectory() as tmpdir:
            run = MockRun(tmpdir)
            ctx = RunContext(run)

            queue1 = asyncio.Queue(maxsize=10)
            queue2 = asyncio.Queue(maxsize=20)

            ctx._register_channel('reused', queue1)
            ctx._register_channel('reused', queue2)

            assert ctx._channels['reused'] is queue2
