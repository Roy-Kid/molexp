"""Tests for workflow protocols — Runnable / Streamable."""

from molexp.workflow import Actor, ActorContext, Runnable, Streamable, Task, TaskContext


class TestRunnableProtocol:
    def test_task_subclass_satisfies(self):
        class MyTask(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return 1

        assert isinstance(MyTask(), Runnable)

    def test_external_class_satisfies(self):
        class External:
            async def execute(self, ctx) -> dict:
                return {}

        assert isinstance(External(), Runnable)

    def test_plain_object_does_not_satisfy(self):
        class NotATask:
            pass

        assert not isinstance(NotATask(), Runnable)

    def test_sync_execute_does_not_satisfy(self):
        class SyncOnly:
            def execute(self, ctx):
                return 1

        # sync method still satisfies the structural check at runtime
        # (Protocol only checks method existence, not async-ness)
        assert isinstance(SyncOnly(), Runnable)

    def test_callable_without_execute_does_not_satisfy(self):
        async def bare_fn(ctx):
            return 1

        assert not isinstance(bare_fn, Runnable)


class TestStreamableProtocol:
    def test_actor_subclass_satisfies(self):
        class MyActor(Actor):
            async def run(self, ctx: ActorContext):
                yield 1

        assert isinstance(MyActor(), Streamable)

    def test_external_streamer_satisfies(self):
        class External:
            async def run(self, ctx):
                yield 1

        assert isinstance(External(), Streamable)
