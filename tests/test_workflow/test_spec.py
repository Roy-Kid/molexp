"""Tests for Workflow + WorkflowCompiler."""

from molexp.workflow import Task, TaskContext, WorkflowCompiler
from molexp.workflow._graph_decl import TaskRegistration
from molexp.workflow._helpers import _stable_workflow_id


class TestWorkflowDecorators:
    def test_single_task(self):
        wf = WorkflowCompiler(name="simple")

        @wf.task
        async def fetch(ctx):
            return 1

        spec = wf.compile()
        assert spec.name == "simple"
        assert len(spec._tasks) == 1
        assert spec._tasks[0].name == "fetch"

    def test_dependencies(self):
        wf = WorkflowCompiler(name="chain")

        @wf.task
        async def a(ctx):
            return 1

        @wf.task(depends_on=["a"])
        async def b(ctx):
            return 2

        spec = wf.compile()
        assert spec._tasks[1].depends_on == ["a"]

    def test_custom_name(self):
        wf = WorkflowCompiler(name="named")

        @wf.task(name="custom_name")
        async def fn(ctx):
            return 1

        spec = wf.compile()
        assert spec._tasks[0].name == "custom_name"

    def test_actor_decorator(self):
        wf = WorkflowCompiler(name="stream")

        @wf.actor
        async def streamer(ctx):
            yield 1

        spec = wf.compile()
        assert spec._tasks[0].is_actor is True


class TestWorkflowAdd:
    def test_add_task(self):
        class DoubleTask(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return 2

        spec = WorkflowCompiler(name="oop").add(DoubleTask()).compile()
        assert len(spec._tasks) == 1
        assert spec._tasks[0].name == "double"

    def test_add_external_runnable(self):
        class External:
            async def execute(self, ctx) -> int:
                return 42

        spec = WorkflowCompiler(name="ext").add(External(), name="ext").compile()
        assert spec._tasks[0].name == "ext"

    def test_chaining(self):
        class A(Task):
            async def execute(self, ctx):
                return 1

        class B(Task):
            async def execute(self, ctx):
                return 2

        spec = WorkflowCompiler(name="chain").add(A()).add(B(), depends_on=["a"]).compile()
        assert len(spec._tasks) == 2

    def test_strip_task_suffix(self):
        class FetchTask(Task):
            async def execute(self, ctx):
                return 1

        spec = WorkflowCompiler(name="strip").add(FetchTask()).compile()
        assert spec._tasks[0].name == "fetch"

    def test_mix_decorator_and_add(self):
        class PostTask(Task):
            async def execute(self, ctx):
                return "post"

        wf = WorkflowCompiler(name="mixed")

        @wf.task
        async def pre(ctx):
            return "pre"

        wf.add(PostTask(), depends_on=["pre"])
        spec = wf.compile()
        assert [t.name for t in spec._tasks] == ["pre", "post"]


class TestStableWorkflowId:
    def test_deterministic(self):
        regs = [TaskRegistration("a", lambda: None, []), TaskRegistration("b", lambda: None, ["a"])]
        id1 = _stable_workflow_id("test", regs)
        id2 = _stable_workflow_id("test", regs)
        assert id1 == id2

    def test_different_name_different_id(self):
        regs = [TaskRegistration("a", lambda: None, [])]
        id1 = _stable_workflow_id("wf1", regs)
        id2 = _stable_workflow_id("wf2", regs)
        assert id1 != id2
