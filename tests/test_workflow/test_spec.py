"""Tests for WorkflowSpec / WorkflowBuilder / DSL."""


from molexp.workflow import (
    Task,
    TaskContext,
    WorkflowBuilder,
    workflow,
)
from molexp.workflow.spec import TaskRegistration, _stable_workflow_id


class TestWorkflowDSL:
    def test_single_task(self):
        wf = workflow(name="simple")

        @wf.task
        async def fetch(ctx):
            return 1

        spec = wf.build()
        assert spec.name == "simple"
        assert len(spec._tasks) == 1
        assert spec._tasks[0].name == "fetch"

    def test_dependencies(self):
        wf = workflow(name="chain")

        @wf.task
        async def a(ctx):
            return 1

        @wf.task(depends_on=["a"])
        async def b(ctx):
            return 2

        spec = wf.build()
        assert spec._tasks[1].depends_on == ["a"]

    def test_custom_name(self):
        wf = workflow(name="named")

        @wf.task(name="custom_name")
        async def fn(ctx):
            return 1

        spec = wf.build()
        assert spec._tasks[0].name == "custom_name"

    def test_actor_decorator(self):
        wf = workflow(name="stream")

        @wf.actor
        async def streamer(ctx):
            yield 1

        spec = wf.build()
        assert spec._tasks[0].is_actor is True


class TestWorkflowBuilder:
    def test_add_task(self):
        class DoubleTask(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return 2

        spec = WorkflowBuilder(name="oop").add(DoubleTask()).build()
        assert len(spec._tasks) == 1
        assert spec._tasks[0].name == "double"

    def test_add_external_runnable(self):
        class External:
            async def execute(self, ctx) -> int:
                return 42

        spec = WorkflowBuilder(name="ext").add(External(), name="ext").build()
        assert spec._tasks[0].name == "ext"

    def test_chaining(self):
        class A(Task):
            async def execute(self, ctx):
                return 1

        class B(Task):
            async def execute(self, ctx):
                return 2

        spec = (
            WorkflowBuilder(name="chain")
            .add(A())
            .add(B(), depends_on=["a"])
            .build()
        )
        assert len(spec._tasks) == 2

    def test_strip_task_suffix(self):
        class FetchTask(Task):
            async def execute(self, ctx):
                return 1

        spec = WorkflowBuilder(name="strip").add(FetchTask()).build()
        assert spec._tasks[0].name == "fetch"


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
