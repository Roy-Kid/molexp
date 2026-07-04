"""Tests for Workflow + WorkflowCompiler."""

from molexp.workflow import Task, WorkflowCompiler
from molexp.workflow._graph_decl import TaskRegistration
from molexp.workflow._helpers import _stable_workflow_id


class TestWorkflowDecorators:
    def test_custom_name(self):
        wf = WorkflowCompiler(name="named")

        @wf.task(name="custom_name")
        async def fn(ctx):
            return 1

        spec = wf.compile()
        assert spec._tasks[0].name == "custom_name"


class TestWorkflowAdd:
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
