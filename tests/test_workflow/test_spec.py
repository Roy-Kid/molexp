"""Tests for WorkflowCompiler task registration/naming + the stable workflow id."""

from __future__ import annotations

from molexp.workflow import Task, WorkflowCompiler
from molexp.workflow._graph_decl import TaskRegistration
from molexp.workflow._helpers import _stable_workflow_id


class TestWorkflowCompiler:
    def test_decorator_name_override(self):
        wf = WorkflowCompiler(name="named")

        @wf.task(name="custom_name")
        async def fn(ctx):
            return 1

        spec = wf.compile()
        assert spec._tasks[0].name == "custom_name"

    def test_add_strips_task_suffix_and_snake_cases(self):
        class FetchTask(Task):
            async def execute(self, ctx):
                return 1

        spec = WorkflowCompiler(name="strip").add(FetchTask()).compile()
        assert spec._tasks[0].name == "fetch"

    def test_mixed_decorator_and_add_preserves_registration_order(self):
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
    def test_is_deterministic(self):
        regs = [
            TaskRegistration("a", lambda: None, []),
            TaskRegistration("b", lambda: None, ["a"]),
        ]
        assert _stable_workflow_id("test", regs) == _stable_workflow_id("test", regs)
