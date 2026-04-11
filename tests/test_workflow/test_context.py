"""Tests for TaskContext — workflow/workspace bridge."""

from molexp.workflow.context import TaskContext, ActorContext


class TestTaskContext:
    def test_workflow_properties(self):
        ctx = TaskContext(state={"x": 1}, deps="my_deps", inputs=42)
        assert ctx.state == {"x": 1}
        assert ctx.deps == "my_deps"
        assert ctx.inputs == 42

    def test_workspace_methods_return_none_without_run(self):
        ctx = TaskContext(state=None, deps=None, inputs=None)
        assert ctx.save_artifact("test", {}) is None
        assert ctx.get_artifact_path("test") is None
        assert ctx.find_asset("missing") is None
        assert ctx.checkpoint() is None
        assert ctx.get_result("key") is None

    def test_set_result_noop_without_run(self):
        ctx = TaskContext(state=None, deps=None, inputs=None)
        ctx.set_result("key", "value")  # should not raise


class TestTaskContextRunContext:
    def test_run_context_none_without_run(self):
        ctx = TaskContext(state=None, deps=None, inputs=None)
        assert ctx.run_context is None
        assert ctx.dry_run is False

    def test_run_context_returns_attached(self):
        sentinel = object()
        ctx = TaskContext(state=None, deps=None, inputs=None, run_context=sentinel)
        assert ctx.run_context is sentinel

    def test_dry_run_is_explicit_execution_flag(self):
        ctx = TaskContext(state=None, deps=None, inputs=None, dry_run=True)
        assert ctx.dry_run is True


class TestActorContext:
    def test_inherits_task_context(self):
        ctx = ActorContext(state={"x": 1}, deps=None, inputs=None)
        assert ctx.state == {"x": 1}
