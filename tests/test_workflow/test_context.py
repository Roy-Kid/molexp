"""Tests for the dataflow-by-name TaskContext.

After the dataflow-by-name refactor, ``ctx.inputs`` and ``ctx.config`` are GONE
from the public surface: a task body receives its inputs as typed parameters
bound by name (see ``node._bind_call_args``). The only data surface left on
``ctx`` is ``workdir`` (a content-addressed scratch ``Path``). ``run_context``
and ``deps`` were already removed (accessing them raises ``AttributeError``).

``state`` is in staged removal: ``ctx.state`` emits a ``DeprecationWarning`` on
access and returns a READ-ONLY snapshot — user code can no longer mutate engine
state through it. See :class:`TestDeprecatedStateChannel`.
"""

from __future__ import annotations

import pytest

from molexp.workflow.context import TaskContext


class TestPureTaskContext:
    def test_workspace_plumbing_removed(self):
        ctx = TaskContext(inputs=None)
        for name in (
            "inputs",
            "config",
            "artifact",
            "log",
            "find_asset",
            "checkpoint",
            "set_result",
            "get_result",
        ):
            assert not hasattr(ctx, name), (
                f"TaskContext.{name} must be absent; inputs bind to parameters and "
                f"capabilities flow via the engine's materialization layer."
            )


class TestDeprecatedStateChannel:
    """Staged removal of ``ctx.state`` (pure-task-context state-elimination).

    Values now arrive via ``ctx.inputs`` (values-on-edges engine); ``ctx.state``
    is a deprecated, read-only escape hatch until hard removal.
    """

    def test_state_emits_deprecation_warning_with_migration_message(self):
        ctx = TaskContext(inputs=None, state={"x": 1})
        with pytest.warns(DeprecationWarning) as record:
            _ = ctx.state
        msg = str(record[0].message)
        assert "values now bind to named task parameters" in msg
        assert "ctx.state will be removed" in msg

    def test_workflow_state_view_read_only_legacy_patterns(self):
        from molexp.workflow._engine.state import WorkflowState

        state = WorkflowState()
        state.record("tick", 41)
        ctx = TaskContext(inputs=None, state=state)
        with pytest.warns(DeprecationWarning):
            view = ctx.state
        # Legacy read patterns still return correct values.
        assert view.results.get("tick") == 41
        assert view.results["tick"] == 41
        assert view.results.get("missing") is None
        assert "tick" in view.completed
        assert view.failed is False
        assert view.error is None
        # Mutation through the view is impossible.
        with pytest.raises(TypeError):
            view.results["tick"] = 99  # type: ignore[index]
        with pytest.raises(AttributeError):
            view.results = {}  # type: ignore[misc]
        with pytest.raises(AttributeError):
            view.failed = True  # type: ignore[misc]
        # Engine state was untouched.
        assert state.results["tick"] == 41
        assert state.failed is False

    @pytest.mark.asyncio
    async def test_legacy_engine_read_warns_and_returns_correct_value(self):
        """A task body still reading ``ctx.state.results`` mid-run gets the
        correct upstream value (plus the DeprecationWarning)."""
        from molexp.workflow import WorkflowCompiler, WorkflowRuntime

        wf = WorkflowCompiler(name="legacy-state-read", entry="a")

        @wf.task
        async def a(ctx) -> int:
            return 7

        @wf.task(depends_on=["a"])
        async def b(ctx) -> int:
            with pytest.warns(DeprecationWarning):
                legacy = ctx.state.results["a"]
            return legacy

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs["b"] == 7
