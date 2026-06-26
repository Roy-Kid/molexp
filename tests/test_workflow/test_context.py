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
    def test_inputs_and_config_are_not_public(self):
        # The engine still constructs the context with the data internally, but
        # a task body cannot read it off ``ctx`` — inputs bind to parameters.
        ctx = TaskContext(inputs=42, config={"k": "v"})
        for name in ("inputs", "config"):
            assert not hasattr(ctx, name), (
                f"TaskContext.{name} must be absent from the public surface; "
                f"inputs bind to the task body's typed parameters by name."
            )

    def test_workdir_is_the_only_data_surface(self):
        from pathlib import Path

        ctx = TaskContext(inputs=None, workdir=Path("/tmp/wd"))
        assert ctx.workdir == Path("/tmp/wd")
        assert TaskContext(inputs=None).workdir is None

    def test_run_context_and_deps_removed(self):
        ctx = TaskContext(inputs=None)
        for name in ("run_context", "deps"):
            with pytest.raises(AttributeError):
                getattr(ctx, name)

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

    def test_frozen_cannot_assign(self):
        ctx = TaskContext(inputs=1)
        with pytest.raises(AttributeError):
            ctx.workdir = object()  # type: ignore[misc]

    def test_state_default_none_still_warns(self):
        # state defaults to None; access warns even then (the ATTRIBUTE is deprecated).
        with pytest.warns(DeprecationWarning):
            assert TaskContext(inputs=None).state is None


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
        assert "values now arrive via ctx.inputs" in msg
        assert "ctx.state will be removed" in msg

    def test_mapping_state_snapshot_is_read_only(self):
        backing = {"x": 1}
        ctx = TaskContext(inputs=None, state=backing)
        with pytest.warns(DeprecationWarning):
            snap = ctx.state
        assert snap == {"x": 1}
        with pytest.raises(TypeError):
            snap["x"] = 2  # type: ignore[index]
        # The snapshot is a COPY — mutating it is impossible, and the backing
        # dict was never exposed.
        assert backing == {"x": 1}

    def test_workflow_state_view_read_only_legacy_patterns(self):
        from molexp.workflow._pydantic_graph.state import WorkflowState

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

    def test_state_snapshot_taken_at_access_time(self):
        """Loop-iteration overwrites stay observable: each ``ctx.state`` access
        snapshots the CURRENT results; an already-taken view does not track
        later engine mutations."""
        from molexp.workflow._pydantic_graph.state import WorkflowState

        state = WorkflowState()
        state.record("n", 1)
        ctx = TaskContext(inputs=None, state=state)
        with pytest.warns(DeprecationWarning):
            first = ctx.state
        state.record("n", 2)  # loop iteration overwrites in place
        with pytest.warns(DeprecationWarning):
            second = ctx.state
        assert first.results["n"] == 1
        assert second.results["n"] == 2

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
        assert result.status == "completed"
        assert result.outputs["b"] == 7


class TestNoProfileConfigInContextModule:
    def test_context_module_does_not_import_profile_config(self):
        import inspect

        from molexp.workflow import context as context_mod

        src = inspect.getsource(context_mod)
        assert "from molexp.profile import ProfileConfig" not in src


class TestActorContextRemoved:
    def test_actor_context_no_longer_exported(self):
        import molexp.workflow as wf
        import molexp.workflow.context as context_mod

        assert not hasattr(context_mod, "ActorContext")
        assert "ActorContext" not in getattr(wf, "__all__", ())
