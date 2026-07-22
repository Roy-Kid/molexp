"""Tests for the dataflow-by-name :class:`molexp.workflow.context.TaskContext`.

After the dataflow-by-name refactor, ``ctx.inputs`` / ``ctx.config`` and the old
workspace-plumbing surface are GONE: a task body receives its inputs as typed
parameters bound by name, and the only data surface left on ``ctx`` is
``workdir``. ``ctx.state`` is in staged removal — it emits a
``DeprecationWarning`` and returns a READ-ONLY :class:`ReadOnlyStateView`.
"""

from __future__ import annotations

import pytest

from molexp.workflow.context import TaskContext


class TestTaskContext:
    def test_workspace_plumbing_surface_is_absent(self):
        """The pre-refactor data/capability surface is gone from ``ctx``."""
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

    def test_state_access_emits_deprecation_warning_with_migration_message(self):
        ctx = TaskContext(inputs=None, state={"x": 1})
        with pytest.warns(DeprecationWarning) as record:
            _ = ctx.state
        msg = str(record[0].message)
        assert "values now bind to named task parameters" in msg
        assert "ctx.state will be removed" in msg


class TestReadOnlyStateView:
    """``ctx.state`` returns a frozen, read-only snapshot of engine state:
    legacy reads still resolve, mutation is impossible, and the underlying
    engine state is never touched."""

    def test_reads_legacy_values_and_blocks_mutation(self):
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
