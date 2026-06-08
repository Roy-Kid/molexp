"""Tests for the pure-task-context TaskContext.

After pure-task-context-02, TaskContext is a frozen plain class exposing
``inputs`` + ``config`` + a retained read-only ``state`` (minimal retention per
decision #2 — full state-elimination is a separate spec). ``run_context`` and
``deps`` are removed (accessing them raises ``AttributeError``). ``config`` is a
plain ``Mapping[str, Any]``, not a ``ProfileConfig``.
"""

from __future__ import annotations

import pytest

from molexp.workflow.context import TaskContext


class TestPureTaskContext:
    def test_public_attributes(self):
        ctx = TaskContext(inputs=42, config={"k": "v"}, state={"x": 1})
        assert ctx.inputs == 42
        assert ctx.config == {"k": "v"}
        assert ctx.state == {"x": 1}

    def test_run_context_and_deps_removed(self):
        ctx = TaskContext(inputs=None)
        for name in ("run_context", "deps"):
            with pytest.raises(AttributeError):
                getattr(ctx, name)

    def test_workspace_plumbing_removed(self):
        ctx = TaskContext(inputs=None)
        for name in ("artifact", "log", "find_asset", "checkpoint", "set_result", "get_result"):
            assert not hasattr(ctx, name), (
                f"TaskContext.{name} must be absent; capabilities flow as inputs "
                f"or via the engine's materialization layer."
            )

    def test_frozen_cannot_assign(self):
        ctx = TaskContext(inputs=1)
        with pytest.raises(AttributeError):
            ctx.inputs = 2  # type: ignore[misc]

    def test_config_is_plain_mapping_not_profile_config(self):
        from collections.abc import Mapping

        ctx = TaskContext(inputs=None, config={"epochs": 5, "dataset": "md17"})
        assert isinstance(ctx.config, Mapping)
        assert ctx.config["epochs"] == 5
        assert ctx.config["dataset"] == "md17"
        assert isinstance(ctx.config, (dict, Mapping))

    def test_default_config_is_empty_mapping(self):
        assert TaskContext(inputs=None).config == {}

    def test_state_retained_read_only_default_none(self):
        # state is retained (decision #2) for loop/branch data-flow; defaults None.
        assert TaskContext(inputs=None).state is None
        sentinel = object()
        assert TaskContext(inputs=None, state=sentinel).state is sentinel


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
