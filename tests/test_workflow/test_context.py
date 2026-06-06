"""Tests for the slimmed TaskContext.

After the rectification, TaskContext exposes exactly five attributes:
``state``, ``deps``, ``inputs``, ``config``, ``run_context``. Workspace
plumbing (``artifact``, ``log``, ``find_asset``, ``checkpoint``,
``set_result``, ``get_result``) is gone. ``config`` is a plain
``Mapping[str, Any]``, not a ``ProfileConfig``.
"""

from __future__ import annotations

from molexp.workflow.context import TaskContext


class TestSlimmedTaskContext:
    def test_five_public_attributes(self):
        ctx = TaskContext(state={"x": 1}, deps="d", inputs=42)
        assert ctx.state == {"x": 1}
        assert ctx.deps == "d"
        assert ctx.inputs == 42
        assert ctx.config == {}
        assert ctx.run_context is None

    def test_workspace_plumbing_removed(self):
        ctx = TaskContext(state=None, deps=None, inputs=None)
        for name in ("artifact", "log", "find_asset", "checkpoint", "set_result", "get_result"):
            assert not hasattr(ctx, name), (
                f"TaskContext.{name} must be removed; workspace plumbing now "
                f"flows through opaque ``run_context`` instead."
            )

    def test_config_is_plain_mapping_not_profile_config(self):
        from collections.abc import Mapping

        ctx = TaskContext(
            state=None, deps=None, inputs=None, config={"epochs": 5, "dataset": "md17"}
        )
        assert isinstance(ctx.config, Mapping)
        assert ctx.config["epochs"] == 5
        assert ctx.config["dataset"] == "md17"

        # Crucially: it must accept a plain dict (no ProfileConfig adapter).
        assert isinstance(ctx.config, (dict, Mapping))

    def test_default_config_is_empty_mapping(self):
        ctx = TaskContext(state=None, deps=None, inputs=None)
        assert ctx.config == {}

    def test_run_context_is_opaque_passthrough(self):
        sentinel = object()
        ctx = TaskContext(state=None, deps=None, inputs=None, run_context=sentinel)
        assert ctx.run_context is sentinel

    def test_run_context_is_arbitrary_duck_typed_object(self):
        class Anything:
            work_dir = "/tmp/anywhere"
            config = {}  # noqa: RUF012
            run = None

        obj = Anything()
        ctx = TaskContext(state=None, deps=None, inputs=None, run_context=obj)
        assert ctx.run_context is obj


class TestNoProfileConfigInContextModule:
    def test_context_module_does_not_import_profile_config(self):
        import inspect

        from molexp.workflow import context as context_mod

        src = inspect.getsource(context_mod)
        assert "from molexp.profile import ProfileConfig" not in src
        assert "ProfileConfig" not in src or "# " in src.split("ProfileConfig")[0][-2:]


class TestActorContextRemoved:
    def test_actor_context_no_longer_exported(self):
        """ActorContext was collapsed into TaskContext — the never-implemented
        receive/send channel primitives it carried were removed (P1-2)."""
        import molexp.workflow as wf
        import molexp.workflow.context as context_mod

        assert not hasattr(context_mod, "ActorContext")
        assert "ActorContext" not in getattr(wf, "__all__", ())
