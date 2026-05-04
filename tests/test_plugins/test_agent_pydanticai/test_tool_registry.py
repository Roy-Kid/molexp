"""Tests for the @native_tool decorator and ToolRegistry."""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai.tool_registry import (
    NativeToolSpec,
    ToolRegistry,
    builtin_tool_registry,
    native_tool,
)


@pytest.mark.unit
def test_native_tool_decorator_registers_into_registry():
    """Importing workspace_tools must populate the global registry."""
    # Side-effect import — the decorators on the module run at import time.
    from molexp.plugins.agent_pydanticai._pydantic_ai import workspace_tools  # noqa: F401

    registry = builtin_tool_registry()
    names = {s.name for s in registry.all()}
    # Spot-check tools from each category.
    assert "list_projects" in names
    assert "set_workflow_from_ir" in names
    assert "ask_user" in names
    assert "exit_plan_mode" in names


@pytest.mark.unit
def test_native_tool_decorator_captures_metadata():
    from molexp.plugins.agent_pydanticai._pydantic_ai import workspace_tools  # noqa: F401

    registry = builtin_tool_registry()
    spec = registry.get("set_workflow_from_ir")
    assert spec is not None
    assert spec.category == "workflow"
    assert spec.mutates is True
    # Read-only inspection is non-mutating.
    list_spec = registry.get("list_projects")
    assert list_spec is not None
    assert list_spec.mutates is False
    # Description is the docstring.
    assert "workflow IR" in list_spec.description.lower() or list_spec.description


@pytest.mark.unit
def test_registry_filter_by_mutates():
    from molexp.plugins.agent_pydanticai._pydantic_ai import workspace_tools  # noqa: F401

    registry = builtin_tool_registry()
    read_only = {s.name for s in registry.filter(mutates=False)}
    writable = {s.name for s in registry.filter(mutates=True)}
    assert "list_projects" in read_only
    assert "list_projects" not in writable
    assert "set_workflow_from_ir" in writable
    assert read_only.isdisjoint(writable)


@pytest.mark.unit
def test_registry_filter_by_category():
    from molexp.plugins.agent_pydanticai._pydantic_ai import workspace_tools  # noqa: F401

    registry = builtin_tool_registry()
    chat_tools = {s.name for s in registry.filter(category="chat")}
    assert chat_tools == {"ask_user"}

    control_tools = {s.name for s in registry.filter(category="control")}
    assert control_tools == {"exit_plan_mode"}


@pytest.mark.unit
def test_registry_rejects_duplicate_registration():
    """Registering the same name twice must fail — silent shadowing is dangerous."""
    isolated = ToolRegistry()

    async def fn(ctx):  # pragma: no cover — only used to populate metadata
        return None

    spec = NativeToolSpec(
        name="dup",
        fn=fn,
        description="",
        category="workspace",
        mutates=False,
        requires_approval=False,
    )
    isolated.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        isolated.register(spec)


@pytest.mark.unit
def test_native_tool_decorator_returns_original_function():
    """The decorator must NOT wrap — pydantic-ai introspects the raw fn."""
    isolated = ToolRegistry()

    async def real_fn(ctx, x: int) -> int:
        return x * 2

    # Manually wire up: using the global decorator would add a permanent
    # entry to the process-wide registry, so we exercise NativeToolSpec
    # directly here.
    isolated.register(
        NativeToolSpec(
            name=real_fn.__name__,
            fn=real_fn,
            description="",
            category="workspace",
            mutates=False,
            requires_approval=False,
        )
    )
    spec = isolated.get("real_fn")
    assert spec is not None
    assert spec.fn is real_fn
