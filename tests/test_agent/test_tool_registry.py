"""Phase 1a: ToolRegistry + native_tool tag-only decorator."""

from __future__ import annotations

import pytest

from molexp.agent import (
    DuplicateToolError,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    native_tool,
)
from molexp.agent.tools.policy import ToolPolicy
from molexp.agent.tools.registry import (
    get_native_spec,
    is_native_tool,
)


def _spec(name: str, **kwargs) -> ToolSpec:
    base = {
        "name": name,
        "description": f"tool {name}",
        "input_schema": {"type": "object", "properties": {}},
    }
    base.update(kwargs)
    return ToolSpec(**base)


async def _stub(args: dict, ctx: ToolContext) -> ToolResult:
    return ToolResult(ok=True, value=args)


def test_register_and_get() -> None:
    registry = ToolRegistry()
    registry.register(_spec("native:read"), _stub)
    found = registry.get("native:read")
    assert found is not None
    assert found.spec.name == "native:read"


def test_register_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(_spec("native:read"), _stub)
    with pytest.raises(DuplicateToolError):
        registry.register(_spec("native:read"), _stub)


def test_list_filters_by_policy() -> None:
    registry = ToolRegistry()
    registry.register(_spec("native:read"), _stub)
    registry.register(_spec("native:write_file", mutates=True), _stub)
    only_reads = registry.list(ToolPolicy(allow=("native:read",)))
    names = {s.name for s in only_reads}
    assert names == {"native:read"}


def test_list_deny_overrides_allow() -> None:
    registry = ToolRegistry()
    registry.register(_spec("native:read"), _stub)
    registry.register(_spec("native:write_file", mutates=True), _stub)
    visible = registry.list(ToolPolicy(allow=("native:*",), deny=("native:write_*",)))
    assert {s.name for s in visible} == {"native:read"}


def test_schemas_returns_model_facing_subset() -> None:
    registry = ToolRegistry()
    registry.register(_spec("native:read", category="workspace", mutates=False), _stub)
    schemas = registry.schemas()
    assert len(schemas) == 1
    assert schemas[0].name == "native:read"
    # ToolSchema only carries name/description/input_schema — no
    # harness-internal flags like ``mutates`` or ``requires_approval``.
    schema_fields = set(type(schemas[0]).model_fields)
    assert schema_fields == {"name", "description", "input_schema"}
    assert isinstance(schemas[0].input_schema, dict)


def test_native_tool_decorator_only_tags() -> None:
    spec = _spec("native:tagged")

    @native_tool(spec)
    async def my_tool(args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True)

    # Tag is present, but no module-level singleton was touched.
    assert is_native_tool(my_tool)
    assert get_native_spec(my_tool) is spec

    # Construct two independent registries; neither sees the tool until
    # the service-bootstrap code explicitly registers it.
    a = ToolRegistry()
    b = ToolRegistry()
    assert "native:tagged" not in a
    assert "native:tagged" not in b

    a.register(spec, my_tool)
    assert "native:tagged" in a
    assert "native:tagged" not in b


def test_policy_visibility_and_approval() -> None:
    spec = _spec("native:run", mutates=True)
    policy = ToolPolicy(approval_overrides={"native:run": False})
    assert policy.visible(spec) is True
    # mutating tool would normally need approval; override flips it off
    assert policy.needs_approval(spec) is False

    policy2 = ToolPolicy()
    assert policy2.needs_approval(spec) is True
