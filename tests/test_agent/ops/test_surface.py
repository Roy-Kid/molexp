"""Declared tool surface — subset policy, not a deny-list."""

from __future__ import annotations

import pytest

from molexp.agent.ops.builtins import (
    CHAT_TOOL_NAMES,
    FULL_TOOL_NAMES,
    declared_requirements,
    tool_names_for_surface,
)
from molexp.agent.ops.surface import (
    CHAT_SURFACE,
    FULL_SURFACE,
    LIFECYCLE_SURFACE,
    SurfaceKey,
    classify_undeclared,
    deny_reason,
    is_workspace_mutator_name,
    required_keys,
    surface_for_mode,
)


def test_surface_for_mode() -> None:
    assert surface_for_mode("chat") is CHAT_SURFACE
    assert surface_for_mode("full") is FULL_SURFACE
    assert surface_for_mode("lifecycle") is LIFECYCLE_SURFACE


def test_unknown_surface_raises() -> None:
    with pytest.raises(ValueError, match="unknown ops surface"):
        surface_for_mode("readonly")
    with pytest.raises(ValueError, match="unknown ops surface"):
        surface_for_mode("archive")


def test_chat_surface_excludes_archive_and_mutate() -> None:
    assert SurfaceKey.ARCHIVE not in CHAT_SURFACE.keys
    assert SurfaceKey.LIFECYCLE not in CHAT_SURFACE.keys
    assert SurfaceKey.WORKSPACE_MUTATE not in CHAT_SURFACE.keys
    assert SurfaceKey.ARCHIVE in FULL_SURFACE.keys
    assert SurfaceKey.LIFECYCLE in LIFECYCLE_SURFACE.keys
    assert SurfaceKey.WORKSPACE_MUTATE not in FULL_SURFACE.keys
    assert SurfaceKey.WORKSPACE_MUTATE not in LIFECYCLE_SURFACE.keys


def test_tool_names_derived_from_required_subset() -> None:
    assert tool_names_for_surface("chat") == CHAT_TOOL_NAMES
    assert "workspace_ensure" not in CHAT_TOOL_NAMES
    assert "run_land" not in CHAT_TOOL_NAMES
    assert "cancel_run" not in FULL_TOOL_NAMES
    assert "cancel_run" in tool_names_for_surface("lifecycle")
    assert "harvest_run" in tool_names_for_surface("lifecycle")
    assert "workspace_ensure" in tool_names_for_surface("full")


def test_declared_archive_tools_are_not_classified_as_mutators() -> None:
    """Classifier is MCP-only; archive builtins declare ARCHIVE themselves."""
    assert not is_workspace_mutator_name("workspace_ensure")
    assert not is_workspace_mutator_name("run_land")
    assert not is_workspace_mutator_name("workspace_inspect")
    assert not is_workspace_mutator_name("code_run")
    assert not is_workspace_mutator_name("molcrafts_search")
    declared = declared_requirements()
    assert declared["workspace_ensure"] == frozenset({SurfaceKey.ARCHIVE})
    assert declared["run_land"] == frozenset({SurfaceKey.ARCHIVE})
    assert required_keys("workspace_ensure", declared=declared) == frozenset({SurfaceKey.ARCHIVE})


def test_classifier_tags_undeclared_mcp_mutators() -> None:
    assert is_workspace_mutator_name("molexp_molexp_add_project")
    assert is_workspace_mutator_name("molexp_add_experiment")
    assert is_workspace_mutator_name("molexp_create_run")
    assert classify_undeclared("molexp_molexp_create_run") == frozenset(
        {SurfaceKey.WORKSPACE_MUTATE}
    )
    assert classify_undeclared("molcrafts_search") == frozenset()


def test_chat_surface_denies_mutators_and_allows_scratch() -> None:
    declared = declared_requirements()
    denied = deny_reason(CHAT_SURFACE, "molexp_molexp_create_run", declared=declared)
    assert denied is not None
    assert "workspace_mutate" in denied
    assert "chat" in denied
    assert deny_reason(CHAT_SURFACE, "code_write", declared=declared) is None
    archive = deny_reason(CHAT_SURFACE, "workspace_ensure", declared=declared)
    assert archive is not None
    assert "archive" in archive


def test_full_surface_allows_archive_and_still_denies_mcp_mutators() -> None:
    declared = declared_requirements()
    assert deny_reason(FULL_SURFACE, "workspace_ensure", declared=declared) is None
    assert deny_reason(FULL_SURFACE, "run_land", declared=declared) is None
    mcp = deny_reason(FULL_SURFACE, "molexp_add_project", declared=declared)
    assert mcp is not None


def test_injected_board_tools_have_empty_requirement() -> None:
    """Plan injects board tools onto the chat surface — they must not be gated."""
    declared = declared_requirements()
    for name in ("place_task", "run_capability", "bind_board"):
        assert deny_reason(CHAT_SURFACE, name, declared=declared) is None, name
