"""Plan-mode integration: catalog filter + persisted goal fields."""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai._pydantic_ai.catalog import MolexpToolCatalog
from molexp.plugins.agent_pydanticai._pydantic_ai.workspace_tools import (
    READ_ONLY_TOOLS,
    WRITE_TOOLS,
    get_read_only_tools,
)
from molexp.plugins.agent_pydanticai.sessions_store import (
    get_persisted_session,
    write_session_metadata,
)


@pytest.mark.unit
def test_read_only_catalog_excludes_write_tools():
    catalog = MolexpToolCatalog(read_only=True)
    toolset = catalog.build()
    # FunctionToolset stores tools in a private dict — accept either layout.
    names = _toolset_function_names(toolset)
    write_names = {t.__name__ for t in WRITE_TOOLS}
    assert write_names.isdisjoint(names), (
        f"plan-mode catalog leaked write tools: {names & write_names}"
    )
    read_names = {t.__name__ for t in READ_ONLY_TOOLS}
    assert read_names.issubset(names)


@pytest.mark.unit
def test_full_catalog_includes_write_tools():
    catalog = MolexpToolCatalog(read_only=False)
    toolset = catalog.build()
    names = _toolset_function_names(toolset)
    write_names = {t.__name__ for t in WRITE_TOOLS}
    assert write_names.issubset(names)


@pytest.mark.unit
def test_get_read_only_tools_includes_chat_plumbing():
    names = {t.__name__ for t in get_read_only_tools()}
    assert "ask_user" in names


@pytest.mark.unit
def test_session_metadata_round_trips_plan_mode_and_skill_id(tmp_path):
    write_session_metadata(
        tmp_path,
        "sess-abc123",
        status="completed",
        goal_description="run experiment",
        plan_mode=True,
        skill_id="skill-1",
        skill_instructions="extra rule",
        instructions_override="full custom",
    )
    summary = get_persisted_session(tmp_path, "sess-abc123")
    assert summary is not None
    assert summary.plan_mode is True
    assert summary.skill_id == "skill-1"


# ── helpers ─────────────────────────────────────────────────────────────────


def _toolset_function_names(toolset) -> set[str]:
    """Best-effort extraction of registered tool function names."""
    candidates = (
        getattr(toolset, "tools", None),
        getattr(toolset, "_tools", None),
        getattr(toolset, "functions", None),
    )
    for source in candidates:
        if source is None:
            continue
        if isinstance(source, dict):
            return set(source.keys())
        if isinstance(source, (list, tuple, set)):
            names: set[str] = set()
            for item in source:
                name = getattr(item, "name", None) or getattr(item, "__name__", None)
                if name:
                    names.add(name)
            if names:
                return names
    raise AssertionError(
        f"Could not introspect toolset {type(toolset)!r}; "
        "update _toolset_function_names for the new pydantic-ai layout."
    )
