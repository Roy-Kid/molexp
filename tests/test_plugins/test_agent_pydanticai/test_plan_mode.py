"""Plan-mode integration: catalog scoping by builtin skill + persisted goal fields."""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai.builtin_skills import get_builtin_skill
from molexp.plugins.agent_pydanticai._pydantic_ai.catalog import MolexpToolCatalog
from molexp.plugins.agent_pydanticai.sessions_store import (
    get_persisted_session,
    write_session_metadata,
)
from molexp.plugins.agent_pydanticai.tool_registry import builtin_tool_registry


@pytest.mark.unit
def test_plan_skill_does_not_filter_the_native_surface():
    """Plan mode does NOT restrict tools — the agent needs the full
    surface (including write-tool signatures and ``list_task_types``)
    to draft a workflow IR. The constraint is on the output: the agent
    must finalize via ``exit_plan_mode``, which is enforced by the
    system-prompt addendum, not by hiding tools."""
    plan_skill = get_builtin_skill("builtin-plan")
    assert plan_skill is not None

    plan_names = _toolset_function_names(MolexpToolCatalog(skill=plan_skill).build())
    full_names = _toolset_function_names(MolexpToolCatalog(skill=None).build())
    assert plan_names == full_names, (
        f"plan mode dropped tools that the unrestricted catalog has: "
        f"{full_names - plan_names}"
    )


@pytest.mark.unit
def test_plan_mode_catalog_includes_exit_plan_mode_and_write_tools():
    """Spot-check the catalog still surfaces both inspection and write
    tools under PLAN_SKILL — the agent uses write-tool signatures (e.g.
    ``set_workflow_from_ir``) to draft the IR, then commits the plan
    via ``exit_plan_mode`` rather than calling those write tools."""
    plan_skill = get_builtin_skill("builtin-plan")
    catalog = MolexpToolCatalog(skill=plan_skill)
    names = _toolset_function_names(catalog.build())
    for required in (
        "list_projects",
        "list_task_types",
        "set_workflow_from_ir",
        "execute_run",
        "ask_user",
        "exit_plan_mode",
    ):
        assert required in names, f"plan-mode catalog missing {required!r}"


@pytest.mark.unit
def test_full_catalog_includes_write_tools():
    catalog = MolexpToolCatalog(skill=None)
    names = _toolset_function_names(catalog.build())
    write_names = {s.name for s in builtin_tool_registry().filter(mutates=True)}
    assert write_names.issubset(names)


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
