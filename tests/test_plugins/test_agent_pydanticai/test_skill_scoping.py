"""Tests for Skill.matches_tool, three-tier SkillStore, and PLAN_SKILL filter."""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai.builtin_skills import get_builtin_skill
from molexp.plugins.agent_pydanticai.skills import Skill, SkillScope, SkillStore


# ── Skill.matches_tool ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_matches_tool_empty_allow_means_allow_all():
    skill = Skill(id="s", name="any", goal_template="x")
    assert skill.matches_tool("anything") is True
    assert skill.matches_tool("set_workflow_from_ir") is True


@pytest.mark.unit
def test_matches_tool_allow_glob_narrows_surface():
    skill = Skill(
        id="s", name="read", goal_template="x",
        allowed_tools=["list_*", "get_*"],
    )
    assert skill.matches_tool("list_projects") is True
    assert skill.matches_tool("get_run_status") is True
    assert skill.matches_tool("set_workflow_from_ir") is False


@pytest.mark.unit
def test_matches_tool_denial_wins_over_allow():
    skill = Skill(
        id="s", name="x", goal_template="x",
        allowed_tools=["list_*"],
        denied_tools=["list_runs"],
    )
    assert skill.matches_tool("list_projects") is True
    assert skill.matches_tool("list_runs") is False


@pytest.mark.unit
def test_matches_tool_supports_mcp_namespacing():
    skill = Skill(
        id="s", name="x", goal_template="x",
        allowed_tools=["mcp:python.*"],
    )
    assert skill.matches_tool("mcp:python.run") is True
    assert skill.matches_tool("mcp:other.run") is False


# ── PLAN_SKILL filter contract ──────────────────────────────────────────────


@pytest.mark.unit
def test_plan_skill_does_not_restrict_tools():
    """Plan mode constrains the OUTPUT (must finalize via exit_plan_mode),
    not the input surface — the agent needs ``list_task_types`` + every
    write tool's signature to draft a valid IR. A read-only filter
    would actively prevent that."""
    plan = get_builtin_skill("builtin-plan")
    assert plan is not None
    assert plan.allowed_tools == []
    assert plan.denied_tools == []
    # Sanity-check via the matcher: every tool is exposed.
    for tool_name in (
        "list_projects",
        "get_run_status",
        "ask_user",
        "exit_plan_mode",
        "set_workflow_from_ir",
        "execute_run",
        "create_experiment",
    ):
        assert plan.matches_tool(tool_name) is True, (
            f"plan mode unexpectedly hides {tool_name!r}"
        )


@pytest.mark.unit
def test_plan_skill_metadata():
    plan = get_builtin_skill("builtin-plan")
    assert plan is not None
    assert plan.builtin is True
    assert plan.scope == SkillScope.BUILTIN
    assert plan.requires_exit_tool == "exit_plan_mode"
    assert plan.default_plan_mode is True


# ── Three-tier SkillStore ───────────────────────────────────────────────────


@pytest.mark.unit
def test_store_lists_all_three_tiers_in_display_order(tmp_path):
    user_home = tmp_path / "home"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SkillStore(workspace, user_home_dir=user_home)
    store.create(name="user-A", goal_template="u", scope=SkillScope.USER)
    store.create(name="ws-A", goal_template="w", scope=SkillScope.WORKSPACE)

    items = store.list_all()
    scopes = [s.scope for s in items]
    # Builtin first, then user, then workspace.
    assert scopes[0] == SkillScope.BUILTIN
    user_idx = next(i for i, s in enumerate(items) if s.name == "user-A")
    ws_idx = next(i for i, s in enumerate(items) if s.name == "ws-A")
    assert user_idx < ws_idx


@pytest.mark.unit
def test_store_find_by_slash_workspace_shadows_user_shadows_builtin(tmp_path):
    user_home = tmp_path / "home"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SkillStore(workspace, user_home_dir=user_home)

    store.create(
        name="user-X",
        goal_template="u",
        slash_name="x",
        scope=SkillScope.USER,
    )
    resolved_user = store.find_by_slash("x")
    assert resolved_user is not None
    assert resolved_user.scope == SkillScope.USER

    store.create(
        name="ws-X",
        goal_template="w",
        slash_name="x",
        scope=SkillScope.WORKSPACE,
    )
    resolved_ws = store.find_by_slash("x")
    assert resolved_ws is not None
    # Workspace shadows user.
    assert resolved_ws.scope == SkillScope.WORKSPACE
    assert resolved_ws.name == "ws-X"


@pytest.mark.unit
def test_store_delete_only_acts_on_specified_scope(tmp_path):
    user_home = tmp_path / "home"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SkillStore(workspace, user_home_dir=user_home)
    store.create(name="A", goal_template="a", scope=SkillScope.USER)
    ws_skill = store.create(name="B", goal_template="b", scope=SkillScope.WORKSPACE)

    # Deleting from workspace must NOT touch user-tier records.
    assert store.delete(ws_skill.id, scope=SkillScope.WORKSPACE) is True
    user_records = store.list_scope(SkillScope.USER)
    assert {s.name for s in user_records} == {"A"}


@pytest.mark.unit
def test_store_rejects_builtin_writes(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SkillStore(workspace, user_home_dir=tmp_path / "home")

    with pytest.raises(ValueError, match="cannot be created"):
        store.create(
            name="evil", goal_template="x", scope=SkillScope.BUILTIN
        )

    with pytest.raises(ValueError, match="immutable"):
        store.update("builtin-plan", scope=SkillScope.BUILTIN, name="x")

    with pytest.raises(ValueError, match="cannot be deleted"):
        store.delete("builtin-plan", scope=SkillScope.BUILTIN)


@pytest.mark.unit
def test_store_get_resolves_across_tiers(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SkillStore(workspace, user_home_dir=tmp_path / "home")

    # Builtin lookup works without any disk writes.
    plan = store.get("builtin-plan")
    assert plan is not None
    assert plan.builtin is True

    # User-tier lookup works after a create at that scope.
    user_skill = store.create(name="U", goal_template="u", scope=SkillScope.USER)
    found = store.get(user_skill.id)
    assert found is not None
    assert found.scope == SkillScope.USER
