"""Tests for the skill store and Skill template materialization."""

from __future__ import annotations

import json

import pytest

from molexp.plugins.agent_pydanticai.skills import (
    SKILLS_FILE,
    Skill,
    SkillScope,
    SkillStore,
)


@pytest.mark.unit
def test_skill_materialize_substitutes_placeholders():
    skill = Skill(
        id="s1",
        name="Plot",
        goal_template="plot {{metric}} vs {{param}} in {{project}}",
        constraints=["scope=project"],
        success_criteria=["a chart is produced"],
    )
    rendered = skill.materialize(
        {"metric": "energy", "param": "temperature", "project": "X"}
    )
    assert rendered["description"] == "plot energy vs temperature in X"
    assert rendered["constraints"] == ["scope=project"]
    assert rendered["success_criteria"] == ["a chart is produced"]


@pytest.mark.unit
def test_skill_materialize_missing_param_raises():
    skill = Skill(id="s1", name="Plot", goal_template="needs {{a}} and {{b}}")
    with pytest.raises(KeyError, match="Missing parameter 'b'"):
        skill.materialize({"a": "x"})


@pytest.mark.unit
def test_skill_materialize_no_placeholders_passes_through():
    skill = Skill(id="s1", name="Static", goal_template="just go")
    assert skill.materialize()["description"] == "just go"


@pytest.mark.unit
def test_store_create_persists_to_disk(tmp_path):
    store = SkillStore(tmp_path)
    skill = store.create(name="Energy plot", goal_template="plot energy")
    assert skill.name == "Energy plot"
    raw = json.loads((tmp_path / SKILLS_FILE).read_text())
    assert len(raw) == 1
    assert raw[0]["id"] == skill.id


@pytest.mark.unit
def test_store_list_returns_all_records(tmp_path):
    store = SkillStore(tmp_path)
    store.create(name="A", goal_template="a")
    store.create(name="B", goal_template="b")
    workspace_skills = store.list_scope(SkillScope.WORKSPACE)
    assert {s.name for s in workspace_skills} == {"A", "B"}


@pytest.mark.unit
def test_store_list_all_includes_builtin_plan(tmp_path):
    """list_all surfaces the builtin /plan skill alongside user skills."""
    store = SkillStore(tmp_path)
    store.create(name="A", goal_template="a")
    items = store.list_all()
    ids = {s.id for s in items}
    assert "builtin-plan" in ids
    plan = next(s for s in items if s.id == "builtin-plan")
    assert plan.scope == SkillScope.BUILTIN
    assert plan.builtin is True


@pytest.mark.unit
def test_store_get_returns_none_for_missing(tmp_path):
    store = SkillStore(tmp_path)
    assert store.get("nope") is None


@pytest.mark.unit
def test_store_update_changes_fields_and_bumps_timestamp(tmp_path):
    store = SkillStore(tmp_path)
    skill = store.create(name="A", goal_template="a")
    updated = store.update(skill.id, name="A renamed", tags=["x"])
    assert updated.name == "A renamed"
    assert updated.tags == ["x"]
    assert updated.updated_at >= skill.updated_at


@pytest.mark.unit
def test_store_update_unknown_field_raises(tmp_path):
    store = SkillStore(tmp_path)
    skill = store.create(name="A", goal_template="a")
    with pytest.raises(ValueError, match="Unknown skill fields"):
        store.update(skill.id, bogus="x")


@pytest.mark.unit
def test_store_update_missing_id_raises(tmp_path):
    store = SkillStore(tmp_path)
    with pytest.raises(KeyError, match="not found"):
        store.update("nope", name="x")


@pytest.mark.unit
def test_store_delete_returns_false_when_missing(tmp_path):
    store = SkillStore(tmp_path)
    assert store.delete("nope") is False


@pytest.mark.unit
def test_store_delete_removes_record(tmp_path):
    store = SkillStore(tmp_path)
    skill = store.create(name="A", goal_template="a")
    assert store.delete(skill.id) is True
    assert store.list_scope(SkillScope.WORKSPACE) == []


@pytest.mark.unit
def test_store_handles_corrupt_json(tmp_path):
    (tmp_path / SKILLS_FILE).write_text("not json")
    store = SkillStore(tmp_path)
    assert store.list_scope(SkillScope.WORKSPACE) == []
    # New writes still succeed (old garbage is overwritten).
    store.create(name="A", goal_template="a")
    assert len(store.list_scope(SkillScope.WORKSPACE)) == 1


@pytest.mark.unit
def test_store_create_rejects_duplicate_explicit_id(tmp_path):
    store = SkillStore(tmp_path)
    store.create(name="A", goal_template="a", skill_id="dup")
    with pytest.raises(ValueError, match="already exists"):
        store.create(name="B", goal_template="b", skill_id="dup")
