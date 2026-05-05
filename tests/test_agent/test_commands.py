"""Tests for the slash-command parser + reserved-name validation."""

from __future__ import annotations

import pytest

from molexp.agent.state.commands import parse
from molexp.agent.state.skills import RESERVED_SLASH_NAMES, SkillStore

# ── Builtin commands ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_plan_builtin(tmp_path):
    store = SkillStore(tmp_path)
    parsed = parse("/plan", store)
    assert parsed.kind == "builtin"
    assert parsed.name == "plan"
    assert parsed.plan_mode is True


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(RESERVED_SLASH_NAMES))
def test_each_reserved_name_resolves_to_builtin(tmp_path, name):
    parsed = parse(f"/{name}", SkillStore(tmp_path))
    assert parsed.kind == "builtin"
    assert parsed.name == name


@pytest.mark.unit
def test_help_builtin_does_not_force_plan_mode(tmp_path):
    parsed = parse("/help", SkillStore(tmp_path))
    assert parsed.plan_mode is False


# ── Skill commands ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_skill_with_args(tmp_path):
    store = SkillStore(tmp_path)
    store.create(
        name="Plot",
        goal_template="plot {{metric}} vs {{param}}",
        slash_name="plot",
    )
    parsed = parse("/plot metric=energy param=temp", store)
    assert parsed.kind == "skill"
    assert parsed.parameters == {"metric": "energy", "param": "temp"}
    assert parsed.plan_mode is False


@pytest.mark.unit
def test_skill_default_plan_mode_propagates(tmp_path):
    store = SkillStore(tmp_path)
    store.create(
        name="Audit",
        goal_template="audit {{scope}}",
        slash_name="audit",
        default_plan_mode=True,
    )
    parsed = parse("/audit scope=project", store)
    assert parsed.kind == "skill"
    assert parsed.plan_mode is True


@pytest.mark.unit
def test_skill_with_quoted_value(tmp_path):
    store = SkillStore(tmp_path)
    store.create(
        name="Greet",
        goal_template="say {{message}}",
        slash_name="greet",
    )
    parsed = parse('/greet message="hello world"', store)
    assert parsed.kind == "skill"
    assert parsed.parameters == {"message": "hello world"}


@pytest.mark.unit
def test_unknown_slash_command_yields_error(tmp_path):
    parsed = parse("/nope", SkillStore(tmp_path))
    assert parsed.kind == "error"
    assert "/nope" in parsed.error


@pytest.mark.unit
def test_missing_required_param_yields_error(tmp_path):
    store = SkillStore(tmp_path)
    store.create(
        name="Plot",
        goal_template="plot {{metric}}",
        slash_name="plot",
    )
    parsed = parse("/plot", store)
    assert parsed.kind == "error"
    assert "metric" in parsed.error


@pytest.mark.unit
def test_arg_without_equals_is_error(tmp_path):
    store = SkillStore(tmp_path)
    store.create(name="Plot", goal_template="plot", slash_name="plot")
    parsed = parse("/plot foo", store)
    assert parsed.kind == "error"
    assert "key=value" in parsed.error


@pytest.mark.unit
def test_non_slash_input_returns_error(tmp_path):
    parsed = parse("hello", SkillStore(tmp_path))
    assert parsed.kind == "error"


# ── Reserved-name + uniqueness validation in SkillStore ─────────────────────


@pytest.mark.unit
def test_create_rejects_reserved_slash_name(tmp_path):
    store = SkillStore(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        store.create(name="P", goal_template="x", slash_name="plan")


@pytest.mark.unit
def test_create_rejects_invalid_slash_name(tmp_path):
    store = SkillStore(tmp_path)
    with pytest.raises(ValueError, match="Invalid slash_name"):
        store.create(name="P", goal_template="x", slash_name="Bad Name")


@pytest.mark.unit
def test_create_rejects_duplicate_slash_name(tmp_path):
    store = SkillStore(tmp_path)
    store.create(name="A", goal_template="a", slash_name="dup")
    with pytest.raises(ValueError, match="already used"):
        store.create(name="B", goal_template="b", slash_name="dup")


@pytest.mark.unit
def test_update_can_change_slash_name(tmp_path):
    store = SkillStore(tmp_path)
    skill = store.create(name="A", goal_template="a", slash_name="one")
    updated = store.update(skill.id, slash_name="two")
    assert updated.slash_name == "two"
    assert store.find_by_slash("one") is None
    assert store.find_by_slash("two") is not None


@pytest.mark.unit
def test_find_by_slash_skips_blank_slash_names(tmp_path):
    store = SkillStore(tmp_path)
    store.create(name="Hidden", goal_template="x")  # no slash_name
    assert store.find_by_slash("") is None
    assert store.find_by_slash("Hidden") is None
