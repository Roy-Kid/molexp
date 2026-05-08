"""Tests for ``SubsystemStore`` and ``Workspace.subsystem_store`` (spec ac-001 / ac-002).

`SubsystemStore` vendors a per-kind directory under
``<workspace_root>/.subsystems/<kind>/``. Construction is side-effect-free;
``.dir()`` and ``.file(name)`` mkdir lazily. Kind validation rejects
empty / path-traversal / leading-dot / uppercase / non-ASCII strings.
"""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace
from molexp.workspace.subsystem import SubsystemStore


class TestSubsystemStorePaths:
    def test_dir_returns_kind_subdir_under_subsystems_root(self, tmp_path):
        store = SubsystemStore(tmp_path, "agent.sessions")
        assert store.dir() == tmp_path / ".subsystems" / "agent.sessions"

    def test_file_returns_named_file_inside_kind_subdir(self, tmp_path):
        store = SubsystemStore(tmp_path, "agent.skills")
        assert store.file("skills.json") == (
            tmp_path / ".subsystems" / "agent.skills" / "skills.json"
        )


class TestSubsystemStoreLazyMkdir:
    def test_construction_is_side_effect_free(self, tmp_path):
        SubsystemStore(tmp_path, "agent.sessions")
        assert not (tmp_path / ".subsystems").exists()

    def test_dir_call_creates_kind_subdir_lazily(self, tmp_path):
        store = SubsystemStore(tmp_path, "agent.sessions")
        path = store.dir()
        assert path.exists()
        assert path.is_dir()

    def test_file_call_creates_parent_kind_subdir_lazily(self, tmp_path):
        store = SubsystemStore(tmp_path, "agent.tools")
        target = store.file("tools.json")
        assert target.parent.exists()
        assert target.parent.is_dir()
        # `file()` does not create the file itself, only its parent dir.
        assert not target.exists()

    def test_dir_is_idempotent_on_repeat_call(self, tmp_path):
        store = SubsystemStore(tmp_path, "agent.mcp")
        store.dir()
        # Second call must not error or recreate.
        path_again = store.dir()
        assert path_again == tmp_path / ".subsystems" / "agent.mcp"


class TestSubsystemStoreKindValidation:
    @pytest.mark.parametrize(
        "bad_kind",
        [
            "",  # empty
            "agent/sessions",  # path separator
            "../escape",  # parent escape
            ".hidden",  # leading dot (collides with dotfile convention)
            "AGENT.SESSIONS",  # uppercase forbidden
            "agent sessions",  # space
            "agent.sessions/",  # trailing separator
            "agent\\sessions",  # backslash
        ],
    )
    def test_construction_rejects_invalid_kind(self, tmp_path, bad_kind):
        with pytest.raises(ValueError):
            SubsystemStore(tmp_path, bad_kind)

    @pytest.mark.parametrize(
        "ok_kind",
        [
            "agent.sessions",
            "agent.skills",
            "agent",  # single segment is fine
            "a",  # single char
            "agent-tools",  # hyphen
            "agent_tools",  # underscore
            "agent.tools.v2",  # multi-segment
        ],
    )
    def test_construction_accepts_valid_kind(self, tmp_path, ok_kind):
        # Should not raise.
        SubsystemStore(tmp_path, ok_kind)


class TestWorkspaceSubsystemStoreCache:
    def test_subsystem_store_returns_same_instance_per_kind(self, tmp_path):
        ws = Workspace(tmp_path)
        first = ws.subsystem_store("agent.skills")
        second = ws.subsystem_store("agent.skills")
        assert first is second

    def test_subsystem_store_distinguishes_kinds(self, tmp_path):
        ws = Workspace(tmp_path)
        sessions_store = ws.subsystem_store("agent.sessions")
        skills_store = ws.subsystem_store("agent.skills")
        assert sessions_store is not skills_store
        assert sessions_store.dir().name == "agent.sessions"
        assert skills_store.dir().name == "agent.skills"

    def test_subsystem_store_is_lazy_no_dir_until_accessed(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.subsystem_store("agent.sessions")
        assert not (tmp_path / ".subsystems").exists()

    def test_subsystem_store_validates_kind(self, tmp_path):
        ws = Workspace(tmp_path)
        with pytest.raises(ValueError):
            ws.subsystem_store("agent/bad")
