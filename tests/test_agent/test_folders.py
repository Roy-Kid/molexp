"""Workspace-native agent Concept types — ``Agent`` + ``AgentSession``.

After the OKF rehome onto ``molexp.workspace.Folder`` (wsokf-06), the agent's
two on-disk Concept types are workspace Folders (not knowledge Folders) that
register through the shared concept-type registry and make their rich
``meta.yaml`` the identity authority. This module pins:

* subclass identity (``molexp.workspace.Folder``);
* ``@concept_type`` registration + registry resolution;
* ``meta.yaml`` round-trip of the rich ``AgentMeta`` / ``AgentSessionMeta``;
* session CRUD sugar over the workspace ``cls=``-keyed CRUD (disk is truth);
* ``concept_from_dir`` reconstruction through the workspace walker;
* binary ``messages.jsonl`` routed through the injectable filesystem.
"""

from __future__ import annotations

from molexp.agent.folders import (
    AGENT_KIND,
    AGENT_SESSION_KIND,
    Agent,
    AgentSession,
)
from molexp.agent.folders_metadata import AgentMeta, AgentSessionMeta
from molexp.knowledge.types import resolve_concept_type
from molexp.workspace import Folder as WSFolder
from molexp.workspace import folder as ws_folder

# ── ac-001: subclass identity ────────────────────────────────────────────────


def test_agent_classes_subclass_workspace_folder() -> None:
    assert issubclass(Agent, WSFolder)
    assert issubclass(AgentSession, WSFolder)


# ── ac-003: registry resolution ──────────────────────────────────────────────


def test_concept_types_resolve_from_registry() -> None:
    assert resolve_concept_type(AGENT_KIND, WSFolder) is Agent
    assert resolve_concept_type(AGENT_SESSION_KIND, WSFolder) is AgentSession


# ── ac-004 / ac-009: meta.yaml authority round-trip ──────────────────────────


def test_agent_meta_yaml_is_rich_authority(tmp_path) -> None:
    agent = Agent(
        name="reviewer",
        root_path=tmp_path / "lab",
        system_prompt="be terse",
        model="deepseek:chat",
        tier="cheap",
        description="a reviewer",
    )
    agent.materialize()
    meta_file = tmp_path / "lab" / "reviewer" / "meta.yaml"
    assert meta_file.exists()
    raw = meta_file.read_text()
    assert "type: agent.agent" in raw
    assert "be terse" in raw

    reloaded = Agent(name="reviewer", root_path=tmp_path / "lab")
    meta = reloaded.read_agent_meta()
    assert isinstance(meta, AgentMeta)
    assert meta.type == AGENT_KIND
    assert meta.system_prompt == "be terse"
    assert meta.model == "deepseek:chat"
    assert meta.tier == "cheap"
    assert reloaded.system_prompt == "be terse"


def test_session_meta_yaml_round_trips(tmp_path) -> None:
    agent = Agent(name="reviewer", root_path=tmp_path / "lab")
    session = agent.add_session("chat-1", goal_summary="solve X", status="running")
    meta = session.read_session_meta()
    assert isinstance(meta, AgentSessionMeta)
    assert meta.type == AGENT_SESSION_KIND
    assert meta.goal_summary == "solve X"
    assert meta.status == "running"


# ── ac-005: CRUD sugar over workspace cls= CRUD; disk is truth ────────────────


def test_session_crud_round_trips_through_disk(tmp_path) -> None:
    agent = Agent(name="reviewer", root_path=tmp_path / "lab")
    agent.add_session("chat-1", goal_summary="solve X", status="running")

    reloaded = Agent(name="reviewer", root_path=tmp_path / "lab")
    assert [s.name for s in reloaded.list_sessions()] == ["chat-1"]
    assert reloaded.has_session("chat-1")
    session = reloaded.get_session("chat-1")
    assert isinstance(session, AgentSession)
    assert session.goal_summary == "solve X"
    assert session.status == "running"


def test_remove_session(tmp_path) -> None:
    agent = Agent(name="reviewer", root_path=tmp_path / "lab")
    agent.add_session("chat-1")
    assert agent.has_session("chat-1")
    agent.remove_session("chat-1")
    reloaded = Agent(name="reviewer", root_path=tmp_path / "lab")
    assert not reloaded.has_session("chat-1")
    assert reloaded.list_sessions() == []


# ── ac-006: concept_from_dir reconstruction via the workspace walker ─────────


def test_concept_from_dir_rebuilds_agent_session(tmp_path) -> None:
    agent = Agent(name="reviewer", root_path=tmp_path / "lab")
    session = agent.add_session("chat-1", goal_summary="solve X")
    session_dir = session.resolve()

    fresh = Agent(name="reviewer", root_path=tmp_path / "lab")
    rebuilt = ws_folder.concept_from_dir(session_dir, fresh)
    assert isinstance(rebuilt, AgentSession)
    assert rebuilt.read_session_meta().goal_summary == "solve X"


# ── ac-007: binary messages routed through the injectable fs ─────────────────


def test_messages_path_is_flat_and_binary(tmp_path) -> None:
    agent = Agent(name="reviewer", root_path=tmp_path / "lab")
    session = agent.add_session("chat-1")
    assert str(session.messages_path).endswith("reviewer/chat-1/messages.jsonl")
    # empty session: read returns the empty tuple, write(()) is a no-op-remove
    assert session.read_messages() == ()
    session.write_messages(())
    assert not (tmp_path / "lab" / "reviewer" / "chat-1" / "messages.jsonl").exists()
