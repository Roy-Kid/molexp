"""Agent-layer Concept types — :mod:`molexp.agent.folders` (Agent + AgentSession).

Rehomed onto :class:`molexp.workspace.Folder` (wsokf-06): both agent Concepts
register with the shared concept-type registry and make their rich ``meta.yaml``
the identity authority. This pins registry resolution, the typed ``meta.yaml``
round-trip, the session CRUD sugar (disk is truth), and the flat binary
``messages.jsonl`` persistence contract.
"""

from __future__ import annotations

from molexp.agent.folders import AGENT_KIND, AGENT_SESSION_KIND, Agent, AgentSession
from molexp.agent.folders_metadata import AgentMeta
from molexp.knowledge.types import resolve_concept_type
from molexp.workspace import Folder as WSFolder


def test_agent_concept_kinds_resolve_to_subclasses() -> None:
    """OKF registry maps each agent kind back to its Folder subclass."""
    assert resolve_concept_type(AGENT_KIND, WSFolder) is Agent
    assert resolve_concept_type(AGENT_SESSION_KIND, WSFolder) is AgentSession


class TestAgent:
    def test_meta_yaml_round_trips_rich_authority(self, tmp_path) -> None:
        agent = Agent(
            name="reviewer",
            root_path=tmp_path / "lab",
            system_prompt="be terse",
            model="deepseek:chat",
            tier="cheap",
            description="a reviewer",
        )
        agent.materialize()
        raw = (tmp_path / "lab" / "reviewer" / "meta.yaml").read_text()
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

    def test_session_crud_round_trips_through_disk(self, tmp_path) -> None:
        agent = Agent(name="reviewer", root_path=tmp_path / "lab")
        agent.add_session("chat-1", goal_summary="solve X", status="running")

        reloaded = Agent(name="reviewer", root_path=tmp_path / "lab")
        assert [s.name for s in reloaded.list_sessions()] == ["chat-1"]
        assert reloaded.has_session("chat-1")
        session = reloaded.get_session("chat-1")
        assert isinstance(session, AgentSession)
        assert session.goal_summary == "solve X"
        assert session.status == "running"

    def test_remove_session_deletes_from_disk(self, tmp_path) -> None:
        agent = Agent(name="reviewer", root_path=tmp_path / "lab")
        agent.add_session("chat-1")
        assert agent.has_session("chat-1")
        agent.remove_session("chat-1")

        reloaded = Agent(name="reviewer", root_path=tmp_path / "lab")
        assert not reloaded.has_session("chat-1")
        assert reloaded.list_sessions() == []


class TestAgentSession:
    def test_messages_path_flat_and_empty_write_removes_file(self, tmp_path) -> None:
        agent = Agent(name="reviewer", root_path=tmp_path / "lab")
        session = agent.add_session("chat-1")
        assert str(session.messages_path).endswith("reviewer/chat-1/messages.jsonl")
        # Empty session: read is the empty tuple; write(()) is a no-op remove.
        assert session.read_messages() == ()
        session.write_messages(())
        assert not (tmp_path / "lab" / "reviewer" / "chat-1" / "messages.jsonl").exists()
