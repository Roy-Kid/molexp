"""Session harvest + export — :mod:`molexp.agent.harvest`.

``harvest_session`` turns an on-disk :class:`AgentSession` into a sourced
``KnowledgeItem`` (via the workspace knowledge-write spine); ``export_session_zip``
archives the session folder.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from molexp.agent.folders import Agent, AgentSession
from molexp.agent.harvest import export_session_zip, harvest_session
from molexp.workspace import KnowledgeItem, read_workspace_events
from molexp.workspace.workspace import Workspace


@pytest.fixture
def agent_session(tmp_path: Path) -> tuple[Workspace, Agent, AgentSession]:
    ws = Workspace(root=tmp_path / "ws")
    ws.materialize()
    agent = Agent(name="agent", root_path=ws.root)
    agent.materialize()
    session = agent.add_session("s1")
    session.materialize()
    return ws, agent, session


class TestHarvestSession:
    def test_writes_knowledge_item_and_emits_event(
        self, agent_session: tuple[Workspace, Agent, AgentSession]
    ) -> None:
        ws, _agent, session = agent_session
        item = harvest_session(
            session,
            kind="Observation",
            narrative="User explored the workspace.",
            created_by="tester",
            host=ws,  # knowledge + event spine live on the workspace root
        )
        assert isinstance(item, KnowledgeItem)
        assert "explored" in item.body()
        events = read_workspace_events(ws.root, type="knowledge.created")
        assert len(events) == 1

    def test_empty_narrative_raises(
        self, agent_session: tuple[Workspace, Agent, AgentSession]
    ) -> None:
        _ws, agent, session = agent_session
        with pytest.raises(ValueError):
            harvest_session(
                session,
                kind="Observation",
                narrative="  ",
                created_by="tester",
                host=agent,
            )


class TestExportSessionZip:
    def test_archives_session_folder_contents(
        self, agent_session: tuple[Workspace, Agent, AgentSession]
    ) -> None:
        _ws, _agent, session = agent_session
        # Drop a file the archive must include.
        path = Path(session.resolve()) / "entries.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

        data = export_session_zip(session)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            assert any(n.endswith("entries.jsonl") for n in names)
