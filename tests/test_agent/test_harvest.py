"""agent-record-export-05 — harvest_session + export_session_zip."""

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


def test_harvest_session_writes_knowledge_and_emits(
    agent_session: tuple[Workspace, Agent, AgentSession],
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


def test_harvest_session_empty_narrative_raises(
    agent_session: tuple[Workspace, Agent, AgentSession],
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


def test_export_session_zip_uses_archive(
    agent_session: tuple[Workspace, Agent, AgentSession],
) -> None:
    _ws, _agent, session = agent_session
    # drop a file the zip must include
    path = Path(session.resolve()) / "entries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")

    data = export_session_zip(session)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert any(n.endswith("entries.jsonl") or n == "entries.jsonl" for n in names)
