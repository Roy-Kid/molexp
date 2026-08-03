"""Plan Mode experiment context resolve / ensure (agent-tasks messages)."""

from __future__ import annotations

from pathlib import Path

from molexp.server.routes.agent_tasks import (
    _ensure_experiment_context,
    _parse_context_answer,
    _resolve_experiment_context,
)
from molexp.services.agent_task_store import (
    PersistedAgentTask,
    read_agent_task_events,
    read_agent_task_metadata,
    write_agent_task_metadata,
)
from molexp.workspace import Workspace


def test_parse_context_answer_splits_slash_and_colon() -> None:
    assert _parse_context_answer("PE_Plan / random_walk") == ("PE_Plan", "random_walk")
    assert _parse_context_answer("PE_Plan:random_walk") == ("PE_Plan", "random_walk")
    assert _parse_context_answer("  only-exp  ") == ("", "only-exp")


def test_resolve_matches_slug_and_underscore_aliases(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws", name="lab")
    project = ws.add_project("PE CorseGrain Rg Plan")
    project.add_experiment("random walk scaling")

    # Free-text underscored form should hit slugified on-disk ids.
    hit = _resolve_experiment_context(ws, "PE_CorseGrain_Rg_Plan / random_walk_scaling")
    assert hit is not None
    assert hit[0] == project.id
    assert hit[1] == "random-walk-scaling" or hit[1] == project.list_experiments()[0].id


def test_ensure_creates_missing_project_and_experiment(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws", name="lab")
    assert ws.list_projects() == []

    resolved, err = _ensure_experiment_context(ws, "PE_CorseGrain_Rg_Plan / random_walk_scaling")
    assert err is None
    assert resolved is not None
    project_id, experiment_id = resolved
    assert ws.has_project(project_id) or any(p.id == project_id for p in ws.list_projects())
    project = next(p for p in ws.list_projects() if p.id == project_id)
    assert any(e.id == experiment_id for e in project.list_experiments())


def test_ensure_idempotent_on_second_reply(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws", name="lab")
    first, err1 = _ensure_experiment_context(ws, "demo / exp1")
    second, err2 = _ensure_experiment_context(ws, "demo / exp1")
    assert err1 is None and err2 is None
    assert first == second
    assert len(ws.list_projects()) == 1


def test_ensure_ambiguous_experiment_only_name(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws", name="lab")
    ws.add_project("a").add_experiment("shared")
    ws.add_project("b").add_experiment("shared")

    resolved, err = _ensure_experiment_context(ws, "shared")
    assert resolved is None
    assert err is not None
    assert "more than one" in err


def test_post_message_mode_plan_does_not_run_chat(tmp_path: Path) -> None:
    """Composer toggle sends mode=plan; must not fall into the chat runtime."""
    import asyncio

    from molexp.server.routes.agent_tasks import post_agent_task_message
    from molexp.server.schemas import UserMessageCreateRequest

    root = tmp_path / "ws"
    ws = Workspace(root, name="lab")
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id="task-switch-plan",
            session_id="task-switch-plan",
            title="was chat",
            goal="hello",
            status="completed",
            created_at="2026-01-01T00:00:00+00:00",
            plan_mode=False,
            active_mode="chat",
        ),
    )

    async def _run() -> None:
        resp = await post_agent_task_message(
            "task-switch-plan",
            UserMessageCreateRequest(content="draft a PE plan", mode="plan"),
            workspace=ws,
        )
        assert "plan" in resp.message.lower() or "context" in resp.message.lower()

    asyncio.run(_run())
    meta = read_agent_task_metadata(root, "task-switch-plan")
    assert meta is not None
    assert meta.active_mode == "plan"
    assert meta.plan_mode is True
    assert meta.status == "awaiting_user"
    events = read_agent_task_events(root, "task-switch-plan")
    assert any(e.get("type") == "clarification_required" for e in events)
    # Must not have spun a chat session under a new session_id.
    assert meta.session_id == "task-switch-plan"
