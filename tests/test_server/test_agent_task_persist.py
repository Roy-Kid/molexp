"""Agent task transcript survives process restart (disk flush)."""

from __future__ import annotations

from pathlib import Path

from molexp.services.agent_task_store import (
    PersistedAgentTask,
    merge_agent_task_events,
    read_agent_task_events,
    read_agent_task_metadata,
    write_agent_task_metadata,
)


def test_merge_agent_task_events_dedupes_and_appends(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id="task-abc",
            session_id="sess-1",
            title="t",
            goal="g",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
        ),
    )
    e1 = {"type": "loop_started", "ts": "t1", "payload": {"user_input": "hi"}}
    e2 = {"type": "token_delta", "ts": "t2", "payload": {"text": "a"}}
    assert merge_agent_task_events(root, "task-abc", [e1, e2]) == 2
    assert merge_agent_task_events(root, "task-abc", [e1, e2]) == 0  # dedupe
    e3 = {"type": "loop_completed", "ts": "t3", "payload": {"text": "done"}}
    assert merge_agent_task_events(root, "task-abc", [e2, e3]) == 1
    events = read_agent_task_events(root, "task-abc")
    assert [e["type"] for e in events] == ["loop_started", "token_delta", "loop_completed"]


def test_hydrate_does_not_fail_plan_running_without_chat_runtime(tmp_path: Path) -> None:
    """Plan Mode uses PlanTask registry — missing chat runtime must not → failed."""
    from molexp.server.routes.agent_tasks import _hydrate_disk_task
    from molexp.workspace import Workspace

    root = tmp_path / "ws"
    ws = Workspace(root, name="lab")
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id="task-plan-1",
            session_id="task-plan-1",
            title="plan",
            goal="do science",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            plan_mode=True,
            active_mode="plan",
            project_id="p",
            experiment_id="e",
            active_plan_task_id=None,  # just accepted context, plan starting
        ),
    )
    task = _hydrate_disk_task(ws, read_agent_task_metadata(root, "task-plan-1"))  # type: ignore[arg-type]
    assert task.status == "running"


def test_hydrate_demotes_plan_when_plan_task_id_is_gone(tmp_path: Path) -> None:
    """Stored plan-task id with empty registry → failed (server restart)."""
    from molexp.server.routes.agent_tasks import _hydrate_disk_task
    from molexp.workspace import Workspace

    root = tmp_path / "ws"
    ws = Workspace(root, name="lab")
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id="task-plan-dead",
            session_id="task-plan-dead",
            title="plan",
            goal="do science",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            plan_mode=True,
            active_mode="plan",
            project_id="p",
            experiment_id="e",
            active_plan_task_id="plan-task-missing",
        ),
    )
    task = _hydrate_disk_task(ws, read_agent_task_metadata(root, "task-plan-dead"))  # type: ignore[arg-type]
    assert task.status == "failed"
    meta = read_agent_task_metadata(root, "task-plan-dead")
    assert meta is not None and meta.status == "failed"
    events = read_agent_task_events(root, "task-plan-dead")
    assert any(
        e.get("type") == "error" and (e.get("payload") or {}).get("stage") == "server_restart"
        for e in events
    )


def test_hydrate_demotes_waiting_approval_without_live_plan(tmp_path: Path) -> None:
    from molexp.server.routes.agent_tasks import _hydrate_disk_task
    from molexp.workspace import Workspace

    root = tmp_path / "ws"
    ws = Workspace(root, name="lab")
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id="task-plan-wait",
            session_id="task-plan-wait",
            title="plan",
            goal="do science",
            status="waiting_approval",
            created_at="2026-01-01T00:00:00+00:00",
            plan_mode=True,
            active_mode="plan",
            project_id="p",
            experiment_id="e",
            active_plan_task_id="plan-task-gone",
        ),
    )
    task = _hydrate_disk_task(ws, read_agent_task_metadata(root, "task-plan-wait"))  # type: ignore[arg-type]
    assert task.status == "failed"


def test_aggressive_reap_clears_zombie_running_without_plan_id(tmp_path: Path) -> None:
    """POST path must not 409 forever on disk-only running + no plan id."""
    from molexp.server.routes.agent_tasks import _reap_stale_in_flight
    from molexp.workspace import Workspace

    root = tmp_path / "ws"
    ws = Workspace(root, name="lab")
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id="task-zombie",
            session_id="task-zombie",
            title="plan",
            goal="do science",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            plan_mode=True,
            active_mode="plan",
            project_id="p",
            experiment_id="e",
            active_plan_task_id=None,
        ),
    )
    meta = read_agent_task_metadata(root, "task-zombie")
    assert meta is not None
    reaped = _reap_stale_in_flight(ws, meta, aggressive=True)
    assert reaped.status == "failed"


def test_cancel_plan_zombie_without_runtime_is_idempotent(tmp_path: Path) -> None:
    """Cancel must free disk-only plan tasks (no chat/plan registry entry)."""
    import asyncio

    from molexp.server.routes.agent_tasks import cancel_agent_task
    from molexp.workspace import Workspace

    root = tmp_path / "ws"
    ws = Workspace(root, name="lab")
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id="task-cancel-z",
            session_id="task-cancel-z",
            title="plan",
            goal="do science",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            plan_mode=True,
            active_mode="plan",
            project_id="p",
            experiment_id="e",
            active_plan_task_id=None,
        ),
    )

    async def _run() -> None:
        resp = await cancel_agent_task("task-cancel-z", workspace=ws)
        assert resp.message == "cancelled"

    asyncio.run(_run())
    meta = read_agent_task_metadata(root, "task-cancel-z")
    assert meta is not None
    assert meta.status == "cancelled"


def test_flush_runtime_turn_writes_events_and_status(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from molexp.server.agent_runtime.runtime import AgentSessionRuntime
    from molexp.server.agent_runtime.turn import AgentTurn

    root = tmp_path / "ws"
    root.mkdir()
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id="task-xyz",
            session_id="sess-xyz",
            title="plan pe",
            goal="goal",
            status="running",
            created_at=datetime.now(UTC).isoformat(),
            active_mode="chat",
        ),
    )

    class _FakeEvent:
        def model_dump(self, mode: str = "json") -> dict:
            return {
                "kind": "loop_completed",
                "timestamp": "2026-01-01T00:00:01Z",
                "text": "ok",
            }

    runtime = AgentSessionRuntime(
        runner=SimpleNamespace(),  # type: ignore[arg-type]
        session=SimpleNamespace(session_id="sess-xyz"),  # type: ignore[arg-type]
        goal="goal",
        created_at=datetime.now(UTC).isoformat(),
        workspace_root=str(root),
    )
    runtime.task_id = "task-xyz"
    turn = AgentTurn()
    turn.events.append(_FakeEvent())  # type: ignore[arg-type]
    turn.status = "completed"
    runtime._flush_turn_to_disk(turn)

    events = read_agent_task_events(root, "task-xyz")
    assert len(events) == 1
    assert events[0]["type"] == "loop_completed"
    meta = read_agent_task_metadata(root, "task-xyz")
    assert meta is not None
    assert meta.status == "completed"
