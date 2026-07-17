"""agent_task_store layout — colocated under agent/."""

from __future__ import annotations

from pathlib import Path

from molexp.services.agent_task_store import (
    AGENT_HOME_NAME,
    TASKS_SUBDIR,
    PersistedAgentTask,
    delete_agent_task,
    list_agent_task_metadata,
    read_agent_task_metadata,
    write_agent_task_metadata,
)


def _task(task_id: str = "task-abc") -> PersistedAgentTask:
    return PersistedAgentTask(
        task_id=task_id,
        session_id="sess1",
        title="Hello",
        goal="do a thing",
        status="running",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_write_lands_under_interactive_tasks(tmp_path: Path) -> None:
    write_agent_task_metadata(tmp_path, _task())
    path = tmp_path / AGENT_HOME_NAME / TASKS_SUBDIR / "task-abc" / "metadata.json"
    assert path.is_file()
    loaded = read_agent_task_metadata(tmp_path, "task-abc")
    assert loaded is not None
    assert loaded.session_id == "sess1"
    assert loaded.goal == "do a thing"


def test_turn_mode_and_plan_execution_round_trip(tmp_path: Path) -> None:
    task = PersistedAgentTask(
        **{
            **_task().__dict__,
            "active_mode": "plan",
            "active_turn_id": "turn-1",
            "active_plan_task_id": "plan-1",
            "pending_plan_draft": "revise it",
        }
    )
    write_agent_task_metadata(tmp_path, task)
    loaded = read_agent_task_metadata(tmp_path, task.task_id)
    assert loaded is not None
    assert loaded.active_mode == "plan"
    assert loaded.active_turn_id == "turn-1"
    assert loaded.active_plan_task_id == "plan-1"
    assert loaded.pending_plan_draft == "revise it"


def test_list_and_delete(tmp_path: Path) -> None:
    write_agent_task_metadata(tmp_path, _task("task-a"))
    write_agent_task_metadata(tmp_path, _task("task-b"))
    ids = {t.task_id for t in list_agent_task_metadata(tmp_path)}
    assert ids == {"task-a", "task-b"}

    assert delete_agent_task(tmp_path, "task-a") is True
    assert read_agent_task_metadata(tmp_path, "task-a") is None
    assert not (tmp_path / AGENT_HOME_NAME / TASKS_SUBDIR / "task-a").exists()
    assert {t.task_id for t in list_agent_task_metadata(tmp_path)} == {"task-b"}
