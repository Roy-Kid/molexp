"""agent_task_store — on-disk agent-task metadata, colocated under ``agent/``."""

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


class TestAgentTaskMetadata:
    def test_write_read_round_trips_all_fields_under_the_tasks_home(self, tmp_path: Path) -> None:
        task = PersistedAgentTask(
            task_id="task-abc",
            session_id="sess1",
            title="Hello",
            goal="do a thing",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            active_mode="plan",
            active_turn_id="turn-1",
            active_plan_task_id="plan-1",
            pending_plan_draft="revise it",
        )
        write_agent_task_metadata(tmp_path, task)
        path = tmp_path / AGENT_HOME_NAME / TASKS_SUBDIR / "task-abc" / "task.json"
        assert path.is_file()
        assert not (
            tmp_path / AGENT_HOME_NAME / TASKS_SUBDIR / "task-abc" / "metadata.json"
        ).exists()

        loaded = read_agent_task_metadata(tmp_path, "task-abc")
        assert loaded is not None
        assert loaded.session_id == "sess1"
        assert loaded.goal == "do a thing"
        assert loaded.active_mode == "plan"
        assert loaded.active_turn_id == "turn-1"
        assert loaded.active_plan_task_id == "plan-1"
        assert loaded.pending_plan_draft == "revise it"

    def test_list_and_delete_removes_only_the_named_task(self, tmp_path: Path) -> None:
        write_agent_task_metadata(tmp_path, _task("task-a"))
        write_agent_task_metadata(tmp_path, _task("task-b"))
        assert {t.task_id for t in list_agent_task_metadata(tmp_path)} == {"task-a", "task-b"}

        assert delete_agent_task(tmp_path, "task-a") is True
        assert read_agent_task_metadata(tmp_path, "task-a") is None
        assert not (tmp_path / AGENT_HOME_NAME / TASKS_SUBDIR / "task-a").exists()
        assert {t.task_id for t in list_agent_task_metadata(tmp_path)} == {"task-b"}

    def test_task_id_rejects_path_escape(self, tmp_path: Path) -> None:
        """Traversal-shaped ids must not resolve outside ``agent/_tasks/``."""
        write_agent_task_metadata(tmp_path, _task("task-ok"))
        victim = tmp_path / "outside"
        victim.mkdir()
        (victim / "keep.txt").write_text("safe")

        assert read_agent_task_metadata(tmp_path, "../outside") is None
        assert delete_agent_task(tmp_path, "../outside") is False
        assert delete_agent_task(tmp_path, "task-ok/../../outside") is False
        assert victim.is_dir()
        assert (victim / "keep.txt").read_text() == "safe"
