"""``Run.get_result`` fallback to persisted execution node outputs.

Driver-side results (``ctx.set_result`` → ``run.json`` ``context.results``)
always win. When the key is absent there — the normal situation for
CLI-executed runs (``molexp run`` never calls ``set_result``) —
``Run.get_result`` falls back to the completed node outputs persisted in
the run's most recent execution's ``workflow.json``.

Fidelity rule: a node output flagged ``outputs_lossy`` (the original value
was not JSON-serializable, so only a truncated observability rendering was
persisted) is NEVER returned as a real result — a warning explains why and
``None`` is returned, matching the existing not-found contract.

The last test class writes the execution document through the REAL
workflow-layer writer (``mark_task_status``) so the field-name contract
between the two layers cannot silently drift.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.workspace.models import ExecutionRecord
from molexp.workspace.run import Run

if TYPE_CHECKING:
    from molexp._typing import JSONValue

# ── helpers ─────────────────────────────────────────────────────────────────


def _write_workflow_json(run: Run, execution_id: str, tasks: list[dict[str, JSONValue]]) -> None:
    """Handcraft an execution document in the persisted schema."""
    exec_dir = Path(str(run.run_dir)) / "executions" / execution_id
    exec_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "execution_id": execution_id,
        "status": "completed",
        "task_configs": tasks,
        "links": [],
    }
    (exec_dir / "workflow.json").write_text(json.dumps(document))


def _append_execution(run: Run, execution_id: str) -> None:
    """Register a finished execution attempt in the run's history."""
    now = datetime.now()
    record = ExecutionRecord(
        execution_id=execution_id,
        started_at=now,
        finished_at=now,
        status="succeeded",
    )
    run.update_ops(lambda s: s.model_copy(update={"executions": (*s.executions, record)}))


def _last_execution_id(run: Run) -> str:
    return run.execution_history[-1].execution_id


# ── driver-side precedence ──────────────────────────────────────────────────


class TestDriverResultPrecedence:
    def test_driver_side_result_wins_over_node_output(self, run):
        with run.start() as ctx:
            ctx.set_result("train", "driver-value")
        _write_workflow_json(
            run,
            _last_execution_id(run),
            [{"task_id": "train", "status": "completed", "outputs": "node-value"}],
        )
        assert run.get_result("train") == "driver-value"

    def test_driver_side_none_result_does_not_fall_back(self, run):
        with run.start() as ctx:
            ctx.set_result("train", None)
        _write_workflow_json(
            run,
            _last_execution_id(run),
            [{"task_id": "train", "status": "completed", "outputs": "node-value"}],
        )
        assert run.get_result("train") is None


# ── CLI-style fallback ──────────────────────────────────────────────────────


class TestExecutionNodeOutputFallback:
    def test_cli_style_run_returns_node_output(self, run):
        # CLI runs never call ctx.set_result — only the execution document
        # carries the workflow node outputs.
        with run.start():
            pass
        _write_workflow_json(
            run,
            _last_execution_id(run),
            [{"task_id": "train", "status": "completed", "outputs": {"loss": 0.125}}],
        )
        assert run.get_result("train") == {"loss": 0.125}

    def test_most_recent_execution_wins(self, run):
        _append_execution(run, "exec-old")
        _write_workflow_json(
            run, "exec-old", [{"task_id": "train", "status": "completed", "outputs": "old"}]
        )
        _append_execution(run, "exec-new")
        _write_workflow_json(
            run, "exec-new", [{"task_id": "train", "status": "completed", "outputs": "new"}]
        )
        assert run.get_result("train") == "new"

    def test_non_completed_node_is_not_a_result(self, run):
        _append_execution(run, "exec-a")
        _write_workflow_json(
            run, "exec-a", [{"task_id": "train", "status": "failed", "outputs": "partial"}]
        )
        assert run.get_result("train") is None

    def test_unknown_key_returns_none(self, run):
        _append_execution(run, "exec-a")
        _write_workflow_json(
            run, "exec-a", [{"task_id": "train", "status": "completed", "outputs": 1}]
        )
        assert run.get_result("other") is None

    def test_no_execution_returns_none(self, run):
        # Existing not-found behavior: run.json exists but nothing ran.
        assert run.get_result("train") is None

    def test_no_run_json_returns_none(self, experiment):
        # Never-materialized run keeps the existing contract.
        unmaterialized = Run(parent=experiment, parameters={})
        assert unmaterialized.get_result("train") is None

    def test_execution_without_workflow_json_returns_none(self, run):
        _append_execution(run, "exec-a")
        assert run.get_result("train") is None

    def test_malformed_workflow_json_returns_none(self, run):
        _append_execution(run, "exec-a")
        exec_dir = Path(str(run.run_dir)) / "executions" / "exec-a"
        exec_dir.mkdir(parents=True, exist_ok=True)
        (exec_dir / "workflow.json").write_text("{not json")
        assert run.get_result("train") is None


# ── lossy outputs are never results ─────────────────────────────────────────


class TestLossyOutputs:
    def test_lossy_output_returns_none_with_actionable_warning(self, run):
        _append_execution(run, "exec-a")
        _write_workflow_json(
            run,
            "exec-a",
            [
                {
                    "task_id": "train",
                    "status": "completed",
                    "outputs": "<Model object at 0x…>",
                    "outputs_lossy": True,
                }
            ],
        )
        # mollog dispatches synchronously to attached handlers (stdlib
        # caplog/capfd do not see its sink) — collect the warning directly.
        from mollog import get_logger

        collected: list[str] = []

        class _CollectingHandler:
            def handle(self, record) -> None:
                collected.append(getattr(record, "message", str(record)))

        logger = get_logger("molexp.workspace.run")
        handler = _CollectingHandler()
        logger.add_handler(handler)
        try:
            assert run.get_result("train") is None
        finally:
            logger.remove_handler(handler)
        assert any("JSON-serializable" in message for message in collected), collected
        assert any("set_result" in message for message in collected), collected


# ── drift guard: document written by the real workflow-layer writer ─────────


class TestWorkflowWriterContract:
    """Cross-layer field-name contract (workspace reader ⟷ workflow writer)."""

    @staticmethod
    def _seed_pending_task(run: Run, execution_id: str, name: str) -> Path:
        from molexp.workflow._pydantic_graph.persistence import write_initial_workflow_json

        run_dir = Path(str(run.run_dir))
        write_initial_workflow_json(run_dir, execution_id)
        wf_path = run_dir / "executions" / execution_id / "workflow.json"
        document = json.loads(wf_path.read_text())
        document["task_configs"] = [{"task_id": name, "status": "pending"}]
        wf_path.write_text(json.dumps(document))
        return run_dir

    def test_fallback_reads_output_written_by_mark_task_status(self, run):
        from molexp.workflow._pydantic_graph.persistence import mark_task_status

        with run.start():
            pass
        execution_id = _last_execution_id(run)
        run_dir = self._seed_pending_task(run, execution_id, "train")
        mark_task_status(
            run_dir, execution_id, "train", "completed", output={"loss": 0.5}, snapshot_key="k"
        )
        assert run.get_result("train") == {"loss": 0.5}

    def test_lossy_flag_written_by_mark_task_status_is_respected(self, run):
        from molexp.workflow._pydantic_graph.persistence import mark_task_status

        with run.start():
            pass
        execution_id = _last_execution_id(run)
        run_dir = self._seed_pending_task(run, execution_id, "train")
        mark_task_status(
            run_dir, execution_id, "train", "completed", output=object(), snapshot_key="k"
        )
        assert run.get_result("train") is None
