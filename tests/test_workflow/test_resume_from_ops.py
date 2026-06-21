"""Workflow resume seeding sources execution history from ``_ops`` (wsokf-07).

``workflow/_pydantic_graph/persistence.py``'s ``last_resumable_execution_id`` /
``seed_from_execution`` pick the most-recent non-succeeded execution from
``run.read_ops().executions`` (the OKF hot-state sidecar) instead of
``run.metadata.execution_history``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import molexp as me
from molexp.workflow._pydantic_graph.persistence import (
    last_resumable_execution_id,
    seed_from_execution,
)
from molexp.workspace.run_ops import RunOpsState


def _make_run(tmp_path: Path):
    ws = me.Workspace(tmp_path / "ws")
    return ws.add_project("demo").add_experiment("train").add_run(params={"seed": 0})


def _write_ops_history(run, records: list[dict]) -> None:
    state = RunOpsState.model_validate({"status": "failed", "executions": records})
    run.write_ops(state)


class TestLastResumableFromOps:
    def test_picks_last_non_succeeded(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        _write_ops_history(
            run,
            [
                {
                    "execution_id": "exec-1",
                    "started_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                    "status": "failed",
                },
                {
                    "execution_id": "exec-2",
                    "started_at": datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
                    "status": "succeeded",
                },
                {
                    "execution_id": "exec-3",
                    "started_at": datetime(2026, 1, 3, tzinfo=UTC).isoformat(),
                    "status": "failed",
                },
            ],
        )
        assert last_resumable_execution_id(run) == "exec-3"

    def test_skips_trailing_succeeded(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        _write_ops_history(
            run,
            [
                {
                    "execution_id": "exec-1",
                    "started_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                    "status": "failed",
                },
                {
                    "execution_id": "exec-2",
                    "started_at": datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
                    "status": "cancelled",
                },
            ],
        )
        assert last_resumable_execution_id(run) == "exec-2"

    def test_empty_history_returns_none(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        run.materialize()
        assert last_resumable_execution_id(run) is None


class TestSeedFromOps:
    def test_seeds_completed_outputs_from_ops_execution(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        exec_id = "exec-seed"
        _write_ops_history(
            run,
            [
                {
                    "execution_id": exec_id,
                    "started_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                    "status": "failed",
                }
            ],
        )
        exec_dir = Path(run.run_dir) / "executions" / exec_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        (exec_dir / "workflow.json").write_text(
            json.dumps(
                {
                    "task_configs": [
                        {"task_id": "prep", "status": "completed", "outputs": {"value": 7}}
                    ]
                }
            )
        )

        selected, seeds = seed_from_execution(run)
        assert selected == exec_id
        assert seeds == {"prep": {"value": 7}}

    def test_pending_run_yields_no_seed(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        run.materialize()
        assert seed_from_execution(run) == (None, None)
