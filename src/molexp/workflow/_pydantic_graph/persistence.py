"""RunStorePersistence: pydantic-graph BaseStatePersistence backed by a single workflow.json.

All workflow execution state (steps + end) is consolidated into one file:
    <run_dir>/execution/<execution_id>/workflow.json

This replaces the previous per-snapshot file layout
(``WorkflowStep:{uuid}.json``, ``__end__.json``), making it easy to inspect
progress with a single ``cat workflow.json``.
"""

from __future__ import annotations

import json
from mollog import get_logger
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

from pydantic_graph import End, exceptions
from pydantic_graph.nodes import BaseNode
from pydantic_graph.persistence import BaseStatePersistence, EndSnapshot, NodeSnapshot

from .state import WorkflowState

logger = get_logger(__name__)


class RunStorePersistence(BaseStatePersistence[WorkflowState, WorkflowState]):
    """Persist workflow graph snapshots inside a molexp Run directory.

    All snapshots are written atomically to a single ``workflow.json``:

    .. code-block:: json

        {
          "execution_id": "exec-abc12345",
          "status": "running",
          "steps": [
            {"index": 1, "status": "success", "step_outputs": {...}},
            {"index": 2, "status": "running", "step_outputs": {...}}
          ],
          "end": null
        }

    Args:
        run_dir: Path to the run's directory (``run.run_dir`` or equivalent).
        execution_id: Unique string for this execution (stored as sub-directory).
    """

    def __init__(self, run_dir: Path, execution_id: str) -> None:
        self._exec_dir = run_dir / "execution" / execution_id
        self._exec_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot: NodeSnapshot | EndSnapshot | None = None
        self._workflow_file = self._exec_dir / "workflow.json"
        # In-memory state; written atomically on every mutation.
        self._state: dict[str, Any] = {
            "execution_id": execution_id,
            "status": "running",
            "steps": [],
            "end": None,
        }
        self._write_workflow()

    # ── BaseStatePersistence protocol ────────────────────────────────────────

    async def snapshot_node(
        self, state: WorkflowState, next_node: BaseNode
    ) -> None:
        self._last_snapshot = NodeSnapshot(state=state, node=next_node)
        if type(next_node).__name__ == "WorkflowStep":
            level_index = getattr(next_node, "level_index", len(self._state["steps"]))
            self._state["steps"].append({
                "_snapshot_id": self._last_snapshot.id,
                "index": level_index + 1,  # 1-indexed for human display
                "status": "pending",
                "step_outputs": {
                    k: _safe_serialize(v)
                    for k, v in state.step_outputs.items()
                },
            })
            self._write_workflow()

    async def snapshot_node_if_new(
        self,
        snapshot_id: str,
        state: WorkflowState,
        next_node: BaseNode,
    ) -> None:
        if self._last_snapshot and self._last_snapshot.id == snapshot_id:
            return
        await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: WorkflowState, end: End[WorkflowState]) -> None:
        self._last_snapshot = EndSnapshot(state=state, result=end)
        self._state["status"] = "completed"
        self._state["end"] = {
            "step_outputs": {
                k: _safe_serialize(v)
                for k, v in state.step_outputs.items()
            },
        }
        self._write_workflow()

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        if self._last_snapshot is None or snapshot_id != self._last_snapshot.id:
            raise LookupError(f"No snapshot found with id={snapshot_id!r}")

        assert isinstance(self._last_snapshot, NodeSnapshot), (
            "Only NodeSnapshot can be recorded"
        )
        exceptions.GraphNodeStatusError.check(self._last_snapshot.status)
        self._last_snapshot.status = "running"
        self._update_snapshot_status(snapshot_id, "running")

        start = perf_counter()
        try:
            yield
        except Exception:
            self._last_snapshot.duration = perf_counter() - start
            self._last_snapshot.status = "error"
            self._update_snapshot_status(snapshot_id, "error")
            raise
        else:
            self._last_snapshot.duration = perf_counter() - start
            self._last_snapshot.status = "success"
            self._update_snapshot_status(snapshot_id, "success")

    async def load_all(self) -> list:
        """Return all stored snapshots (in-memory only)."""
        if self._last_snapshot is not None:
            return [self._last_snapshot]
        return []

    async def load_next(self) -> NodeSnapshot[WorkflowState, WorkflowState] | None:
        if (
            isinstance(self._last_snapshot, NodeSnapshot)
            and self._last_snapshot.status == "created"
        ):
            self._last_snapshot.status = "pending"
            return self._last_snapshot
        return None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _update_snapshot_status(self, snapshot_id: str, status: str) -> None:
        for step in self._state["steps"]:
            if step.get("_snapshot_id") == snapshot_id:
                step["status"] = status
                self._write_workflow()
                return

    def _write_workflow(self) -> None:
        """Atomically write workflow.json (tmp → rename)."""
        clean = {
            "execution_id": self._state["execution_id"],
            "status": self._state["status"],
            "steps": [
                {k: v for k, v in step.items() if k != "_snapshot_id"}
                for step in self._state["steps"]
            ],
            "end": self._state["end"],
        }
        tmp = self._workflow_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(clean, indent=2, default=str))
        os.replace(tmp, self._workflow_file)


def _safe_serialize(obj: Any) -> Any:
    """Best-effort serialization for snapshot data."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return {
                k: _safe_serialize(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        return str(obj)
