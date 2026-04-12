"""RunStorePersistence: pydantic-graph BaseStatePersistence backed by Run files.

Wraps pydantic-graph's in-memory snapshot model and persists JSON side-cars
atomically to:
    <run_dir>/execution/<execution_id>/<snapshot_id>.json

In-memory snapshots support the required BaseStatePersistence protocol.
On-disk records enable post-run inspection (not used for cross-process resume
in Phase 3 — that is planned for Phase 4).
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

    Args:
        run_dir: Path to the run's directory (run.run_dir or equivalent)
        execution_id: Unique string for this execution (stored as sub-directory)
    """

    def __init__(self, run_dir: Path, execution_id: str) -> None:
        self._exec_dir = run_dir / "execution" / execution_id
        self._exec_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot: NodeSnapshot | EndSnapshot | None = None

    # ── BaseStatePersistence protocol ────────────────────────────────────────

    async def snapshot_node(
        self, state: WorkflowState, next_node: BaseNode
    ) -> None:
        self._last_snapshot = NodeSnapshot(state=state, node=next_node)
        self._write_snapshot_file(self._last_snapshot)

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
        self._write_end_file(state, end)

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        if self._last_snapshot is None or snapshot_id != self._last_snapshot.id:
            raise LookupError(f"No snapshot found with id={snapshot_id!r}")

        assert isinstance(self._last_snapshot, NodeSnapshot), (
            "Only NodeSnapshot can be recorded"
        )
        exceptions.GraphNodeStatusError.check(self._last_snapshot.status)
        self._last_snapshot.status = "running"

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
        """Return all stored snapshots (in-memory only for Phase 3)."""
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

    def _write_snapshot_file(self, snapshot: NodeSnapshot) -> None:
        data = {
            "snapshot_id": snapshot.id,
            "status": snapshot.status,
            "kind": "node",
            "step_outputs": {
                k: _safe_serialize(v)
                for k, v in snapshot.state.step_outputs.items()
            },
            "node_type": type(snapshot.node).__name__,
            "node_data": _safe_serialize(snapshot.node),
        }
        self._write_atomic(snapshot.id, data)

    def _write_end_file(self, state: WorkflowState, end: Any) -> None:
        data = {
            "snapshot_id": "__end__",
            "status": "success",
            "kind": "end",
            "step_outputs": {
                k: _safe_serialize(v) for k, v in state.step_outputs.items()
            },
        }
        self._write_atomic("__end__", data)

    def _update_snapshot_status(self, snapshot_id: str, status: str) -> None:
        path = self._exec_dir / f"{snapshot_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            data["status"] = status
            self._write_atomic(snapshot_id, data)
        except Exception:
            logger.exception(f"Failed to update snapshot status for {snapshot_id}")

    def _write_atomic(self, snapshot_id: str, data: dict) -> None:
        target = self._exec_dir / f"{snapshot_id}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        os.replace(tmp, target)


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
