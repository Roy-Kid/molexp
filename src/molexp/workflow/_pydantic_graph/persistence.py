"""RunStorePersistence: pydantic-graph BaseStatePersistence backed by a single workflow.json.

All workflow execution state (steps + end) is consolidated into one
file::

    <run_dir>/executions/<execution_id>/workflow.json

This replaces the previous per-snapshot file layout
(``WorkflowStep:{uuid}.json``, ``__end__.json``), making it easy to
inspect progress with a single ``cat workflow.json``.

Atomic writes route through workspace's
:func:`molexp.workspace.atomic_write_json` so the atomicity guarantee
is workspace's, not a workflow-layer reinvention. (Rectification spec
2026-05-09 — workspace ← workflow direction.)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import cast

from mollog import get_logger
from pydantic_graph import BaseNode, End
from pydantic_graph.exceptions import GraphNodeStatusError
from pydantic_graph.persistence import BaseStatePersistence, EndSnapshot, NodeSnapshot

from molexp.workspace import atomic_write_json

from ..._typing import HashablePayload, JSONValue
from ..protocols import UserDeps
from .state import WorkflowState

logger = get_logger(__name__)


@dataclass
class _StepRecord:
    """One scheduler-frame snapshot persisted to ``workflow.json``."""

    snapshot_id: str
    index: int
    status: str
    outputs: dict[str, JSONValue]


@dataclass
class _EndRecord:
    """Workflow-end snapshot — final outputs only."""

    outputs: dict[str, JSONValue]


@dataclass
class _PersistenceState:
    """In-memory mirror of the on-disk ``workflow.json`` document."""

    execution_id: str
    status: str = "running"
    steps: list[_StepRecord] = field(default_factory=list)
    end: _EndRecord | None = None

    def to_jsonable(self) -> dict[str, JSONValue]:
        """Serialize for ``json.dumps`` — strips internal ``snapshot_id``."""
        steps_json: list[JSONValue] = [
            {
                "index": step.index,
                "status": step.status,
                "outputs": dict(step.outputs),
            }
            for step in self.steps
        ]
        return {
            "execution_id": self.execution_id,
            "status": self.status,
            "steps": steps_json,
            "end": ({"outputs": dict(self.end.outputs)} if self.end is not None else None),
        }


class RunStorePersistence(BaseStatePersistence[WorkflowState, WorkflowState]):
    """Persist workflow graph snapshots inside a molexp Run directory.

    All snapshots are written atomically to a single ``workflow.json``:

    .. code-block:: json

        {
          "execution_id": "exec-abc12345",
          "status": "running",
          "steps": [
            {"index": 1, "status": "success", "outputs": {...}},
            {"index": 2, "status": "running", "outputs": {...}}
          ],
          "end": null
        }

    Spec 04 §7 — the ``outputs`` field name is the canonical one and
    matches :attr:`WorkflowResult.outputs`. Per CLAUDE.md, per-execution
    artifacts are not subject to the long-term workspace BC promise; old
    per-attempt ``workflow.json`` files written by earlier versions are
    not parsed.

    Args:
        run_dir: Path to the run's directory (``run.run_dir`` or equivalent).
        execution_id: Unique string for this execution (stored as sub-directory).
    """

    def __init__(self, run_dir: Path, execution_id: str) -> None:
        self._run_dir = run_dir
        self._execution_id = execution_id
        self._exec_dir = run_dir / "executions" / execution_id
        self._exec_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot: (
            NodeSnapshot[WorkflowState, WorkflowState]
            | EndSnapshot[WorkflowState, WorkflowState]
            | None
        ) = None
        self._workflow_file = self._exec_dir / "workflow.json"
        # In-memory state; written atomically on every mutation.
        self._state = _PersistenceState(execution_id=execution_id)

        self._write_workflow()

    # ── BaseStatePersistence protocol ────────────────────────────────────────

    async def snapshot_node(
        self,
        state: WorkflowState,
        next_node: BaseNode[WorkflowState, UserDeps, WorkflowState],
    ) -> None:
        self._last_snapshot = NodeSnapshot(state=state, node=next_node)
        # One per-task Step frame per snapshot. The runtime no longer injects
        # this persistence backend into the graph runner (only the initial
        # workflow.json write happens), so this path is exercised only if a
        # caller wires it up explicitly; the index is the running frame count.
        self._state.steps.append(
            _StepRecord(
                snapshot_id=self._last_snapshot.id,
                index=len(self._state.steps) + 1,  # 1-indexed for human display
                status="pending",
                outputs={k: _safe_serialize(v) for k, v in state.results.items()},
            )
        )
        self._write_workflow()

    async def snapshot_node_if_new(
        self,
        snapshot_id: str,
        state: WorkflowState,
        next_node: BaseNode[WorkflowState, UserDeps, WorkflowState],
    ) -> None:
        if self._last_snapshot and self._last_snapshot.id == snapshot_id:
            return
        await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: WorkflowState, end: End[WorkflowState]) -> None:
        self._last_snapshot = EndSnapshot(state=state, result=end)
        self._state.status = "completed"
        self._state.end = _EndRecord(
            outputs={k: _safe_serialize(v) for k, v in state.results.items()},
        )
        self._write_workflow()

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        if self._last_snapshot is None or snapshot_id != self._last_snapshot.id:
            raise LookupError(f"No snapshot found with id={snapshot_id!r}")

        assert isinstance(self._last_snapshot, NodeSnapshot), "Only NodeSnapshot can be recorded"
        GraphNodeStatusError.check(self._last_snapshot.status)
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
        snapshot = self._last_snapshot
        if isinstance(snapshot, NodeSnapshot) and snapshot.status == "created":
            snapshot.status = "pending"
            return cast(NodeSnapshot[WorkflowState, WorkflowState], snapshot)
        return None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _update_snapshot_status(self, snapshot_id: str, status: str) -> None:
        for step in self._state.steps:
            if step.snapshot_id == snapshot_id:
                step.status = status
                self._write_workflow()
                return

    def _write_workflow(self) -> None:
        """Atomically write workflow.json through workspace's helper."""
        atomic_write_json(self._workflow_file, self._state.to_jsonable())


def _safe_serialize(obj: HashablePayload) -> JSONValue:
    """Best-effort serialization for snapshot data."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return {k: _safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        return str(obj)
