"""RunStorePersistence: pydantic-graph BaseStatePersistence backed by a single workflow.json.

All workflow execution state (steps + end) is consolidated into one file:
    <run_dir>/executions/<execution_id>/workflow.json

This replaces the previous per-snapshot file layout
(``WorkflowStep:{uuid}.json``, ``__end__.json``), making it easy to inspect
progress with a single ``cat workflow.json``.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator, cast

from mollog import get_logger
from pydantic_graph import End, exceptions
from pydantic_graph.nodes import BaseNode
from pydantic_graph.persistence import BaseStatePersistence, EndSnapshot, NodeSnapshot

from molexp.workspace.assets import (
    AssetCatalog,
    AssetManifest,
    AssetScope,
    ExecutionStateAsset,
    Producer,
)
from molexp.workspace.utils import generate_asset_id

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
        self._state: dict[str, Any] = {
            "execution_id": execution_id,
            "status": "running",
            "steps": [],
            "end": None,
        }

        # Best-effort wiring to manifest + catalog.  If the run directory
        # does not sit under a recognizable workspace layout, skip silently.
        self._state_asset_id: str | None = None
        self._manifest: AssetManifest | None = None
        self._catalog: AssetCatalog | None = None
        self._scope: AssetScope | None = None
        self._try_attach_asset_plumbing()

        self._write_workflow()

    # ── BaseStatePersistence protocol ────────────────────────────────────────

    async def snapshot_node(
        self,
        state: WorkflowState,
        next_node: BaseNode[WorkflowState, Any, WorkflowState],
    ) -> None:
        self._last_snapshot = NodeSnapshot(state=state, node=next_node)
        if type(next_node).__name__ == "WorkflowStep":
            level_index = getattr(next_node, "level_index", len(self._state["steps"]))
            self._state["steps"].append(
                {
                    "_snapshot_id": self._last_snapshot.id,
                    "index": level_index + 1,  # 1-indexed for human display
                    "status": "pending",
                    "step_outputs": {k: _safe_serialize(v) for k, v in state.step_outputs.items()},
                }
            )
            self._write_workflow()

    async def snapshot_node_if_new(
        self,
        snapshot_id: str,
        state: WorkflowState,
        next_node: BaseNode[WorkflowState, Any, WorkflowState],
    ) -> None:
        if self._last_snapshot and self._last_snapshot.id == snapshot_id:
            return
        await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: WorkflowState, end: End[WorkflowState]) -> None:
        self._last_snapshot = EndSnapshot(state=state, result=end)
        self._state["status"] = "completed"
        self._state["end"] = {
            "step_outputs": {k: _safe_serialize(v) for k, v in state.step_outputs.items()},
        }
        self._write_workflow()

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        if self._last_snapshot is None or snapshot_id != self._last_snapshot.id:
            raise LookupError(f"No snapshot found with id={snapshot_id!r}")

        assert isinstance(self._last_snapshot, NodeSnapshot), "Only NodeSnapshot can be recorded"
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
        snapshot = self._last_snapshot
        if isinstance(snapshot, NodeSnapshot) and snapshot.status == "created":
            snapshot.status = "pending"
            return cast(NodeSnapshot[WorkflowState, WorkflowState], snapshot)
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
        self._register_execution_state()

    def _try_attach_asset_plumbing(self) -> None:
        """Detect the workspace layout and bind manifest+catalog if possible.

        Expected layout:
            ``<workspace_root>/projects/<p>/experiments/<e>/runs/<r>``
        """
        parents = self._run_dir.resolve().parents
        if len(parents) < 6:
            return
        exp_dir = parents[1]
        proj_dir = parents[3]
        workspace_root = parents[5]
        if not (workspace_root / "workspace.json").exists():
            return
        self._manifest = AssetManifest(self._run_dir)
        self._catalog = AssetCatalog(workspace_root)
        self._scope = AssetScope(
            kind="run",
            ids=(proj_dir.name, exp_dir.name, self._run_dir.name),
        )

    def _register_execution_state(self) -> None:
        """Register or update an ExecutionStateAsset for this workflow.json."""
        if self._manifest is None or self._catalog is None or self._scope is None:
            return
        now = datetime.now()
        rel_path = Path("executions") / self._execution_id / "workflow.json"
        if self._state_asset_id is None:
            self._state_asset_id = generate_asset_id()
            asset = ExecutionStateAsset(
                asset_id=self._state_asset_id,
                name=f"workflow_{self._execution_id}",
                scope=self._scope,
                path=rel_path,
                created_at=now,
                updated_at=now,
                producer=Producer(
                    run_id=self._scope.ids[-1].removeprefix("run-"),
                    execution_id=self._execution_id,
                ),
                execution_id=self._execution_id,
                status=self._state.get("status", "running"),
            )
            self._manifest.register(asset)
            self._catalog.register(asset)
        else:
            existing = self._manifest.get(self._state_asset_id)
            if not isinstance(existing, ExecutionStateAsset):
                return
            updated = existing.model_copy(
                update={
                    "updated_at": now,
                    "status": self._state.get("status", existing.status),
                }
            )
            self._manifest.update(updated)
            self._catalog.update(updated)


def _safe_serialize(obj: Any) -> Any:
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
