"""Run-scoped store plugin: artifacts, events, lineage, approval."""

from __future__ import annotations

from pathlib import Path

from molexp.harness.host.context import Context
from molexp.harness.host.keys import Keys
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

__all__ = ["RunStoresPlugin"]


class RunStoresPlugin:
    """Publish the four run-local stores onto the host.

    ``workspace_root`` is the Stage-facing root (plan: the run dir; curate:
    the real workspace root). Store files always live under *run_dir*.

    SQLite store objects have no ``close``; connections die with the
    process. ``unload`` drops the service keys only.
    """

    name = "run_stores"
    inject: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        workspace_root: Path | None = None,
    ) -> None:
        self._run_id = run_id
        self._run_dir = Path(run_dir)
        self._workspace_root = Path(workspace_root) if workspace_root is not None else self._run_dir

    def apply(self, ctx: Context) -> None:
        """Open stores under ``run_dir/artifacts`` and ``run_dir/harness.sqlite``."""
        artifact_store = FileArtifactStore(root=self._run_dir / "artifacts")
        db_path = self._run_dir / "harness.sqlite"
        ctx.provide(Keys.RUN_ID, self._run_id)
        ctx.provide(Keys.WORKSPACE_ROOT, self._workspace_root)
        ctx.provide(Keys.ARTIFACTS, artifact_store)
        ctx.provide(Keys.EVENTS, SQLiteEventLog(path=db_path))
        ctx.provide(
            Keys.LINEAGE,
            SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store),
        )
        ctx.provide(Keys.APPROVAL, SQLiteApprovalStore(path=db_path))
