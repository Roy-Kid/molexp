"""``RunAssets`` — RunContext's working-dir + asset-I/O facade.

Tier-3 collaborator of :class:`~molexp.workspace.run.RunContext` (see the
``workspace-slim-03-runcontext`` decomposition). Bundles the asset scope,
manifest, the typed accessors (``artifact`` / ``log`` /
``checkpoint`` / ``metrics``), the data-asset import/lookup verbs, the
execution scratch-directory helper, and error-trace persistence — every
"do I/O against this run's assets" entry point in one place.

The producer identity and active execution id are transient lifecycle
state owned by the facade, so they are injected as callables
(``producer`` / ``get_execution_id``) rather than copied.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .assets import (
    ArtifactAccessor,
    AssetManifest,
    AssetScope,
    CheckpointAccessor,
    ErrorTraceAsset,
    ImportAction,
    LogAccessor,
    Producer,
)
from .metrics import MetricsWriter
from .utils import generate_asset_id

if TYPE_CHECKING:
    from .run import Run


class RunAssets:
    """Working-directory + asset-access surface for one run execution."""

    def __init__(
        self,
        run: Run,
        work_dir: Path,
        scope: AssetScope,
        producer: Callable[[], Producer],
        get_execution_id: Callable[[], str | None],
    ) -> None:
        self._run = run
        self._work_dir = work_dir
        self._scope = scope
        self._producer = producer
        self._get_execution_id = get_execution_id
        self._manifest = AssetManifest(work_dir)

        # Only the artifact accessor emits on the event spine (frequency
        # budget: log lines / checkpoints stay silent). The workspace root is
        # resolved through the run's ownership chain — the same path the run
        # lifecycle uses; a detached run (unit-test fixture) simply emits
        # nothing, per the spine's derived/non-fatal contract.
        try:
            event_root = Path(str(run.experiment.project.workspace.root))
        except (RuntimeError, AttributeError):
            # A detached run (bare test fixture) has no ownership chain.
            event_root = None
        self.artifact = ArtifactAccessor(
            work_dir, scope, self._manifest, producer, event_root=event_root
        )
        self.log = LogAccessor(work_dir, scope, self._manifest, producer, get_execution_id)
        self.checkpoint = CheckpointAccessor(work_dir, scope, self._manifest, producer)
        self.metrics = MetricsWriter(work_dir)

    # ── Working directories ─────────────────────────────────────────────

    def folder(self, subpath: str | Path) -> Path:
        """Create and return a working directory under this execution.

        The returned path is ``<run>/executions/<execution_id>/<subpath>``,
        materialized for the caller. Use it for task scratch / intermediate
        files so every execution's working files live under its own
        ``project → experiment → run → execution`` slot instead of a
        hand-rolled global directory: the workspace owns the path layout and
        the directory creation, so callers never assemble paths or ``mkdir``
        themselves.

        Args:
            subpath: Path **relative** to the execution directory, e.g.
                ``"scratch/CAT"`` or ``"output"``. May be nested.

        Returns:
            The created directory as an absolute :class:`~pathlib.Path`.

        Raises:
            RuntimeError: if no execution is active — call this inside
                ``with run.start() as ctx:`` (or ``with run as ctx:``).
            ValueError: if *subpath* is absolute or escapes the execution slot.
        """
        execution_id = self._get_execution_id()
        if execution_id is None:
            raise RuntimeError(
                "RunContext.folder() requires an active execution; call it inside "
                "`with run.start() as ctx:`."
            )
        rel = Path(subpath)
        if rel.is_absolute():
            raise ValueError(f"RunContext.folder: subpath must be relative, got {subpath!r}")
        exec_dir = (self._work_dir / "executions" / execution_id).resolve()
        target = (exec_dir / rel).resolve()
        if not target.is_relative_to(exec_dir):
            raise ValueError(
                f"RunContext.folder: subpath {subpath!r} escapes the execution directory"
            )
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_data_dir(
        self,
        asset_name: str,
        *,
        fallback: str | Path | None = None,
    ) -> Path:
        """Resolve a data directory path.

        Searches the asset hierarchy first. If no asset is found and
        *fallback* is given, creates ``workspace_root / fallback`` and
        returns it.  All return values are :class:`~pathlib.Path`.

        Args:
            asset_name: Name of the asset to look up.
            fallback: Relative path under workspace root to create when the
                asset is not found.

        Returns:
            Resolved data directory path.

        Raises:
            FileNotFoundError: If no asset found and no fallback specified.
        """
        asset = self.find_asset(asset_name)
        if asset is not None:
            return Path(asset.path)
        if fallback is not None:
            fallback = Path(fallback)
            data_dir = Path(self._run.experiment.project.workspace.root) / fallback
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir
        raise FileNotFoundError(f"Asset {asset_name!r} not found and no fallback specified.")

    # ── Asset access ────────────────────────────────────────────────────

    def register_asset(  # noqa: ANN201
        self,
        name: str,
        src: Path | str,
        action: ImportAction = "copy",
        meta: dict | None = None,
    ):
        """Import a ``DataAsset`` into this run's experiment scope."""
        return self._run.experiment.data_assets.import_asset(
            name=name, src=src, action=action, meta=meta or {}
        )

    def get_asset(self, name: str, scope: str = "project"):  # noqa: ANN201
        if scope == "experiment":
            return self._run.experiment.data_assets.get(name)
        if scope == "project":
            return self._run.experiment.project.data_assets.get(name)
        if scope == "workspace":
            return self._run.experiment.project.workspace.data_assets.get(name)
        raise ValueError(f"Unknown scope: {scope!r}")

    def find_asset(self, name: str):  # noqa: ANN201
        for scope in ("experiment", "project", "workspace"):
            asset = self.get_asset(name, scope=scope)
            if asset is not None:
                return asset
        return None

    # ── Run log + error trace ───────────────────────────────────────────

    def append_run_log(self, message: str) -> None:
        """Append a single timestamped line to the ``run`` LogAsset."""
        ts = datetime.now().isoformat(timespec="seconds")
        self.log("run").append(f"{ts}  {message}")

    def save_error_details(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Persist an ``ErrorTraceAsset`` for an exception that propagated."""
        tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)
        self.save_error_report(
            error_type=exc_type.__name__,
            message=str(exc_val),
            traceback_text="".join(tb_lines),
        )

    def save_error_report(
        self,
        *,
        error_type: str,
        message: str,
        traceback_text: str | None = None,
    ) -> None:
        """Persist ``executions/<exec_id>/error.txt`` + its ``ErrorTraceAsset``.

        Shared by both failure paths: an exception that propagated out of the
        ``with run.start():`` block (:meth:`save_error_details` supplies the
        traceback) and the far more common engine-swallowed task failure,
        where the workflow engine resolves the run to FAILED via
        ``mark_failed`` without re-raising — the runtime forwards the
        formatted task traceback through ``mark_failed(..., traceback_text=…)``
        and the lifecycle passes it here, so error.txt carries the real stack.
        The placeholder note below survives only for failure signals that
        genuinely carried no traceback (e.g. a manual ``mark_failed("msg")``).
        """
        exec_id = self._get_execution_id() or "unbound"
        rel_path = Path("executions") / exec_id / "error.txt"
        target = self._work_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        body = f"Error: {datetime.now().isoformat()}\nType: {error_type}\nMessage: {message}\n\n"
        if traceback_text:
            body += traceback_text
        else:
            body += (
                "(No Python traceback was captured: the workflow engine recorded "
                "this task failure without re-raising. Per-task detail lives in "
                f"executions/{exec_id}/workflow.json and logs/.)\n"
            )
        target.write_text(body)

        now = datetime.now()
        asset = ErrorTraceAsset(
            asset_id=generate_asset_id(),
            name=f"error_{exec_id}",
            scope=self._scope,
            path=rel_path,
            created_at=now,
            updated_at=now,
            producer=self._producer(),
            exception_type=error_type,
            message=message,
            execution_id=exec_id,
        )
        self._manifest.register(asset)
