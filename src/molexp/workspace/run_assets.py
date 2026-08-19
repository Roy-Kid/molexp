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
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .assets import (
    ArtifactAsset,
    Asset,
    AssetManifest,
    AssetScope,
    CheckpointAccessor,
    CheckpointAsset,
    ErrorTraceAsset,
    LogAccessor,
    LogAsset,
    Producer,
)
from .assets.base import AssetKind
from .file_store import FileStore
from .metrics import MetricsWriter
from .utils import compute_content_hash, generate_asset_id

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
        self._run_dir = work_dir
        self._scope = scope
        self._producer = producer
        self._get_execution_id = get_execution_id
        self._manifest = AssetManifest(work_dir)
        self.files = FileStore(work_dir, fs=run._disk())

        # Only artifact registration emits on the event spine (frequency
        # budget: log lines / checkpoints stay silent). The workspace root is
        # resolved through the run's ownership chain — the same path the run
        # lifecycle uses; a detached run (unit-test fixture) simply emits
        # nothing, per the spine's derived/non-fatal contract.
        try:
            self._event_root = Path(str(run.experiment.project.workspace.root))
        except (RuntimeError, AttributeError):
            self._event_root = None
        self.log = LogAccessor(
            work_dir,
            scope,
            self._manifest,
            producer,
            get_execution_id,
            files=self.files,
        )
        self.checkpoint = CheckpointAccessor(
            work_dir, scope, self._manifest, producer, files=self.files
        )
        self.metrics = MetricsWriter(work_dir, append=self.files.append)

    # ── Working directories ─────────────────────────────────────────────

    @property
    def workdir(self) -> Path:
        """Execution-scoped scratch directory ``<run>/executions/<id>/work``.

        Created on access. Requires an active execution
        (``with run.start() as ctx:``).
        """
        execution_id = self._get_execution_id()
        if execution_id is None:
            raise RuntimeError(
                "RunContext.workdir requires an active execution; call it inside "
                "`with run.start() as ctx:`."
            )
        return self.files.mkdir(Path("executions") / execution_id / "work")

    def task_workdir(self, task_name: str) -> Path:
        """Scratch directory for one task: ``executions/<id>/work/<task>/``."""
        execution_id = self._get_execution_id()
        if execution_id is None:
            raise RuntimeError(
                "RunContext.task_workdir requires an active execution; "
                "call it inside `with run.start() as ctx:`."
            )
        return self.files.mkdir(Path("executions") / execution_id / "work" / task_name)

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

    # ── Catalog ─────────────────────────────────────────────────────────

    def register(
        self,
        path: Path,
        *,
        kind: AssetKind = "artifact",
        name: str | None = None,
        mime: str | None = None,
        tags: dict[str, str] | None = None,
        consumed: Sequence[Asset | str] | None = None,
        **extra: object,
    ) -> Asset:
        """Catalog an existing file. Does not copy or rewrite the payload."""
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"register: not a file: {path}")
        root = self._run_dir.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"register: path {path} is not under run_dir {root}")
        rel = resolved.relative_to(root)
        now = datetime.now()
        producer = self._producer()
        if consumed:
            ids = tuple(item if isinstance(item, str) else item.asset_id for item in consumed)
            producer = producer.model_copy(update={"inputs": ids})
        label = name or resolved.name
        asset: Asset
        if kind == "artifact":
            asset = ArtifactAsset(
                asset_id=generate_asset_id(),
                name=label,
                scope=self._scope,
                path=rel,
                created_at=now,
                updated_at=now,
                producer=producer,
                tags=tags or {},
                mime=mime,
                size=resolved.stat().st_size,
                content_hash=compute_content_hash(resolved),
            )
        elif kind == "log":
            asset = LogAsset(
                asset_id=generate_asset_id(),
                name=label,
                scope=self._scope,
                path=rel,
                created_at=now,
                updated_at=now,
                producer=producer,
                tags=tags or {},
            )
        elif kind == "checkpoint":
            ckpt_id = str(extra.get("ckpt_id") or label)
            parent = extra.get("parent_ckpt_id")
            asset = CheckpointAsset(
                asset_id=generate_asset_id(),
                name=label,
                scope=self._scope,
                path=rel,
                created_at=now,
                updated_at=now,
                producer=producer,
                tags=tags or {},
                ckpt_id=ckpt_id,
                parent_ckpt_id=str(parent) if parent is not None else None,
            )
        elif kind == "error_trace":
            asset = ErrorTraceAsset(
                asset_id=generate_asset_id(),
                name=label,
                scope=self._scope,
                path=rel,
                created_at=now,
                updated_at=now,
                producer=producer,
                tags=tags or {},
                exception_type=str(extra.get("exception_type") or "Error"),
                message=str(extra.get("message") or ""),
                execution_id=str(extra.get("execution_id") or "unbound"),
            )
        elif kind == "data":
            raise ValueError("register: kind='data' uses data_assets.import_asset")
        else:
            raise ValueError(f"register: unknown kind {kind!r}")
        self._manifest.register(asset)
        if kind == "artifact" and self._event_root is not None:
            from .assets._events import emit_asset_added

            emit_asset_added(
                self._event_root,
                asset,
                name=label,
                extra_refs=[producer.run_id] if producer.run_id else (),
            )
        return asset

    def register_artifact(
        self,
        data: object,
        *,
        name: str | None = None,
        mime: str | None = None,
        tags: dict[str, str] | None = None,
        consumed: Sequence[Asset | str] | None = None,
    ) -> ArtifactAsset:
        if name is None:
            if isinstance(data, Path):
                name = Path(data).name
            else:
                raise ValueError("register_artifact: name is required when data is not a Path")
        payload: Path | bytes | dict | list | str
        if isinstance(data, (Path, bytes, dict, list, str)):
            payload = data
        elif isinstance(data, bytearray):
            payload = bytes(data)
        else:
            payload = str(data)
        dest = self.files.put(Path("artifacts") / name, payload)
        asset = self.register(
            dest, kind="artifact", name=name, mime=mime, tags=tags, consumed=consumed
        )
        assert isinstance(asset, ArtifactAsset)
        return asset

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
        body = f"Error: {datetime.now().isoformat()}\nType: {error_type}\nMessage: {message}\n\n"
        if traceback_text:
            body += traceback_text
        else:
            body += (
                "(No Python traceback was captured: the workflow engine recorded "
                "this task failure without re-raising. Per-task detail lives in "
                f"executions/{exec_id}/workflow.json and logs/.)\n"
            )
        target = self.files.put(rel_path, body)
        self.register(
            target,
            kind="error_trace",
            name=f"error_{exec_id}",
            exception_type=error_type,
            message=message,
            execution_id=exec_id,
        )
