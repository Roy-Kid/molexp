"""Run entity and RunContext execution lifecycle.

A **Run** represents a single execution instance within an experiment.
**RunContext** is the context manager that handles lifecycle, artifacts,
checkpoints, and asset access during execution.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path  # local-FS path for RunContext (LLM/worker-local I/O)
from typing import TYPE_CHECKING, cast

from molexp._typing import (
    JSONValue,
    TaskOutput,
)
from molexp.path import Path as MolexpPath  # workspace-abstraction path (Folder.path() return)
from molexp.profile import ProfileConfig

from .assets import (
    AssetCatalog,
    AssetScope,
)
from .base import (
    _load_metadata,
    _reconstruct,
    _save_metadata,
)
from .errors import RunExistsError, RunNotFoundError
from .folder import WORKSPACE_RUN_KIND, Folder
from .fs import PathArg
from .models import (
    FolderMetadata,
    RunMetadata,
    RunStatus,
)
from .utils import generate_id

if TYPE_CHECKING:
    from .experiment import Experiment

# Re-exported for backward compatibility — the canonical definition now
# lives in ``.models`` so the run-lifecycle collaborators can import it
# without a circular ``run.py`` dependency.
from .runcontext import RunContext

__all__ = ["Run", "RunContext", "RunStatus"]


# ── Run ─────────────────────────────────────────────────────────────────────


class Run(Folder):
    """Single execution instance within an experiment.

    Inherits :class:`Folder` (sub-spec 02): ``kind`` is
    :data:`WORKSPACE_RUN_KIND`, ``parent`` is the owning
    :class:`Experiment`. The on-disk directory uses the ``run-<id>``
    prefix preserved from the pre-refactor layout — see
    :meth:`child_dir`.

    Example::

        run = experiment.add_run(parameters={"lr": 0.001})
        with run.start() as ctx:
            result = my_workflow(ctx)
            ctx.set_result("output", result)
    """

    _exists_error_cls = RunExistsError
    _not_found_error_cls = RunNotFoundError

    def __init__(
        self,
        *,
        parent: Experiment | None = None,
        name: str | None = None,
        kind: str = WORKSPACE_RUN_KIND,
        experiment: Experiment | None = None,
        parameters: dict[str, JSONValue] | None = None,
        id: str | None = None,
        workflow_snapshot: dict[str, JSONValue] | None = None,
        target: str | None = None,
        _entity_metadata: RunMetadata | None = None,
    ) -> None:
        resolved_parent = parent if parent is not None else experiment
        if resolved_parent is None:
            raise ValueError("Run: parent (or experiment) is required")
        # ``name`` (Folder convention) is the Run's id — Run has no
        # human-readable name distinct from its slug.
        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else RunMetadata(
                id=id or name or generate_id(),
                parameters=parameters or {},
                workflow_snapshot=workflow_snapshot,
                target=target,
            )
        )

        self._parent = resolved_parent
        self._name = meta.id
        self._kind = kind
        self._root_path = None
        from molexp.workspace.fs_local import LocalFileSystem

        self._fs = getattr(resolved_parent, "_fs", None) or LocalFileSystem()
        self._metadata = FolderMetadata(
            id=meta.id,
            name=meta.id,  # Run has no separate display name
            kind=kind,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        self._children_cache = {}

        # Entity-specific state
        self._entity_metadata: RunMetadata = meta

    # ── Folder hooks ─────────────────────────────────────────────────────

    def resolve(self) -> MolexpPath:
        return self.run_dir

    @classmethod
    def child_dir(cls, parent: Folder, derived_id: str) -> MolexpPath:
        """Folder hook — runs live under ``runs/run-<id>/``."""
        return MolexpPath(parent._fs.join(parent.path(), "runs", f"run-{derived_id}"))

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> Run:
        """Load ``run.json`` and rebuild entity state. See Folder.from_disk hook docs."""
        meta = _load_metadata(RunMetadata, parent._fs.join(child_dir, "run.json"), fs=parent._fs)
        # Runs have no separate human name — ``RunMetadata`` only carries ``id``.
        folder_meta = FolderMetadata(
            id=meta.id,
            name=meta.id,
            kind=WORKSPACE_RUN_KIND,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        attrs = cls.base_from_disk_attrs(parent, folder_meta) | {
            "_entity_metadata": meta,
        }
        return _reconstruct(cls, attrs)

    def children(self, kind: str | None = None) -> list[Folder]:  # noqa: ARG002
        """Run has no entity children — executions live under ``executions/``
        but are not Folder-tracked (sub-spec 03 may revisit)."""
        return []

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def experiment(self) -> Experiment:
        """The owning :class:`Experiment` (alias for :attr:`Folder.parent`)."""
        if self._parent is None:  # pragma: no cover — Run always has a parent
            raise RuntimeError("Run has no parent experiment")
        return cast("Experiment", self._parent)

    @property
    def metadata(self) -> RunMetadata:  # type: ignore[override]
        return self._entity_metadata

    @metadata.setter
    def metadata(self, value: RunMetadata) -> None:
        self._entity_metadata = value

    @property
    def id(self) -> str:
        return self._entity_metadata.id

    @property
    def parameters(self) -> dict[str, JSONValue]:
        return self._entity_metadata.parameters

    @property
    def status(self) -> str:
        return self.metadata.status

    @property
    def run_dir(self) -> MolexpPath:
        return MolexpPath(self._fs.join(self.experiment.experiment_dir, "runs", f"run-{self.id}"))

    @property
    def scope(self):  # noqa: ANN201

        return AssetScope(
            kind="run",
            ids=(self.experiment.project.id, self.experiment.id, self.id),
        )

    @property
    def assets(self):  # noqa: ANN201
        """Scope-filtered catalog view (read-only queries) for this run."""
        from .assets import AssetsView

        return AssetsView(self.experiment.project.workspace.catalog, self.scope)

    def get_result(self, key: str) -> TaskOutput:
        """Read a result value persisted by ``RunContext.set_result``.

        Returns ``None`` when the run has not been executed yet, when the
        key is absent, or when ``run.json`` does not exist on disk.
        """
        from .schema_version import read_versioned_json

        run_json = Path(self.run_dir / "run.json")
        if not run_json.exists() or run_json.stat().st_size == 0:
            return None
        try:
            data = read_versioned_json(run_json)
        except (OSError, ValueError):
            return None
        return data.get("context", {}).get("results", {}).get(key)

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        d = self.run_dir
        self._fs.mkdir(d, parents=True, exist_ok=True)
        _save_metadata(self.metadata, self._fs.join(self.run_dir, "run.json"), fs=self._fs)
        self._catalog_upsert()

    def save(self) -> None:
        _save_metadata(self.metadata, self._fs.join(self.run_dir, "run.json"), fs=self._fs)
        self._catalog_upsert()

    def _write_catalog_row(self, catalog: AssetCatalog) -> None:
        record = {
            "run_id": self.metadata.id,
            "experiment_id": self.experiment.id,
            "status": self.metadata.status,
            "parameters": dict(self.metadata.parameters),
            "profile": self.metadata.profile,
            "config_hash": self.metadata.config_hash,
            "labels": dict(self.metadata.labels),
            "path": f"runs/run-{self.id}",
            "created_at": self.metadata.created_at.isoformat(),
            "finished_at": (
                self.metadata.finished_at.isoformat() if self.metadata.finished_at else None
            ),
            "workflow_snapshot": (
                dict(self.metadata.workflow_snapshot) if self.metadata.workflow_snapshot else None
            ),
        }
        # Batch the run row + its executions into one transaction (was N+1
        # whole-file rewrites under the legacy backend).
        execution_records = [
            {
                "execution_id": rec.execution_id,
                "run_id": self.metadata.id,
                "status": rec.status,
                "started_at": rec.started_at.isoformat(),
                "finished_at": (rec.finished_at.isoformat() if rec.finished_at else None),
                "scheduler_job_id": rec.scheduler_job_id,
            }
            for rec in self.metadata.execution_history
        ]
        catalog.upsert_run_with_executions(record, execution_records)

    # ── Execution ───────────────────────────────────────────────────────

    def start(
        self,
        profile_config: ProfileConfig | None = None,
        *,
        execution_id: str | None = None,
    ) -> RunContext:
        """Return a context manager for executing this run.

        *profile_config* selects the active molcfg profile; when omitted
        the run executes with an empty (defaults-only) :class:`ProfileConfig`.

        *execution_id* pre-allocates the execution slot — used by external
        submitters (e.g. molq) that need to know the per-attempt directory
        ahead of worker startup.

        The returned :class:`RunContext` supports both ``with`` and
        ``async with`` — choose whichever matches the caller's body.
        For the no-arg case, ``Run`` itself is also a context manager
        (sugar that calls ``self.start()`` internally); see
        :meth:`__enter__` / :meth:`__aenter__`.
        """
        return RunContext(self, profile_config=profile_config, execution_id=execution_id)

    # ── Sugar: ``with run as ctx:`` / ``async with run as ctx:`` ────────
    #
    # Equivalent to ``with run.start() as ctx:`` / ``async with``. Sugar
    # form does not accept ``profile_config`` / ``execution_id``; for
    # those, call ``run.start(...)`` explicitly. Internally we cache the
    # ``RunContext`` on first ``__enter__`` so ``__exit__`` sees the
    # same instance.

    def __enter__(self) -> RunContext:
        self._sugar_ctx = self.start()
        return self._sugar_ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        ctx = self._sugar_ctx
        del self._sugar_ctx
        return ctx.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> RunContext:
        self._sugar_ctx = self.start()
        return await self._sugar_ctx.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        ctx = self._sugar_ctx
        del self._sugar_ctx
        return await ctx.__aexit__(exc_type, exc_val, exc_tb)

    def cancel(self) -> None:
        """Mark the run as cancelled in workspace metadata."""
        labels = dict(self.metadata.labels)
        for key in ("pid", "host", "heartbeat"):
            labels.pop(key, None)
        self._update_metadata(
            status=RunStatus.CANCELLED,
            finished_at=datetime.now(),
            labels=labels,
        )

    def delete_execution(self, execution_id: str) -> None:
        """Delete a single execution attempt from this run.

        Removes ``executions/<execution_id>/`` on disk, pops the matching
        entry from ``execution_history``, and drops the catalog row.  The
        run itself is left intact.

        Raises:
            KeyError: If the execution id is not present under this run.
        """
        import shutil

        exec_dir = Path(self.run_dir / "executions" / execution_id)
        history = list(self.metadata.execution_history)
        matched_idx = next(
            (i for i, rec in enumerate(history) if rec.execution_id == execution_id),
            None,
        )
        if matched_idx is None and not exec_dir.exists():
            raise KeyError(f"Execution '{execution_id}' not found under run '{self.id}'")
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
        if matched_idx is not None:
            history.pop(matched_idx)
            self._update_metadata(execution_history=history)
        self.experiment.project.workspace.catalog.remove_execution(execution_id)

    # ── Internal (frozen-metadata mutation helpers) ──────────────────────

    def _set_status(self, status: RunStatus) -> None:
        self.metadata = self.metadata.model_copy(update={"status": status.value})
        self.save()

    def _update_metadata(self, **updates: object) -> None:
        """Forward partial-field updates into ``RunMetadata.model_copy``.

        Values flow through pydantic's per-field validators; the parameter
        type is the true Python top-type ``object`` (not ``Any`` — the
        function does not interact with the values, it only forwards
        them, and pydantic owns the per-field type contract).
        """
        self.metadata = self.metadata.model_copy(update=updates)
        self.save()
