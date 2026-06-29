"""Run entity and RunContext execution lifecycle.

A **Run** represents a single execution instance within an experiment.
**RunContext** is the context manager that handles lifecycle, artifacts,
checkpoints, and asset access during execution.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path  # local-FS path for RunContext (LLM/worker-local I/O)
from typing import TYPE_CHECKING, cast

from mollog import get_logger

from molexp._typing import (
    JSONValue,
    TaskOutput,
)
from molexp.knowledge.types import concept_type
from molexp.path import Path as MolexpPath  # workspace-abstraction path (Folder.path() return)
from molexp.profile import ProfileConfig

from .assets import AssetScope
from .base import (
    _load_metadata,
    _reconstruct,
    _save_metadata,
)
from .errors import RunExistsError, RunNotFoundError
from .folder import WORKSPACE_RUN_KIND, Folder
from .fs import PathArg
from .models import (
    ExecutionRecord,
    FolderMetadata,
    RunMetadata,
    RunStatus,
)
from .run_ops import RUN_OPS_NAME, RunOpsState
from .utils import generate_id

if TYPE_CHECKING:
    from .experiment import Experiment

# Re-exported for backward compatibility — the canonical definition now
# lives in ``.models`` so the run-lifecycle collaborators can import it
# without a circular ``run.py`` dependency.
from .runcontext import RunContext

_logger = get_logger(__name__)

__all__ = ["RETRYABLE_STATUSES", "Run", "RunContext", "RunStatus"]

#: The run statuses ``resume`` / ``rerun`` apply to — the single source of
#: truth for the retryable domain (consumed by both the CLI and the server
#: routes). The three verbs stay orthogonal: ``pending`` is plain run's job,
#: ``succeeded`` is done, and a live ``running`` run must never get a second
#: concurrent execution.
RETRYABLE_STATUSES: frozenset[str] = frozenset({RunStatus.FAILED.value, RunStatus.CANCELLED.value})


# ── Run ─────────────────────────────────────────────────────────────────────


@concept_type(WORKSPACE_RUN_KIND)
class Run(Folder):
    """Single execution instance within an experiment.

    Inherits :class:`Folder` (sub-spec 02): ``kind`` is
    :data:`WORKSPACE_RUN_KIND`, ``parent`` is the owning
    :class:`Experiment`. The on-disk directory uses the ``run-<id>``
    prefix preserved from the pre-refactor layout — see
    :meth:`child_dir`.

    Example::

        run = experiment.add_run(params={"lr": 0.001})
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
        """Current run status, sourced from the ``_ops/run.json`` hot sidecar.

        Hot machine state (status / ownership / heartbeat / executions) lives
        solely in the OKF ``_ops/`` sidecar per the identity-vs-runtime split
        (wsokf-07/wsokf-10); ``run.json`` carries no status field.
        """
        return self.read_ops().status.value

    @property
    def is_retryable(self) -> bool:
        """Whether ``resume`` / ``rerun`` apply (status in :data:`RETRYABLE_STATUSES`)."""
        return self.read_ops().is_retryable

    @property
    def execution_history(self) -> list[ExecutionRecord]:
        """Run-level execution history, read from the ``_ops`` sidecar (wsokf-07)."""
        return list(self.read_ops().executions)

    @property
    def finished_at(self) -> datetime | None:
        """Terminal timestamp, read from the ``_ops`` sidecar (wsokf-07)."""
        return self.read_ops().finished_at

    @property
    def current_execution_id(self) -> str | None:
        """Active/last execution id, read from the ``_ops`` sidecar (wsokf-07)."""
        return self.read_ops().current_execution_id

    # ── OKF _ops/run.json hot-state sidecar (typed; isolated from run.json) ─

    def read_ops(self) -> RunOpsState:
        """Load the typed Run ops state from ``_ops/run.json`` (default if none)."""
        return RunOpsState.model_validate(self.read_ops_json(RUN_OPS_NAME) or {})

    def write_ops(self, state: RunOpsState) -> None:
        """Persist the typed Run ops state to ``_ops/run.json`` (atomic)."""
        self.write_ops_json(RUN_OPS_NAME, state.model_dump(mode="json"))

    def update_ops(self, fn: Callable[[RunOpsState], RunOpsState]) -> RunOpsState:
        """Read-modify-write the typed Run ops state under an advisory lock."""

        def apply(raw: dict[str, JSONValue]) -> dict[str, JSONValue]:
            return fn(RunOpsState.model_validate(raw or {})).model_dump(mode="json")

        return RunOpsState.model_validate(self.update_ops_json(RUN_OPS_NAME, apply))

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
        """Scope-filtered asset view (read-only queries) for this run."""
        from .assets import AssetsView

        return AssetsView(self.experiment.project.workspace.root, self.scope)

    def reregister_artifact(  # noqa: ANN201
        self,
        *,
        name: str | None,
        content_hash: str,
        producer_task: str | None = None,
    ):
        """Re-register a content-addressed artifact into this run's manifest.

        Used by the workflow cache when a cached task output is reused: the
        bytes already live in the content-addressed store, so only a fresh
        manifest entry pointing at the same path + ``content_hash`` is written
        (no recompute, no recopy). Returns the new asset, or ``None`` when the
        bytes are absent in this workspace.
        """
        from .assets import scan

        return scan.reregister_artifact(
            self.experiment.project.workspace.root,
            self.run_dir,
            name=name,
            content_hash=content_hash,
            target_scope=self.scope,
            producer_task=producer_task,
        )

    def get_result(self, key: str) -> TaskOutput:
        """Read a result value for *key*.

        Resolution order:

        1. Driver-side results persisted by ``RunContext.set_result`` into
           ``run.json`` (``context.results``) — always win when the key is
           present, even with a ``None`` value.
        2. Fallback: the completed workflow node named *key* in the run's
           most recent execution's persisted node outputs
           (``executions/<exec_id>/workflow.json``). This keeps results of
           CLI-executed runs (``molexp run``), which never call
           ``set_result``, readable through the same accessor.

        Returns ``None`` when neither source has the key, when the run has
        not been executed yet, or when ``run.json`` does not exist on disk.
        A node output flagged ``outputs_lossy`` (the original value was not
        JSON-serializable, so only a truncated observability rendering was
        persisted) is never returned as a real result — a warning explains
        why and ``None`` is returned.
        """
        from .schema_version import read_versioned_json

        run_json = Path(self.run_dir / "run.json")
        if not run_json.exists() or run_json.stat().st_size == 0:
            return None
        try:
            data = read_versioned_json(run_json)
        except (OSError, ValueError):
            return None
        results = data.get("context", {}).get("results", {})
        if isinstance(results, dict) and key in results:
            return results[key]
        return self._latest_execution_node_output(key)

    def _latest_execution_node_output(self, key: str) -> TaskOutput:
        """Fallback for :meth:`get_result` — read *key* from the latest execution.

        The execution history (newest last) is sourced from the OKF ``_ops``
        sidecar (wsokf-10); its last entry names the most recent attempt.
        Read-only: nothing is written back to disk.
        """
        history = self.read_ops().executions
        if not history:
            return None
        execution_id = history[-1].execution_id
        if not execution_id:
            return None
        from .execution_results import read_completed_node_outputs

        record = read_completed_node_outputs(Path(str(self.run_dir)), execution_id).get(key)
        if record is None:
            return None
        if record.lossy:
            _logger.warning(
                f"run {self.id}: node output {key!r} from execution {execution_id!r} "
                f"is not returned by get_result — the original value was not "
                f"JSON-serializable, so only a lossy (truncated) observability "
                f"rendering was persisted. Make the task return a JSON-safe value, "
                f"or persist it explicitly with ctx.set_result({key!r}, ...) from a "
                f"driver-side run."
            )
            return None
        return record.value

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        d = self.run_dir
        self._fs.mkdir(d, parents=True, exist_ok=True)
        _save_metadata(self.metadata, self._fs.join(self.run_dir, "run.json"), fs=self._fs)

    def save(self) -> None:
        _save_metadata(self.metadata, self._fs.join(self.run_dir, "run.json"), fs=self._fs)

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
        """Mark the run as cancelled in the OKF ``_ops`` hot-state sidecar.

        Sets ``status=cancelled`` + ``finished_at`` and clears the ownership
        stamp (pid/host/heartbeat) in one read-modify-write of ``_ops/run.json``;
        ``run.json`` (identity/provenance) is untouched (wsokf-10).
        """
        now = datetime.now()
        self.update_ops(
            lambda state: state.model_copy(
                update={
                    "status": RunStatus.CANCELLED,
                    "finished_at": now,
                    "owner_pid": None,
                    "owner_host": None,
                    "heartbeat_at": None,
                }
            )
        )

    def delete_execution(self, execution_id: str) -> None:
        """Delete a single execution attempt from this run.

        Removes ``executions/<execution_id>/`` on disk and pops the matching
        entry from ``execution_history``.  The run itself is left intact.

        Raises:
            KeyError: If the execution id is not present under this run.
        """
        import shutil

        exec_dir = Path(self.run_dir / "executions" / execution_id)
        history = list(self.read_ops().executions)
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
            self.update_ops(lambda state: state.model_copy(update={"executions": tuple(history)}))

    # ── Internal (frozen-metadata mutation helpers) ──────────────────────

    def _set_status(self, status: RunStatus) -> None:
        self.update_ops(lambda state: state.model_copy(update={"status": status}))

    @contextlib.contextmanager
    def _metadata_lock(self) -> Iterator[None]:
        """Advisory inter-process lock guarding ``run.json`` read-modify-write.

        Uses a ``run.json.lock`` sidecar next to ``run.json``. Degrades to
        a no-op when the run directory is not a lockable local path (remote
        filesystems, non-POSIX platforms) — see
        :func:`molexp.workspace._file_lock.file_lock`.
        """
        from ._file_lock import file_lock

        with file_lock(Path(str(self.run_dir)) / "run.json.lock"):
            yield

    def _reload_metadata_from_disk(self) -> None:
        """Refresh in-memory metadata from ``run.json`` when it exists.

        Called under :meth:`_metadata_lock` before applying a partial
        update, so concurrent writers (server, CLI, detached workers)
        layering updates onto *distinct* fields don't drop each other's
        writes. Missing or unreadable files keep the in-memory copy
        (first write before ``materialize()``, remote filesystems).
        """
        path = self._fs.join(self.run_dir, "run.json")
        try:
            if not self._fs.exists(path):
                return
            self._entity_metadata = _load_metadata(RunMetadata, path, fs=self._fs)
        except Exception:
            _logger.debug(f"run {self.id}: could not reload run.json; keeping in-memory copy")

    #: Hot machine-state fields that left ``RunMetadata`` in wsokf-10 — they
    #: now live solely in the OKF ``_ops/run.json`` sidecar
    #: (:class:`RunOpsState`). Writers must route them through
    #: :meth:`update_ops`, never :meth:`_update_metadata`.
    _OPS_ONLY_KEYS: frozenset[str] = frozenset(
        {"status", "finished_at", "execution_history", "labels"}
    )

    def _update_metadata(self, **updates: object) -> None:
        """Forward identity/provenance updates into ``RunMetadata.model_copy``.

        Values flow through pydantic's per-field validators; the parameter
        type is the true Python top-type ``object`` (not ``Any`` — the
        function does not interact with the values, it only forwards
        them, and pydantic owns the per-field type contract).

        The read-modify-write cycle (reload from disk → apply updates →
        atomic save) runs under :meth:`_metadata_lock` so concurrent
        processes updating different fields cannot drop each other's
        writes (lost-update protection).

        ``run.json`` holds identity / provenance only (wsokf-10). Hot
        machine state — ``status`` / ``finished_at`` / ``execution_history`` /
        ownership ``labels`` — lives in the ``_ops`` sidecar and must be
        written through :meth:`update_ops`; passing one of those keys here is
        a programming error and raises :class:`ValueError`.

        Raises:
            ValueError: If an ops-only hot-state key is passed.
        """
        offending = self._OPS_ONLY_KEYS & updates.keys()
        if offending:
            raise ValueError(
                f"_update_metadata received hot-state key(s) {sorted(offending)!r}; "
                "these live in the _ops/run.json sidecar — write them via update_ops()"
            )
        with self._metadata_lock():
            self._reload_metadata_from_disk()
            self.metadata = self.metadata.model_copy(update=updates)
            self.save()
