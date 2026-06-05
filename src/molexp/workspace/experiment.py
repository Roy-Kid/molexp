"""Experiment entity — one directory per parameter combination.

Inherits :class:`Folder` (sub-spec 02) so it participates in the
unified workspace folder abstraction: ``kind`` is
:data:`WORKSPACE_EXPERIMENT_KIND`, ``parent`` is the owning
:class:`Project`.

An Experiment is a parameter-space container plus replica configuration
(``n_replicas`` × ``seeds``). Replicas under the same Experiment share
parameters; they differ only in their random seed.

Workspace does **not** know about workflows — pairing an Experiment
with a workflow is the caller's concern. Use the workflow layer to
build a ``WorkflowSpec`` and pass the workspace ``Run`` to its
``execute(run=...)`` method:

    >>> exp = project.add_experiment("lr-1e-3", params={"lr": 1e-3})
    >>> run = exp.add_run()
    >>> result = await my_workflow_spec.execute(run=run)

Construction is side-effect free; ``project.add_experiment(...)``
materializes on disk at call-time (idempotent: if an experiment with
the same slug already exists, it is loaded and returned).
"""  # noqa: RUF002

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .catalog import AssetCatalog
    from .project import Project
    from .workspace import Workspace

from molexp._typing import JSONValue
from molexp.path import Path

from .assets import AssetScope, AssetsView, DataAssetLibrary
from .base import (
    _load_metadata,
    _reconstruct,
    _save_metadata,
)
from .errors import ExperimentExistsError, ExperimentNotFoundError
from .folder import (
    WORKSPACE_EXPERIMENT_KIND,
    WORKSPACE_RUN_KIND,
    Folder,
)
from .fs import PathArg
from .models import ExperimentMetadata, FolderMetadata
from .run import Run
from .utils import generate_id

# Default replica seeds — deterministic, well-separated
_DEFAULT_SEEDS = [42, 123, 456, 789, 1234]

# Standalone home for a compiled workflow IR document, written alongside
# ``experiment.json``. Kept separate (and free of the ``schema_version``
# envelope) so external tooling — notably the molexp VSCode preview — can read
# and diff the raw IR directly without parsing it out of the metadata file.
WORKFLOW_DOC_FILENAME = "workflow.json"


def _parse_ir_document(source: str | None) -> dict | None:
    """Return the parsed IR object if *source* is a JSON document, else ``None``.

    ``workflow_source`` is free-form: it may carry a compiled workflow IR (a
    JSON object), a path / Python-source string (e.g. ``"train.py"``), or be
    empty. Only the JSON-object form is externalized to ``workflow.json``;
    everything else stays embedded in ``experiment.json``.
    """
    if not source:
        return None
    try:
        doc = json.loads(source)
    except (ValueError, TypeError):
        return None
    return doc if isinstance(doc, dict) else None


class Experiment(Folder):
    """Repeatable experiment — a parameter-space container.

    Example::

        exp = project.add_experiment(
            "lr-1e-3",
            params={"lr": 1e-3},
            n_replicas=3,
        )
        run = exp.add_run()
        # Workflow execution is the caller's concern; workspace just
        # provides the Run that workflow.execute(run=...) operates on.
    """

    _exists_error_cls = ExperimentExistsError
    _not_found_error_cls = ExperimentNotFoundError

    def __init__(
        self,
        *,
        parent: Project | None = None,
        name: str,
        kind: str = WORKSPACE_EXPERIMENT_KIND,
        project: Project | None = None,
        id: str | None = None,
        params: dict[str, JSONValue] | None = None,
        n_replicas: int = 1,
        seeds: list[int] | None = None,
        workflow_source: str | None = None,
        workflow_type: str | None = None,
        git_commit: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        default_target: str | None = None,
        _entity_metadata: ExperimentMetadata | None = None,
    ) -> None:
        resolved_parent = parent if parent is not None else project
        if resolved_parent is None:
            raise ValueError("Experiment: parent (or project) is required")

        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else ExperimentMetadata(
                id=id if id is not None else generate_id(),
                name=name,
                description=description,
                tags=list(tags) if tags is not None else [],
                workflow_source=workflow_source,
                workflow_type=workflow_type,
                parameter_space=dict(params) if params else {},
                git_commit=git_commit,
                n_replicas=n_replicas,
                seeds=list(seeds) if seeds is not None else None,
                default_target=default_target,
            )
        )

        self._parent = resolved_parent
        self._name = meta.id
        self._kind = kind
        self._root_path = None
        self._fs = (
            getattr(resolved_parent, "_fs", None)
            or __import__(
                "molexp.workspace.fs_local", fromlist=["LocalFileSystem"]
            ).LocalFileSystem()
        )
        self._metadata = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=kind,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        self._children_cache = {}

        # Entity-specific state
        self._entity_metadata: ExperimentMetadata = meta
        self._data_assets: DataAssetLibrary | None = None

    # ── Folder hooks ─────────────────────────────────────────────────────

    def resolve(self) -> Path:
        return self.experiment_dir

    @classmethod
    def child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """Folder hook — experiments live under ``experiments/<id>/``."""
        return Path(parent._fs.join(parent.path(), "experiments", derived_id))

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> Experiment:
        """Load ``experiment.json`` and rebuild entity state. See Folder.from_disk hook docs.

        When a standalone ``workflow.json`` is present it is the canonical home
        for the compiled IR; its contents are rehydrated into the in-memory
        ``workflow_source`` field so every downstream reader is unaffected by the
        externalized on-disk layout.
        """
        meta = _load_metadata(
            ExperimentMetadata, parent._fs.join(child_dir, "experiment.json"), fs=parent._fs
        )
        doc_path = parent._fs.join(child_dir, WORKFLOW_DOC_FILENAME)
        if parent._fs.is_file(doc_path):
            with parent._fs.open(doc_path) as fh:
                ir = json.load(fh)
            meta = meta.model_copy(update={"workflow_source": json.dumps(ir, sort_keys=True)})
        folder_meta = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=WORKSPACE_EXPERIMENT_KIND,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        attrs = cls.base_from_disk_attrs(parent, folder_meta) | {
            "_entity_metadata": meta,
            "_data_assets": None,
        }
        return _reconstruct(cls, attrs)

    # ── Properties (entity-specific) ─────────────────────────────────────

    @property
    def project(self) -> Project:
        """The owning :class:`Project` (alias for :attr:`Folder.parent`)."""
        if self._parent is None:  # pragma: no cover — Experiment always has a parent
            raise RuntimeError("Experiment has no parent project")
        return cast("Project", self._parent)

    @property
    def metadata(self) -> ExperimentMetadata:  # type: ignore[override]
        return self._entity_metadata

    @metadata.setter
    def metadata(self, value: ExperimentMetadata) -> None:
        self._entity_metadata = value

    @property
    def id(self) -> str:
        return self._entity_metadata.id

    @property
    def name(self) -> str:
        return self._entity_metadata.name

    @property
    def created_at(self):  # noqa: ANN201
        return self._entity_metadata.created_at

    @property
    def description(self) -> str:
        return self._entity_metadata.description

    @property
    def tags(self) -> list[str]:
        return self._entity_metadata.tags

    @property
    def workflow_source(self) -> str | None:
        return self._entity_metadata.workflow_source

    @property
    def parameter_space(self) -> dict[str, JSONValue]:
        return self._entity_metadata.parameter_space

    @property
    def params(self) -> dict[str, JSONValue]:
        """Concrete parameter dict bound to this experiment."""
        return self._entity_metadata.parameter_space

    @property
    def n_replicas(self) -> int:
        return self._entity_metadata.n_replicas

    @property
    def seeds(self) -> list[int] | None:
        return self._entity_metadata.seeds

    @property
    def workspace(self) -> Workspace:
        return self.project.workspace

    @property
    def experiment_dir(self) -> Path:
        return Path(self._fs.join(self.project.project_dir, "experiments", self.id))

    @property
    def scope(self) -> AssetScope:
        return AssetScope(kind="experiment", ids=(self.project.id, self.id))

    @property
    def assets(self) -> AssetsView:
        """Scope-filtered catalog view (read-only queries)."""
        return AssetsView(self.project.workspace.catalog, self.scope)

    @property
    def data_assets(self) -> DataAssetLibrary:
        if self._data_assets is None:
            self._data_assets = DataAssetLibrary(
                self.experiment_dir, self.scope, self.project.workspace.catalog
            )
        return self._data_assets

    def get_seeds(self) -> list[int]:
        """Return replica seeds (length == ``n_replicas``)."""
        seeds = self._entity_metadata.seeds
        if seeds is not None:
            return list(seeds[: self.n_replicas])
        out = list(_DEFAULT_SEEDS)
        while len(out) < self.n_replicas:
            out.append(out[-1] + 111)
        return out[: self.n_replicas]

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        d = self.experiment_dir
        self._fs.mkdir(d, parents=True, exist_ok=True)
        disk_meta = self._persist_workflow_doc()
        _save_metadata(disk_meta, self._fs.join(d, "experiment.json"), fs=self._fs)
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        disk_meta = self._persist_workflow_doc()
        _save_metadata(
            disk_meta,
            self._fs.join(self.experiment_dir, "experiment.json"),
            fs=self._fs,
        )
        self._catalog_upsert()

    @property
    def _workflow_doc_path(self) -> str:
        """Path of the standalone :data:`WORKFLOW_DOC_FILENAME` IR file."""
        return self._fs.join(self.experiment_dir, WORKFLOW_DOC_FILENAME)

    def _persist_workflow_doc(self) -> ExperimentMetadata:
        """Externalize an IR ``workflow_source`` and return the metadata for disk.

        When the source is a compiled workflow IR, it is written to a standalone
        ``workflow.json`` (clean, pretty-printed — the molexp VSCode preview
        reads it directly) and stripped from the returned metadata so the IR has
        a single on-disk home. Non-IR sources (a Python path / source) stay
        embedded and any stale ``workflow.json`` is removed.

        The in-memory ``self._entity_metadata`` is left untouched so live readers
        (server responses, run snapshots) keep seeing the full source until the
        next reload, where :meth:`from_disk` rehydrates it from the file.
        """
        ir = _parse_ir_document(self._entity_metadata.workflow_source)
        doc_path = self._workflow_doc_path
        if ir is not None:
            self._fs.atomic_write_json(doc_path, ir)
            return self._entity_metadata.model_copy(update={"workflow_source": None})
        if self._fs.is_file(doc_path):
            self._fs.remove(doc_path)
        return self._entity_metadata

    def _write_catalog_row(self, catalog: AssetCatalog) -> None:
        meta = self._entity_metadata
        catalog.upsert_experiment(
            {
                "experiment_id": meta.id,
                "project_id": self.project.id,
                "name": meta.name,
                "description": meta.description,
                "tags": list(meta.tags),
                "parameter_space": dict(meta.parameter_space),
                "n_replicas": meta.n_replicas,
                "workflow_source": meta.workflow_source,
                "workflow_type": meta.workflow_type,
                "path": f"projects/{self.project.id}/experiments/{self.id}",
                "created_at": meta.created_at.isoformat(),
                "updated_at": meta.created_at.isoformat(),
            }
        )

    # ── Run CRUD: typed semantic sugar over generic Folder CRUD ────────────

    def add_run(
        self,
        parameters: dict[str, JSONValue] | None = None,
        *,
        id: str | None = None,
        target: str | None = None,
        workflow_snapshot: dict[str, JSONValue] | None = None,
    ) -> Run:
        """Mount a run under this experiment (idempotent on id).

        One-line wrapper over generic ``add_folder``. Signature matches
        the legacy ``Experiment.Run`` factory: ``parameters`` as first
        positional; an explicit ``id=`` overrides auto-generation.
        """
        resolved_id = id if id is not None else generate_id()
        resolved_target = target if target is not None else self._entity_metadata.default_target
        cached = self._children_cache.get(resolved_id)
        if isinstance(cached, Run):
            return cached
        child_dir = Run.child_dir(self, resolved_id)
        if self._fs.is_dir(child_dir):
            existing = Run.from_disk(child_dir, self)
            self._children_cache[resolved_id] = existing
            return existing
        r = Run(
            parent=self,
            name=resolved_id,
            id=resolved_id,
            parameters=parameters,
            workflow_snapshot=workflow_snapshot,
            target=resolved_target,
        )
        r.materialize()
        self._children_cache[resolved_id] = r
        self._upsert_index_row(r)
        return r

    def get_run(self, run_id: str) -> Run:
        return self.get_folder(run_id, cls=Run)

    def has_run(self, run_id: str) -> bool:
        return self.has_folder(run_id, cls=Run)

    def remove_run(self, run_id: str) -> None:
        self.remove_folder(run_id, cls=Run)
        self.project.workspace.catalog.remove_run(run_id)

    # ── Internal helpers ────────────────────────────────────────────────

    def list_runs(self) -> list[Run]:
        """List all runs by scanning the ``runs/`` directory."""
        result: list[Run] = []
        runs_dir = self._fs.join(self.experiment_dir, "runs")
        if not self._fs.is_dir(runs_dir):
            return result
        for entry_name in sorted(self._fs.listdir(runs_dir)):
            entry_path = self._fs.join(runs_dir, entry_name)
            if self._fs.is_dir(entry_path) and self._fs.exists(
                self._fs.join(entry_path, "run.json")
            ):
                result.append(Run.from_disk(entry_path, self))
        return result

    def children(self, kind: str | None = None) -> list[Folder]:
        """List entity children (currently only :class:`Run`)."""
        if kind is not None and kind != WORKSPACE_RUN_KIND:
            return []
        return list(self.list_runs())
