"""Project entity with experiment management.

Inherits :class:`Folder` (sub-spec 02) so it participates in the
unified workspace folder abstraction: ``kind`` is
:data:`WORKSPACE_PROJECT_KIND`, ``parent`` is the owning
:class:`Workspace`. Construction is side-effect free;
``workspace.add_project(...)`` materializes on disk at call-time
(idempotent: existing projects are loaded, missing ones are created).
"""

from __future__ import annotations

from pathlib import Path as _LocalPath
from typing import TYPE_CHECKING, Any, cast

from molexp._typing import JSONValue
from molexp.path import Path

if TYPE_CHECKING:
    from .fs import FileSystem
    from .workspace import Workspace

from molexp.knowledge.types import concept_type

from .assets import AssetScope, AssetsView, DataAssetLibrary, ImportAction
from .base import (
    _load_metadata,
    _reconstruct,
    _save_metadata,
)
from .errors import (
    ProjectExistsError,
    ProjectNotFoundError,
)
from .experiment import Experiment
from .folder import (
    WORKSPACE_EXPERIMENT_KIND,
    WORKSPACE_PROJECT_KIND,
    Folder,
    _validate_target_registered,
)
from .fs import PathArg
from .knowledge_item import KnowledgeItem, KnowledgeKind, SourceRef
from .knowledge_write import write_knowledge_item
from .models import FolderMetadata, ProjectMetadata
from .utils import slugify


@concept_type(WORKSPACE_PROJECT_KIND)
class Project(Folder):
    """Research project container.

    Example::

        ws = Workspace("./lab")
        project = ws.add_project("QM9")
        exp = project.add_experiment("baseline", params={"lr": 1e-3})
    """

    _exists_error_cls = ProjectExistsError
    _not_found_error_cls = ProjectNotFoundError

    def __init__(
        self,
        *,
        parent: Workspace | None = None,
        name: str,
        kind: str = WORKSPACE_PROJECT_KIND,
        id: str | None = None,
        workspace: Workspace | None = None,
        fs: FileSystem | None = None,
        _entity_metadata: ProjectMetadata | None = None,
    ) -> None:
        resolved_parent = parent if parent is not None else workspace
        if resolved_parent is None:
            raise ValueError("Project: parent (or workspace) is required")

        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else ProjectMetadata(
                id=id if id is not None else slugify(name),
                name=name,
            )
        )

        self._parent = resolved_parent
        self._name = meta.id
        self._kind = kind
        self._root_path = None
        if fs is not None:
            self._disk_backend = fs
        self._metadata = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=kind,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        self._children_cache = {}

        self._entity_metadata: ProjectMetadata = meta
        self._data_assets: DataAssetLibrary | None = None

    # ── Folder hooks ─────────────────────────────────────────────────────

    def resolve(self) -> Path:
        return self.project_dir

    @classmethod
    def child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """Folder hook — projects live under ``projects/<id>/``.

        Uses :meth:`~Folder.resolve` (not :meth:`~Folder.path`) so listing a
        project never issues a remote ``mkdir`` on the workspace root.
        """
        return Path(parent._disk().join(parent.resolve(), "projects", derived_id))

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> Project:
        """Load ``project.json`` and rebuild entity state. See Folder.from_disk hook docs."""
        meta = _load_metadata(
            ProjectMetadata, parent._disk().join(child_dir, "project.json"), fs=parent._disk()
        )
        folder_meta = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=WORKSPACE_PROJECT_KIND,
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
    def workspace(self) -> Workspace:
        """The owning :class:`Workspace` (alias for :attr:`Folder.parent`)."""
        if self._parent is None:  # pragma: no cover — Project always has a parent
            raise RuntimeError("Project has no parent workspace")
        return cast("Workspace", self._parent)

    @property
    def metadata(self) -> ProjectMetadata:  # type: ignore[override]
        """Project-entity metadata (shadows :attr:`Folder.metadata`)."""
        return self._entity_metadata

    @metadata.setter
    def metadata(self, value: ProjectMetadata) -> None:
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
    def owner(self) -> str:
        return self._entity_metadata.owner

    @property
    def tags(self) -> list[str]:
        return self._entity_metadata.tags

    @property
    def config(self) -> dict[str, Any]:
        return self._entity_metadata.config

    @property
    def project_dir(self) -> Path:
        ws_root = self.workspace.resolve()
        return Path(self._disk().join(ws_root, "projects", self.id))

    @property
    def scope(self) -> AssetScope:
        return AssetScope(kind="project", ids=(self.id,))

    @property
    def assets(self) -> AssetsView:
        """Scope-filtered asset view (read-only queries)."""
        return AssetsView(self.workspace.root, self.scope)

    @property
    def data_assets(self) -> DataAssetLibrary:
        if self._data_assets is None:
            self._data_assets = DataAssetLibrary(
                self.project_dir, self.scope, event_root=_LocalPath(str(self.workspace.root))
            )
        return self._data_assets

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        d = self.project_dir
        self._disk().mkdir(d, parents=True, exist_ok=True)
        meta_path = self._disk().join(d, "project.json")
        _save_metadata(self._entity_metadata, meta_path, fs=self._disk())
        self.write_meta()

    def save(self) -> None:
        """Persist current metadata to disk."""
        meta_path = self._disk().join(self.project_dir, "project.json")
        _save_metadata(self._entity_metadata, meta_path, fs=self._disk())

    def import_asset(  # noqa: ANN201
        self,
        name: str,
        src: str | _LocalPath,
        action: ImportAction = "copy",
        meta: dict[str, Any] | None = None,
    ):
        """Import a ``DataAsset`` into the project library."""
        return self.data_assets.import_asset(name, src, action, meta)

    # ── Experiment CRUD: add / get / set / del / list ─────────────────────

    def add_experiment(
        self,
        name: str,
        *,
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
    ) -> Experiment:
        """Add an experiment (idempotent on slug: re-add returns same node).

        Writes disk scaffold. To **change** fields of an existing experiment,
        use :meth:`set_experiment` (second ``add_experiment`` does not merge
        new params into an existing record).
        """
        resolved_id = id if id is not None else slugify(name)
        _validate_target_registered(self.workspace, default_target)
        child = self._construct_child(
            Experiment,
            name,
            id=resolved_id,
            params=params,
            n_replicas=n_replicas,
            seeds=seeds,
            workflow_source=workflow_source,
            workflow_type=workflow_type,
            git_commit=git_commit,
            description=description,
            tags=tags,
            default_target=default_target,
        )
        return self.add_folder(child)

    def experiment(self, name: str) -> Experiment:
        """Get an existing experiment by name (must exist).

        Raises:
            ExperimentNotFoundError: No experiment with that slug.
        """
        return self.get_folder(name, cls=Experiment)

    def get_experiment(self, name: str) -> Experiment:
        """Alias of :meth:`experiment`."""
        return self.experiment(name)

    def set_experiment(
        self,
        name: str,
        *,
        params: dict[str, JSONValue] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        n_replicas: int | None = None,
        seeds: list[int] | None = None,
        workflow_source: str | None = None,
        workflow_type: str | None = None,
        default_target: str | None = None,
    ) -> Experiment:
        """Update fields of an existing experiment and write to disk.

        Raises:
            ExperimentNotFoundError: Experiment missing.
        """
        exp = self.experiment(name)
        updates: dict[str, Any] = {}
        if params is not None:
            updates["parameter_space"] = dict(params)
        if description is not None:
            updates["description"] = description
        if tags is not None:
            updates["tags"] = list(tags)
        if n_replicas is not None:
            updates["n_replicas"] = n_replicas
        if seeds is not None:
            updates["seeds"] = list(seeds)
        if workflow_source is not None:
            updates["workflow_source"] = workflow_source
        if workflow_type is not None:
            updates["workflow_type"] = workflow_type
        if default_target is not None:
            _validate_target_registered(self.workspace, default_target)
            updates["default_target"] = default_target
        if updates:
            exp._entity_metadata = exp.metadata.model_copy(update=updates)
            exp.save()
        return exp

    def del_experiment(self, name: str) -> None:
        """Delete an experiment directory and its runs."""
        self.remove_folder(name, cls=Experiment)

    def has_experiment(self, name: str) -> bool:
        return self.has_folder(name, cls=Experiment)

    def remove_experiment(self, name: str) -> None:
        """Alias of :meth:`del_experiment`."""
        self.del_experiment(name)

    def experiments(self) -> list[Experiment]:
        """List all experiments under this project."""
        return self.list_folders(cls=Experiment)

    def list_experiments(self) -> list[Experiment]:
        """Alias of :meth:`experiments`."""
        return self.experiments()

    # ── Knowledge CRUD (same shape as experiments) ───────────────────────

    def add_knowledge(
        self,
        name: str,
        *,
        kind: KnowledgeKind = "ProtocolNote",
        body: str = "",
        sources: list[SourceRef | Folder | str] | None = None,
        created_by: str = "user",
        title: str = "",
    ) -> KnowledgeItem:
        """Add a sourced knowledge item under this project."""
        refs = _normalize_sources(sources, default_host=self)
        return write_knowledge_item(
            self,
            name=name,
            kind=kind,
            sources=refs,
            created_by=created_by,
            body=body,
            title=title or name,
        )

    def knowledge(self, name: str) -> KnowledgeItem:
        """Get an existing knowledge item by name (must exist)."""
        return self.get_folder(name, cls=KnowledgeItem)

    def set_knowledge(
        self,
        name: str,
        *,
        kind: KnowledgeKind | None = None,
        body: str | None = None,
        sources: list[SourceRef | Folder | str] | None = None,
        created_by: str | None = None,
        title: str = "",
    ) -> KnowledgeItem:
        """Update an existing knowledge item (rewrites meta/body)."""
        item = self.knowledge(name)
        meta = item.read_knowledge_meta()
        new_kind = kind if kind is not None else meta.kind
        new_sources = (
            _normalize_sources(sources, default_host=self)
            if sources is not None
            else list(meta.sources)
        )
        new_by = created_by if created_by is not None else meta.created_by
        new_body = body if body is not None else item.body()
        return write_knowledge_item(
            self,
            name=name,
            kind=new_kind,
            sources=new_sources,
            created_by=new_by,
            body=new_body,
            title=title or name,
        )

    def del_knowledge(self, name: str) -> None:
        """Delete a knowledge item directory."""
        self.remove_folder(name, cls=KnowledgeItem)

    def knowledges(self) -> list[KnowledgeItem]:
        """List knowledge items mounted directly under this project."""
        return self.list_folders(cls=KnowledgeItem)

    def children(self, kind: str | None = None) -> list[Folder]:
        """List entity children (experiments by default filter)."""
        if kind is not None and kind != WORKSPACE_EXPERIMENT_KIND:
            return []
        return list(self.experiments())


def _normalize_sources(
    sources: list[SourceRef | Folder | str] | None,
    *,
    default_host: Folder,
) -> list[SourceRef]:
    """Accept SourceRef, Folder, or free strings (dataset: / DOI: / path)."""
    if not sources:
        # Sourced knowledge requires ≥1 SourceRef — default to the host itself.
        return [
            SourceRef(
                kind="experiment" if isinstance(default_host, Experiment) else "file",
                ref=getattr(default_host, "id", default_host.name),
            )
        ]
    out: list[SourceRef] = []
    for s in sources:
        if isinstance(s, SourceRef):
            out.append(s)
        elif isinstance(s, Folder):
            kind = "experiment"
            if s.__class__.__name__ == "Run":
                kind = "run"
            elif s.__class__.__name__ == "Project":
                kind = "file"
            elif s.__class__.__name__ == "Experiment":
                kind = "experiment"
            out.append(SourceRef(kind=kind, ref=getattr(s, "id", s.name)))  # type: ignore[arg-type]
        else:
            text = str(s)
            if text.startswith(("dataset:", "model:", "plugin:")):
                out.append(SourceRef(kind="file", ref=text))
            elif text.upper().startswith("DOI:") or text.startswith("10."):
                out.append(SourceRef(kind="reference", ref=text))
            else:
                out.append(SourceRef(kind="file", ref=text))
    return out
