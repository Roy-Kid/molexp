"""Document-embed target resolution + entity summaries (knowledge-docs-05).

The workspace-layer support for the "Notion-style document" embed verb: a
:class:`~molexp.workspace.concepts.Note` embeds a live workspace entity -- a
``Run`` / ``Experiment`` / ``ReferenceConcept`` (all :class:`Folder` subclasses)
or an :class:`~molexp.workspace.assets.base.Asset` -- as a typed provenance edge.
This module owns the read-only halves of that verb:

- :func:`resolve_embed_target` -- turn a ``Folder | Asset`` target into the
  ``dst`` :class:`Folder` handed to :func:`molexp.workspace.folder.append_link`.
  A ``Folder`` is its own ``dst``; an ``Asset`` is resolved to its in-tree record
  directory ``<scope_dir>/assets/<asset_id>/`` (pointed at, never copied) and
  wrapped in a thin path-only ``Folder`` via the injected ``pin`` builder.
- :func:`default_role_for` -- the per-kind default :class:`EdgeRole` (all drawn
  from the frozen vocabulary; no new roles): ``workspace.run`` /
  ``workspace.experiment`` -> ``records``, ``reference.reference`` -> ``cites``,
  everything else (including any ``Asset``) -> ``references``.
- :func:`summarize_entity` / :class:`EntitySummary` -- a pure read projection of
  an entity's ``id`` / ``kind`` / ``title`` for the future UI card. No new state.

Missing preconditions (an ``Asset`` with no resolvable record dir, or no
``root`` to anchor one) raise -- never a silent fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path

from pydantic import BaseModel

from .assets.base import Asset, AssetScope
from .bundle_index import extract_title
from .concepts import REFERENCE_KIND
from .edges import DEFAULT_EDGE_ROLE, EdgeRole
from .folder import Folder

# Per-kind default edge roles, all members of the frozen ``EdgeRole`` vocabulary
# (edges.py). A kind absent here defaults to ``DEFAULT_EDGE_ROLE`` ("references").
_DEFAULT_ROLE_BY_KIND: dict[str, EdgeRole] = {
    "workspace.run": "records",
    "workspace.experiment": "records",
    REFERENCE_KIND: "cites",
}


class EntitySummary(BaseModel, frozen=True):
    """A read-only projection of a workspace entity for a UI card.

    Attributes:
        id: The entity's stable id -- a ``Folder`` name (run/experiment/reference
            id) or an ``Asset``'s ``asset_id``.
        kind: The concept/asset kind (e.g. ``"workspace.run"`` / ``"data"``).
        title: A human title -- the ``index.md`` H1 (or ``ReferenceMeta.title``)
            when present, else the entity's id/name.
    """

    id: str
    kind: str
    title: str


def _scope_dir(root: Path, scope: AssetScope) -> Path:
    """Anchor an :class:`AssetScope` to its on-disk directory under *root*.

    Mirrors the frozen Layout naming law (the same shape
    :func:`molexp.workspace.assets.scan` enumerates): ``projects/<id>`` /
    ``experiments/<id>`` containers, and the mandatory ``run-`` prefix on a run
    directory. The ``run-`` prefix is added only when absent, so a bare run id in
    ``scope.ids`` and an already-prefixed one both resolve correctly.

    Args:
        root: The workspace root directory.
        scope: The asset's owning scope.

    Returns:
        The scope's on-disk directory.
    """
    directory = root
    ids = scope.ids
    if scope.kind == "workspace":
        return directory
    directory = directory / "projects" / ids[0]
    if scope.kind == "project":
        return directory
    directory = directory / "experiments" / ids[1]
    if scope.kind == "experiment":
        return directory
    run_seg = ids[2] if ids[2].startswith("run-") else f"run-{ids[2]}"
    return directory / "runs" / run_seg


def asset_record_dir(asset: Asset, root: str | PathLike[str] | None) -> Path:
    """Resolve *asset* to its in-tree record dir ``<scope_dir>/assets/<asset_id>/``.

    This is the same location :func:`molexp.workspace.assets.scan` reads a user
    ``DataAsset`` from; the asset payload is only *pointed at* by an embed edge,
    never copied. The directory is anchored from the workspace *root* via the
    frozen Layout naming law (:func:`_scope_dir`).

    Args:
        asset: The asset whose record dir to locate.
        root: The workspace root to anchor the asset's scope against.

    Returns:
        The absolute record directory (guaranteed to exist).

    Raises:
        ValueError: If *root* is ``None`` (an ``Asset`` cannot be anchored).
        FileNotFoundError: If no record dir exists for the asset -- e.g. an asset
            known only from a manifest, with no ``assets/<id>/`` directory.
    """
    if root is None:
        raise ValueError(
            f"asset_record_dir requires a workspace root to anchor asset {asset.asset_id!r}"
        )
    record_dir = _scope_dir(Path(root), asset.scope) / "assets" / asset.asset_id
    if not record_dir.is_dir():
        raise FileNotFoundError(f"no record directory for asset {asset.asset_id!r} at {record_dir}")
    return record_dir


def default_role_for(target: Folder | Asset) -> EdgeRole:
    """Return the per-kind default embed :class:`EdgeRole` for *target*.

    All roles come from the frozen vocabulary (no new roles): a ``Run`` or
    ``Experiment`` folder -> ``records``, a ``ReferenceConcept`` -> ``cites``,
    and any ``Asset`` (or other folder kind) -> ``references``
    (:data:`~molexp.workspace.edges.DEFAULT_EDGE_ROLE`).

    Args:
        target: The embed target (a ``Folder`` or an ``Asset``).

    Returns:
        The default role for *target*'s kind.

    Raises:
        TypeError: If *target* is neither a ``Folder`` nor an ``Asset``.
    """
    if isinstance(target, Asset):
        return DEFAULT_EDGE_ROLE
    if isinstance(target, Folder):
        return _DEFAULT_ROLE_BY_KIND.get(target.kind, DEFAULT_EDGE_ROLE)
    raise TypeError(f"embed target must be a Folder or Asset, got {type(target).__name__}")


def resolve_embed_target(
    target: Folder | Asset,
    *,
    root: str | PathLike[str] | None,
    pin: Callable[[Path], Folder],
) -> Folder:
    """Resolve *target* to the ``dst`` :class:`Folder` for an embed edge.

    A ``Folder`` target is its own ``dst``. An ``Asset`` target is resolved to
    its in-tree record dir (:func:`asset_record_dir`) and wrapped in a thin
    path-only ``Folder`` via *pin* (the bundle's ``_pinned_parent``), so the
    asset payload is only pointed at.

    Args:
        target: The embed target (a ``Folder`` or an ``Asset``).
        root: The workspace root, needed only to anchor an ``Asset``.
        pin: Builds a thin ``Folder`` whose ``resolve()`` equals a given path.

    Returns:
        The ``dst`` folder to hand to :func:`~molexp.workspace.folder.append_link`.

    Raises:
        TypeError: If *target* is neither a ``Folder`` nor an ``Asset``.
        ValueError: If an ``Asset`` target has no *root* to anchor it.
        FileNotFoundError: If an ``Asset`` target has no on-disk record dir.
    """
    if isinstance(target, Asset):
        return pin(asset_record_dir(target, root))
    if isinstance(target, Folder):
        return target
    raise TypeError(f"embed target must be a Folder or Asset, got {type(target).__name__}")


def summarize_entity(
    target: Folder | Asset, *, root: str | PathLike[str] | None = None
) -> EntitySummary:
    """Project *target* to an :class:`EntitySummary` (a pure read; writes nothing).

    - A ``Folder`` (``Run`` / ``Experiment`` / ``ReferenceConcept``): ``id`` is
      the folder name, ``kind`` is its ``meta.yaml`` type (falling back to the
      folder's kind), and ``title`` is the ``index.md`` H1 -- with a
      ``ReferenceConcept``'s ``ReferenceMeta.title`` preferred -- else the name.
    - An ``Asset``: ``id`` is the ``asset_id``, ``kind`` is the asset kind, and
      ``title`` is the asset name. *root* is required to anchor the asset (its
      record dir must be locatable), matching the embed verb's precondition.

    Args:
        target: The entity to summarize (a ``Folder`` or an ``Asset``).
        root: The workspace root; required only when *target* is an ``Asset``.

    Returns:
        The read-only :class:`EntitySummary`.

    Raises:
        TypeError: If *target* is neither a ``Folder`` nor an ``Asset``.
        ValueError: If an ``Asset`` target has no *root* to anchor it.
        FileNotFoundError: If an ``Asset`` target has no on-disk record dir.
    """
    if isinstance(target, Asset):
        # Anchor + validate the asset is locatable in the tree (raises on a
        # missing root or record dir -- no silent fallback).
        asset_record_dir(target, root)
        # ``kind`` is declared only on concrete Asset subclasses (DataAsset,
        # ...), not the abstract base -- read it the way ``assets.scan`` does.
        kind = str(getattr(target, "kind", ""))
        return EntitySummary(id=target.asset_id, kind=kind, title=target.name)
    if isinstance(target, Folder):
        meta = target.read_meta()
        raw_type = meta.get("type")
        kind = str(raw_type) if raw_type is not None else target.kind
        title: str | None = None
        if kind == REFERENCE_KIND:
            ref_title = meta.get("title")
            if isinstance(ref_title, str) and ref_title:
                title = ref_title
        if title is None:
            title = extract_title(target.read_index()) or target.name
        return EntitySummary(id=target.name, kind=kind, title=title)
    raise TypeError(f"summary target must be a Folder or Asset, got {type(target).__name__}")


__all__ = [
    "EntitySummary",
    "asset_record_dir",
    "default_role_for",
    "resolve_embed_target",
    "summarize_entity",
]
