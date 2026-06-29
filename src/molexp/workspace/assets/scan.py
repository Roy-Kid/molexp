"""Manifest-scanning asset query layer — replaces the derived SQLite catalog.

The authoritative source for assets is each scope's ``assets.json``
(:class:`~molexp.workspace.assets.manifest.AssetManifest`). This module answers
the cross-cutting asset queries that the old ``AssetCatalog`` served — but by
scanning those manifests directly, so there is no second on-disk index to keep
in sync (the One-source-of-truth law: every index is derived; here we drop the
index entirely rather than rebuild it).

For a research workspace the asset count is small, so a full scan per query is
cheap and always reflects on-disk truth. Each :class:`~molexp.workspace.assets.base.Asset`
carries its own :class:`~molexp.workspace.assets.base.AssetScope`, so filtering
is done on the asset itself — no scope→directory reconstruction is needed for
reads. Results are ordered deterministically by ``(created_at, asset_id)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from os import PathLike
from pathlib import Path

from ..utils import generate_asset_id
from .base import Asset, AssetScope, Producer
from .manifest import AssetManifest

_SCOPE_KIND_RANK: dict[str, int] = {
    "workspace": 0,
    "project": 1,
    "experiment": 2,
    "run": 3,
}


def _iter_scope_dirs(root: Path) -> Iterator[Path]:
    """Yield every scope directory that may hold an ``assets.json``.

    Order: workspace root, then each project, experiment, and run in sorted
    path order (deterministic, independent of filesystem walk order).
    """
    yield root
    projects_dir = root / "projects"
    if not projects_dir.is_dir():
        return
    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        yield proj_dir
        experiments_dir = proj_dir / "experiments"
        if not experiments_dir.is_dir():
            continue
        for exp_dir in sorted(experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            yield exp_dir
            runs_dir = exp_dir / "runs"
            if not runs_dir.is_dir():
                continue
            for run_dir in sorted(runs_dir.iterdir()):
                if run_dir.is_dir():
                    yield run_dir


def _all_assets(root: Path) -> list[Asset]:
    """Load every asset across all scope manifests under ``root``."""
    out: list[Asset] = []
    for scope_dir in _iter_scope_dirs(root):
        out.extend(AssetManifest(scope_dir).load().values())
    return out


def _sorted(assets: list[Asset]) -> list[Asset]:
    return sorted(assets, key=lambda a: (a.created_at, a.asset_id))


def _kind_value(kind: str | type[Asset] | None) -> str | None:
    """Normalize a ``kind`` filter (str or Asset subclass) to its string value."""
    if kind is None:
        return None
    if isinstance(kind, str):
        return kind
    try:
        return kind.model_fields["kind"].default  # type: ignore[attr-defined]
    except (AttributeError, KeyError):
        return None


def _scope_matches(asset_scope: AssetScope, scope: AssetScope, recursive: bool) -> bool:
    """Mirror ``AssetCatalog`` scope filtering (exact, or recursive prefix)."""
    if not recursive:
        return asset_scope.kind == scope.kind and asset_scope.ids == scope.ids
    arank = _SCOPE_KIND_RANK.get(asset_scope.kind)
    srank = _SCOPE_KIND_RANK.get(scope.kind, 0)
    if arank is None or arank < srank:
        return False
    n = len(scope.ids)
    return asset_scope.ids[:n] == scope.ids


def scan_assets(
    root: str | PathLike[str],
    *,
    kind: str | type[Asset] | None = None,
    scope: AssetScope | None = None,
    producer_run: str | None = None,
    producer_task: str | None = None,
    tag: tuple[str, str] | None = None,
    limit: int | None = None,
    recursive: bool = False,
) -> list[Asset]:
    """Query assets across all manifests, mirroring ``AssetCatalog.query_assets``.

    With ``recursive=True`` and a ``scope``, the match includes any sub-scope
    whose ids extend the given scope's ids. Results are ordered by
    ``(created_at, asset_id)``.
    """
    root = Path(root)
    kind_str = _kind_value(kind)
    out: list[Asset] = []
    for asset in _sorted(_all_assets(root)):
        if kind_str is not None and getattr(asset, "kind", None) != kind_str:
            continue
        if scope is not None and not _scope_matches(asset.scope, scope, recursive):
            continue
        producer = asset.producer
        if producer_run and (producer is None or producer.run_id != producer_run):
            continue
        if producer_task and (producer is None or producer.task_id != producer_task):
            continue
        if tag is not None:
            tk, tv = tag
            if asset.tags.get(tk) != tv:
                continue
        out.append(asset)
        if limit is not None and len(out) >= limit:
            break
    return out


def get_asset(root: str | PathLike[str], asset_id: str) -> Asset | None:
    """Return the asset with ``asset_id`` from any scope, else ``None``."""
    for asset in _all_assets(Path(root)):
        if asset.asset_id == asset_id:
            return asset
    return None


def find_by_content_hash(root: str | PathLike[str], content_hash: str) -> Asset | None:
    """Return the earliest asset whose ``content_hash`` matches, else ``None``.

    Content-addressed lookup used by the workflow cache's re-registration path:
    a match means the bytes are present somewhere in the workspace.
    """
    if not content_hash:
        return None
    for asset in _sorted(_all_assets(Path(root))):
        if asset.content_hash == content_hash:
            return asset
    return None


def reregister_artifact(
    root: str | PathLike[str],
    scope_dir: str | PathLike[str],
    *,
    name: str | None,
    content_hash: str,
    target_scope: AssetScope,
    producer_task: str | None = None,
) -> Asset | None:
    """Idempotently re-register a content-addressed artifact into ``target_scope``.

    Looks the artifact up by ``content_hash`` across the workspace; if found,
    writes a fresh artifact row into ``scope_dir``'s manifest pointing at the
    SAME path + ``content_hash`` (no recompute, no byte recopy). Returns the new
    asset, or ``None`` when the bytes are absent (fresh workspace) so the caller
    can skip gracefully. Idempotent on ``(name, content_hash)`` within the scope.
    """
    source = find_by_content_hash(root, content_hash)
    if source is None:
        return None
    manifest = AssetManifest(scope_dir)
    for existing in manifest.load().values():
        if (
            getattr(existing, "kind", None) == "artifact"
            and existing.content_hash == content_hash
            and existing.scope == target_scope
            and existing.name == name
        ):
            return existing
    now = datetime.now()
    clone = source.model_copy(
        update={
            "asset_id": generate_asset_id(),
            "name": name,
            "scope": target_scope,
            "created_at": now,
            "updated_at": now,
            "producer": Producer(
                run_id=target_scope.ids[-1] if target_scope.ids else None,
                task_id=producer_task,
            ),
        }
    )
    manifest.register(clone)
    return clone
