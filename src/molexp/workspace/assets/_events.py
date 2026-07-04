"""Package-private ``asset.added`` emitter shared by the asset write verbs.

One spelling for the actor + payload vocabulary (vision-loop-12), used by
``ArtifactAccessor.save`` and the ``DataAssetLibrary`` import/register verbs.
Non-fatal by ``emit_workspace_event``'s contract — callers emit only after
their write is durable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .base import Asset

__all__ = ["emit_asset_added"]


def emit_asset_added(
    event_root: Path,
    asset: Asset,
    *,
    name: str,
    extra_refs: Sequence[str] = (),
) -> None:
    """Best-effort ``asset.added`` on the workspace event spine."""
    from ..events import emit_workspace_event

    emit_workspace_event(
        event_root,
        "asset.added",
        "asset-accessor",
        payload={
            "kind": getattr(asset, "kind", "asset"),
            "name": name,
            "content_hash": getattr(asset, "content_hash", None),
        },
        refs=[asset.asset_id, *extra_refs],
    )
