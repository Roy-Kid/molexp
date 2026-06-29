"""Preview route — render a sidecar-backed dataset under ``/api/assets``.

A dataset asset whose on-disk file has a same-stem ``.py`` sidecar (see
:mod:`molexp.server.preview`) can be previewed without the client knowing
anything about the loader. The sidecar's reader runs **only** on this explicit
request — never during listing or indexing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from molexp.workspace.assets import scan

from ..dependencies import get_workspace
from ..exceptions import AssetNotFoundError
from ..preview import (
    DEFAULT_PREVIEW_LIMIT,
    frames_to_extxyz,
    preview_frames,
    snapshot_reader,
)
from ._scope import resolve_scope_dir

router = APIRouter(prefix="/assets", tags=["assets"])


def _resolve_dataset_path(workspace, asset_id: str) -> Path:  # noqa: ANN001
    """Resolve the asset's on-disk path, raising a typed 404 if unknown."""
    asset = scan.get_asset(workspace.root, asset_id)
    if asset is None:
        raise AssetNotFoundError(asset_id)
    scope_dir = resolve_scope_dir(workspace, asset.scope)
    if scope_dir is None:
        raise AssetNotFoundError(asset_id)
    return asset.absolute_path(scope_dir)


@router.get("/{asset_id}/preview")
def preview_asset(
    asset_id: str,
    format: Literal["frames", "png"] = "frames",
    limit: int = DEFAULT_PREVIEW_LIMIT,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> StreamingResponse:
    """Render a preview of a sidecar-backed dataset asset.

    Args:
        asset_id: Catalog id of the dataset asset.
        format: ``frames`` (extended-XYZ bytes for the JS trajectory viewer)
            or ``png`` (headless molvis snapshot).
        limit: Host-owned cap on the number of frames previewed.

    Returns:
        A streaming response — ``chemical/x-xyz`` for ``frames``,
        ``image/png`` for ``png``.

    Raises:
        AssetNotFoundError: Unknown asset id (404).
        PreviewSidecarNotFoundError: No sidecar next to the dataset (404).
        NoReaderInSidecarError / AmbiguousReaderError / PreviewReaderError:
            The sidecar is empty / ambiguous / broken (422).
    """
    dataset_path = _resolve_dataset_path(workspace, asset_id)

    if format == "png":
        png = snapshot_reader(dataset_path, limit=limit)
        return StreamingResponse(io.BytesIO(png), media_type="image/png")

    extxyz = frames_to_extxyz(preview_frames(dataset_path, limit=limit))
    return StreamingResponse(io.BytesIO(extxyz), media_type="chemical/x-xyz")
