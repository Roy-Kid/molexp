"""Sidecar-backed dataset preview — host half.

The host finds the same-stem ``.py`` sidecar, imports it under a private
name, and applies the frame cap. Science lives in plugins:

* :mod:`molexp.plugins.molpy.preview` — ``BaseTrajectoryReader`` + XYZ
* :mod:`molexp.plugins.molvis.snapshot` — PNG via molvis

When molpy/molvis change, update those plugins — not this module.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .exceptions import (
    AmbiguousReaderError,
    MolExpError,
    NoReaderInSidecarError,
    PreviewReaderError,
    PreviewSidecarNotFoundError,
)

_SIDECAR_MODULE_NAME = "_molexp_preview_reader"
DEFAULT_PREVIEW_LIMIT = 200

__all__ = [
    "DEFAULT_PREVIEW_LIMIT",
    "AmbiguousReaderError",
    "NoReaderInSidecarError",
    "PreviewReaderError",
    "PreviewSidecarNotFoundError",
    "SidecarInfo",
    "asset_has_sidecar",
    "frames_to_extxyz",
    "load_preview",
    "load_sidecar_reader",
    "preview_frames",
    "resolve_sidecar",
    "snapshot_reader",
]


@dataclass(frozen=True)
class SidecarInfo:
    """Result of existence-only sidecar resolution."""

    dataset_path: Path
    sidecar_path: Path


def _sidecar_path_for(dataset_path: Path) -> Path:
    stem = dataset_path.name.split(".", 1)[0]
    return dataset_path.parent / f"{stem}.py"


def resolve_sidecar(dataset_path: str | os.PathLike[str]) -> SidecarInfo | None:
    """Probe for a same-stem ``.py`` sidecar without importing it."""
    path = Path(dataset_path)
    sidecar = _sidecar_path_for(path)
    if sidecar == path or not sidecar.is_file():
        return None
    return SidecarInfo(dataset_path=path, sidecar_path=sidecar)


def _import_sidecar_module(sidecar_path: Path):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(_SIDECAR_MODULE_NAME, sidecar_path)
    if spec is None or spec.loader is None:
        raise PreviewReaderError(str(sidecar_path), "cannot create import spec")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PreviewReaderError(str(sidecar_path), f"import failed: {exc}") from exc
    return module


def load_preview(dataset_path: str | os.PathLike[str]) -> Iterable[object]:
    """Import the sidecar and open its molpy reader (plugin)."""
    info = resolve_sidecar(dataset_path)
    if info is None:
        raise PreviewSidecarNotFoundError(str(dataset_path))
    module = _import_sidecar_module(info.sidecar_path)
    try:
        from molexp.plugins.molpy.preview import open_reader
    except ImportError as exc:
        raise PreviewReaderError(str(info.sidecar_path), str(exc)) from exc
    return open_reader(module, info.dataset_path)


def load_sidecar_reader(dataset_path: str | os.PathLike[str]) -> Iterable[object]:
    """Deprecated name for :func:`load_preview`."""
    return load_preview(dataset_path)


def preview_frames(
    dataset_path: str | os.PathLike[str], *, limit: int = DEFAULT_PREVIEW_LIMIT
) -> list[object]:
    """Return at most *limit* frames. Cap is host-owned."""
    stream = load_preview(dataset_path)
    try:
        return list(islice(stream, limit))
    except MolExpError:
        raise
    except Exception as exc:
        raise PreviewReaderError(str(dataset_path), f"iteration failed: {exc}") from exc


def frames_to_extxyz(frames: Iterable[object]) -> bytes:
    """Delegate XYZ encoding to the molpy plugin."""
    from molexp.plugins.molpy.preview import frames_to_extxyz as _encode

    return _encode(frames)


def snapshot_reader(
    dataset_path: str | os.PathLike[str], *, limit: int = DEFAULT_PREVIEW_LIMIT
) -> bytes:
    """Delegate PNG rendering to the molvis plugin."""
    from molexp.plugins.molvis.snapshot import render_png

    return render_png(preview_frames(dataset_path, limit=limit), dataset_path=str(dataset_path))


def asset_has_sidecar(workspace, asset) -> bool:  # noqa: ANN001
    from .routes._scope import resolve_scope_dir

    scope_dir = resolve_scope_dir(workspace, asset.scope)
    if scope_dir is None:
        return False
    return resolve_sidecar(asset.absolute_path(scope_dir)) is not None
