"""Dataset preview via ``molpy.io.BaseTrajectoryReader``.

Sidecars declare exactly one concrete reader subclass. This module
imports molpy's public surface and must track molpy releases.
"""

from __future__ import annotations

import inspect
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.server.exceptions import (
    AmbiguousReaderError,
    NoReaderInSidecarError,
    PreviewReaderError,
)

if TYPE_CHECKING:
    from molpy.io import BaseTrajectoryReader


def require_molpy() -> tuple[type, type]:
    """Return ``(Frame, BaseTrajectoryReader)`` or raise a rebuild hint."""
    try:
        from molpy import Frame
        from molpy.io import BaseTrajectoryReader
    except ImportError as exc:
        raise ImportError(
            "Preview needs molpy. Rebuild the science stack: "
            "`maturin develop --release` in molrs/molrs-python, then "
            "`pip install -e .` in molpy."
        ) from exc
    return Frame, BaseTrajectoryReader


def readers_in(module: object) -> list[type]:
    """Concrete ``BaseTrajectoryReader`` subclasses defined in *module*."""
    _, base = require_molpy()
    found: list[type] = []
    for obj in vars(module).values():
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, base):
            continue
        if obj is base:
            continue
        if getattr(obj, "__module__", None) != getattr(module, "__name__", None):
            continue
        if inspect.isabstract(obj):
            continue
        found.append(obj)
    return found


def open_reader(module: object, dataset_path: Path) -> BaseTrajectoryReader:
    """Instantiate the sidecar's sole reader against *dataset_path*."""
    readers = readers_in(module)
    if not readers:
        raise NoReaderInSidecarError(str(getattr(module, "__file__", dataset_path)))
    if len(readers) > 1:
        raise AmbiguousReaderError(
            str(getattr(module, "__file__", dataset_path)),
            [cls.__name__ for cls in readers],
        )
    try:
        return readers[0](dataset_path)
    except Exception as exc:
        raise PreviewReaderError(str(dataset_path), f"instantiation failed: {exc}") from exc


def frames_to_extxyz(frames: Iterable[object]) -> bytes:
    """Serialize molpy Frames through ``molpy.io.write_xyz_trajectory``."""
    require_molpy()
    from molpy.io import write_xyz_trajectory

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_xyz_trajectory(tmp_path, list(frames))
        return tmp_path.read_bytes()
    except Exception as exc:
        raise PreviewReaderError(str(tmp_path), f"write_xyz_trajectory failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
