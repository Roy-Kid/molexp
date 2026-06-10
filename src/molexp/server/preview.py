"""Sidecar-backed dataset preview.

A non-standard, framework-specific molecular dataset (reference case: a QM9
dataset downloaded via molnex) carries its own loader as a *same-stem* ``.py``
sidecar placed next to it: ``qm9.tar.bz2`` → ``qm9.py``. The sidecar defines
**exactly one** concrete subclass of :class:`molpy.io.BaseTrajectoryReader` — a
pure ``Iterable[molpy.Frame]`` reader. The sidecar knows nothing about molvis
or the server; molexp (the host) imports it on an explicit preview request
only, instantiates the single reader with the dataset path, takes a host-owned
:func:`itertools.islice` of frames, and renders them.

**Everything is index-driven.** Previews operate on a *registered* dataset
asset (resolved through the catalog by ``asset_id``) — there is no scanning of
the workspace file tree to auto-discover previewable files, and no path-based
preview of unregistered files. The sidecar itself is **not** an asset: it is
simply the same-stem ``.py`` sibling of the registered asset's resolved path.

* :func:`resolve_sidecar` is **pure and side-effect-free** — given a known
  (registered) path it computes the sibling ``.py`` and checks existence only.
  It never imports the module, so listing the catalog never executes user code.
* :func:`load_sidecar_reader` is the **explicit** path — it imports the sidecar
  under a private, non-``"__main__"`` module name (so the sidecar's
  ``if __name__ == "__main__"`` guard never runs), collects the concrete
  ``BaseTrajectoryReader`` subclasses defined in that module, requires exactly
  one, and instantiates it.

Trust aligns with the existing ``molexp run`` model: ``entry.py`` already
imports user ``.py`` unsandboxed in-process; importing a sidecar reader
server-side is the same trusted-local-workspace boundary, stated deliberately.
Subprocess sandboxing (cf. ``agent/execution_env.py``) is a deferrable
hardening option, not implemented here.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import (
    AmbiguousReaderError,
    MolExpError,
    NoReaderInSidecarError,
    PreviewReaderError,
    PreviewSidecarNotFoundError,
)

if TYPE_CHECKING:
    # molpy is an optional peer dependency (not declared in pyproject) —
    # preview endpoints degrade gracefully when it is absent.
    import molpy  # ty: ignore[unresolved-import]
    from molpy.io import BaseTrajectoryReader  # ty: ignore[unresolved-import]

# Private module name for sidecar import — never ``"__main__"``, so the
# sidecar's ``if __name__ == "__main__"`` guard stays dormant.
_SIDECAR_MODULE_NAME = "_molexp_preview_reader"

# Host-owned default cap on previewed frames. The sidecar does not dictate it.
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
    "load_sidecar_reader",
    "preview_frames",
    "resolve_sidecar",
    "snapshot_reader",
]


@dataclass(frozen=True)
class SidecarInfo:
    """Result of existence-only sidecar resolution.

    Attributes:
        dataset_path: The dataset file the sidecar binds to.
        sidecar_path: The same-stem ``.py`` sibling that exists on disk.
    """

    dataset_path: Path
    sidecar_path: Path


def _sidecar_path_for(dataset_path: Path) -> Path:
    """Return the same-stem sibling ``.py`` for ``dataset_path``.

    The stem strips *all* suffixes (``qm9.tar.bz2`` → ``qm9``) and works for
    directories too (``qm9/`` → ``qm9.py``), matching how molnex names its
    downloads.
    """
    stem = dataset_path.name.split(".", 1)[0]
    return dataset_path.parent / f"{stem}.py"


def resolve_sidecar(dataset_path: str | os.PathLike[str]) -> SidecarInfo | None:
    """Probe for a same-stem ``.py`` sidecar **without importing it**.

    Pure and side-effect-free: computes the sibling path and checks existence
    only. Safe to call during passive catalog indexing / file listing — it
    never executes user code.

    Args:
        dataset_path: Path to the dataset file or directory.

    Returns:
        A :class:`SidecarInfo` when a sibling ``.py`` exists, else ``None``.
    """
    path = Path(dataset_path)
    sidecar = _sidecar_path_for(path)
    # A dataset that *is* the ``.py`` (stem == name) is not its own sidecar.
    if sidecar == path:
        return None
    if not sidecar.is_file():
        return None
    return SidecarInfo(dataset_path=path, sidecar_path=sidecar)


def _import_sidecar_module(sidecar_path: Path):  # noqa: ANN202
    """Import the sidecar under the private preview module name.

    Mirrors :func:`molexp.entry` workflow loading. The non-``"__main__"`` name
    keeps the sidecar's ``__main__`` guard from running.
    """
    spec = importlib.util.spec_from_file_location(_SIDECAR_MODULE_NAME, sidecar_path)
    if spec is None or spec.loader is None:
        raise PreviewReaderError(str(sidecar_path), "cannot create import spec")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PreviewReaderError(str(sidecar_path), f"import failed: {exc}") from exc
    return module


def _reader_subclasses_in(module) -> list[type]:  # noqa: ANN001
    """Collect concrete ``BaseTrajectoryReader`` subclasses defined in ``module``.

    Filters to classes whose ``__module__`` is the sidecar itself (so imported
    base classes are ignored), excluding the abstract ``BaseTrajectoryReader`` /
    ``MmapTrajectoryReader`` bases.
    """
    from molpy.io import (  # ty: ignore[unresolved-import]
        BaseTrajectoryReader,
        MmapTrajectoryReader,
    )

    found: list[type] = []
    for obj in vars(module).values():
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, BaseTrajectoryReader):
            continue
        if obj in (BaseTrajectoryReader, MmapTrajectoryReader):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isabstract(obj):
            continue
        found.append(obj)
    return found


def load_sidecar_reader(dataset_path: str | os.PathLike[str]) -> BaseTrajectoryReader:
    """Import the sidecar and return its single reader, bound to the dataset.

    Args:
        dataset_path: Path to the dataset file or directory.

    Returns:
        An instance of the sidecar's sole ``BaseTrajectoryReader`` subclass,
        constructed with ``dataset_path``.

    Raises:
        PreviewSidecarNotFoundError: No sibling ``.py`` exists (404).
        NoReaderInSidecarError: The sidecar defines no reader subclass (422).
        AmbiguousReaderError: It defines more than one (422).
        PreviewReaderError: Import or instantiation failed (422).
    """
    info = resolve_sidecar(dataset_path)
    if info is None:
        raise PreviewSidecarNotFoundError(str(dataset_path))

    module = _import_sidecar_module(info.sidecar_path)
    readers = _reader_subclasses_in(module)
    if not readers:
        raise NoReaderInSidecarError(str(info.sidecar_path))
    if len(readers) > 1:
        raise AmbiguousReaderError(str(info.sidecar_path), [r.__name__ for r in readers])

    reader_cls = readers[0]
    try:
        return reader_cls(info.dataset_path)
    except Exception as exc:
        raise PreviewReaderError(str(info.dataset_path), f"instantiation failed: {exc}") from exc


def preview_frames(
    dataset_path: str | os.PathLike[str], *, limit: int = DEFAULT_PREVIEW_LIMIT
) -> list[molpy.Frame]:
    """Return at most ``limit`` frames from the sidecar reader.

    The cap is applied **host-side** via :func:`itertools.islice` on the
    reader's iterator — the sidecar does not get to dictate it.

    Args:
        dataset_path: Path to the dataset file or directory.
        limit: Maximum number of frames to materialize.

    Returns:
        A list of up to ``limit`` :class:`molpy.Frame` objects.

    Raises:
        MolExpError: Any typed preview error (propagated unchanged).
        PreviewReaderError: Iterating the reader failed (422).
    """
    reader = load_sidecar_reader(dataset_path)
    try:
        return list(islice(reader, limit))
    except MolExpError:
        raise
    except Exception as exc:
        raise PreviewReaderError(str(dataset_path), f"iteration failed: {exc}") from exc


def frames_to_extxyz(frames: Iterable[molpy.Frame]) -> bytes:
    """Serialize frames to extended-XYZ trajectory bytes via molpy.

    Args:
        frames: The frames to write (consumed once).

    Returns:
        The extxyz trajectory encoded as UTF-8 bytes.
    """
    from molpy.io import write_xyz_trajectory  # ty: ignore[unresolved-import]

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_xyz_trajectory(tmp_path, list(frames))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def snapshot_reader(
    dataset_path: str | os.PathLike[str], *, limit: int = DEFAULT_PREVIEW_LIMIT
) -> bytes:
    """Render a headless molvis PNG snapshot of the previewed frames.

    molvis is imported **host-side only** and in headless mode
    (``MOLVIS_HEADLESS=1``); the sidecar never touches it.

    Args:
        dataset_path: Path to the dataset file or directory.
        limit: Maximum number of frames to load into the trajectory.

    Returns:
        PNG image bytes.

    Raises:
        PreviewReaderError: molvis is unavailable or rendering failed (422).
    """
    frames = preview_frames(dataset_path, limit=limit)
    os.environ.setdefault("MOLVIS_HEADLESS", "1")
    try:
        from molvis import Molvis  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise PreviewReaderError(str(dataset_path), "molvis is not installed") from exc
    try:
        viewer = Molvis()
        viewer.set_trajectory(frames)
        return viewer.snapshot()
    except Exception as exc:
        raise PreviewReaderError(str(dataset_path), f"snapshot failed: {exc}") from exc


def asset_has_sidecar(workspace, asset) -> bool:  # noqa: ANN001
    """Whether the asset's on-disk file has a same-stem preview sidecar.

    Existence-only — never imports anything. Used to light up the listing flag
    so the UI can offer a preview without executing code.
    """
    from .routes._scope import resolve_scope_dir

    scope_dir = resolve_scope_dir(workspace, asset.scope)
    if scope_dir is None:
        return False
    return resolve_sidecar(asset.absolute_path(scope_dir)) is not None
