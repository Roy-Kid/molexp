"""Detect which *foreign log* formats a run directory holds.

This is about **source logs**, not scientific record packages. Whether a
directory is a MolRec record is defined by the external molrec spec (Zarr V3
root + group attributes); molexp does not re-host that contract. Landing may
tag record-shaped trees for plugins — see ``molexp.agent.ops.land``.

Detection is **by content, never by extension**. ``leap.log`` and
``log.lammps`` share a suffix and nothing else; a ``.csv`` of atom
coordinates is not a metrics table. Every probe reads a bounded prefix of
the candidate file and looks for a marker the format actually guarantees.

A file nobody can classify yields :attr:`LogFormat.UNKNOWN`, which is a
correct answer — the caller leaves it as a plain artifact rather than
approximating a conversion.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# LAMMPS writes this immediately before every thermo header row.
_LAMMPS_MARKER = b"Per MPI rank memory allocation"
# ...and this on the first line of any log it opens.
_LAMMPS_BANNER = b"LAMMPS ("
_TFEVENT_PREFIX = "events.out.tfevents."
_PROBE_BYTES = 64 * 1024


class LogFormat(StrEnum):
    """A foreign log format an ingester can read."""

    LAMMPS_LOG = "lammps_log"
    """A LAMMPS log carrying at least one thermo table."""

    TENSORBOARD = "tensorboard"
    """A directory holding ``events.out.tfevents.*`` files."""

    CSV = "csv"
    """A delimited table that may be metrics — needs an operator mapping."""

    UNKNOWN = "unknown"
    """Recognised as nothing. Never converted."""


@dataclass(frozen=True, slots=True)
class FormatHit:
    """One detected format inside a run directory."""

    format: LogFormat
    path: Path
    """The file (or directory, for TensorBoard) that carries the format."""

    detail: str = ""
    """Short human-readable reason, for the operator-facing report."""


def _head(path: Path, size: int = _PROBE_BYTES) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(size)
    except OSError:
        return b""


def is_lammps_log(path: Path) -> bool:
    """True when *path* is a LAMMPS log that contains a thermo table.

    The banner alone is not enough: a log whose run never reached a thermo
    section has nothing to convert, so require the thermo marker.
    """
    head = _head(path)
    if not head:
        return False
    if _LAMMPS_MARKER in head:
        return True
    if _LAMMPS_BANNER not in head[:512]:
        return False
    # Banner present but the marker sits past the probe window — scan the
    # rest in bounded chunks rather than loading the whole file.
    try:
        with path.open("rb") as handle:
            handle.seek(len(head))
            tail = handle.read(4 * 1024 * 1024)
    except OSError:
        return False
    return _LAMMPS_MARKER in tail


def is_tensorboard_dir(path: Path) -> bool:
    """True when *path* directly contains at least one tfevents file."""
    if not path.is_dir():
        return False
    return any(child.name.startswith(_TFEVENT_PREFIX) for child in path.iterdir())


def has_metrics_buffer(run_dir: Path) -> bool:
    """True when *run_dir* already carries a host metrics JSONL buffer."""
    return (Path(run_dir) / "metrics" / "metrics.jsonl").is_file()


def _iter_files(run_dir: Path, *, max_depth: int) -> Iterator[Path]:
    roots = [(run_dir, 0)]
    while roots:
        current, depth = roots.pop()
        try:
            children = sorted(current.iterdir())
        except OSError:
            continue
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_dir():
                if depth < max_depth:
                    roots.append((child, depth + 1))
            else:
                yield child


def detect_log_formats(run_dir: Path | str, *, max_depth: int = 3) -> list[FormatHit]:
    """Classify every ingestible log under *run_dir*.

    Args:
        run_dir: Directory to classify.
        max_depth: How deep to descend when looking for artifacts.

    Returns:
        One :class:`FormatHit` per detected artifact; an empty list when the
        directory holds nothing an ingester recognises.
    """
    root = Path(run_dir)
    hits: list[FormatHit] = []
    seen_tb: set[Path] = set()

    for path in _iter_files(root, max_depth=max_depth):
        if path.name.startswith(_TFEVENT_PREFIX):
            parent = path.parent
            if parent not in seen_tb:
                seen_tb.add(parent)
                hits.append(FormatHit(LogFormat.TENSORBOARD, parent, "tfevents files"))
            continue
        looks_like_log = path.suffix in {".log", ".lammps", ".out", ".txt"}
        if looks_like_log and is_lammps_log(path):
            hits.append(FormatHit(LogFormat.LAMMPS_LOG, path, "thermo table present"))
            continue
        if path.suffix == ".csv":
            hits.append(FormatHit(LogFormat.CSV, path, "needs an operator column mapping"))

    return hits
