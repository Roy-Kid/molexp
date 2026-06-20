"""Atomic persistence primitives — cross-layer, citable from any layer.

Holds the canonical atomic write helpers (temp-file + ``os.replace``) and
the advisory inter-process file lock. These sit *above* ``molexp.workspace``
in the same category as :mod:`molexp.path` — a cross-layer primitive that
the bottom storage layers (``workspace`` and the OKF ``knowledge`` layer)
both cite without depending on each other.

This module must not import any molexp business layer; only stdlib and the
root-level ``mollog`` logger are permitted.

``molexp.workspace.base`` / ``molexp.workspace._file_lock`` re-export these
symbols as back-compat aliases (same function objects), so existing call
sites keep working unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from mollog import get_logger

try:  # pragma: no cover - platform-dependent import
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX platforms
    _HAS_FCNTL = False

logger = get_logger(__name__)

# How long a writer waits for a contended lock before failing loudly.
DEFAULT_LOCK_TIMEOUT_SECONDS = 10.0
# Poll cadence while waiting for a contended lock.
_POLL_INTERVAL_SECONDS = 0.02


def atomic_write_json(path: Path, data: object) -> None:
    """Write JSON data to a file atomically via write-to-temp + rename.

    On POSIX systems, os.replace is atomic — if the process crashes
    mid-write, the original file remains intact. This prevents data
    corruption for critical files like run.json, metadata files, and
    workflow-state checkpoints.

    The ``data`` parameter is the structural top-type ``object`` rather
    than ``JSONValue`` because :func:`json.dumps` is invoked with
    ``default=str``, which accepts anything that has a string repr.
    Callers are responsible for ensuring the value is meaningful as
    JSON; ``json.dumps`` raises at write time if not.

    Args:
        path: Destination file path.
        data: JSON-serializable value (or anything ``str()``-coercible).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)  # noqa: PTH105
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)  # noqa: PTH108
        raise


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text to a file atomically via write-to-temp + rename.

    Companion to :func:`atomic_write_json` for plain-text artifacts —
    markdown reports, generated source previews, log snapshots, and OKF
    ``meta.yaml`` / ``index.md`` / ``log.md`` files — that are read back
    as strings rather than parsed as JSON. Same temp-file + ``os.replace``
    pattern; if the process crashes mid-write the original file remains
    intact.

    Args:
        path: Destination file path.
        content: Text to write.
        encoding: Text encoding (default ``"utf-8"``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_")
    tmp = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        tmp.replace(path)
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt).
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


class FileLockTimeoutError(TimeoutError):
    """Raised when an advisory file lock cannot be acquired in time."""


@contextlib.contextmanager
def file_lock(
    lock_path: Path,
    *,
    timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Hold an exclusive advisory lock on *lock_path* for the ``with`` body.

    Serializes read-modify-write cycles on a JSON file written by several
    uncoordinated processes (server, foreground CLI, detached workers).
    Each write is atomic, but an RMW without a lock can still drop a
    concurrent update — this lock closes that window via ``fcntl.flock``
    on a sidecar ``.lock`` file.

    The sidecar lock file is created on demand (never deleted — deleting
    a lock file open in another process would break ``flock`` semantics).

    Graceful degradation: ``fcntl`` is POSIX-only and the sidecar may live
    on a filesystem that rejects lock files (remote mounts). In both cases
    the context manager degrades to a no-op with a debug log line.

    Args:
        lock_path: Sidecar lock-file path (e.g. ``run.json.lock``).
        timeout: Seconds to wait for a contended lock.

    Raises:
        FileLockTimeoutError: The lock stayed contended past *timeout*.
    """
    if not _HAS_FCNTL:
        logger.debug(f"file_lock: fcntl unavailable; proceeding without lock for {lock_path}")
        yield
        return

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        logger.debug(
            f"file_lock: cannot create {lock_path}; proceeding without lock", exc_info=True
        )
        yield
        return

    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise FileLockTimeoutError(
                        f"could not acquire advisory lock {lock_path} within "
                        f"{timeout:.1f}s — another molexp process is holding it"
                    ) from None
                time.sleep(_POLL_INTERVAL_SECONDS)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


__all__ = [
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "FileLockTimeoutError",
    "atomic_write_json",
    "atomic_write_text",
    "file_lock",
]
