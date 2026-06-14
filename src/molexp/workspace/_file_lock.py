"""Advisory inter-process file locking for workspace JSON read-modify-write.

``run.json`` is written by several uncoordinated processes — the server,
the foreground CLI, and detached background workers. Each write is atomic
(temp file + rename), but a read-modify-write cycle without a lock can
still drop a concurrent update (last-writer-wins). :func:`file_lock`
serializes those cycles via ``fcntl.flock`` on a sidecar ``.lock`` file.

Design constraints:

- **Cheap** — hold the lock only around the RMW itself, never around
  long-running work.
- **Advisory** — only cooperating writers are excluded; readers are
  unaffected (the JSON file itself stays atomically replaced).
- **Graceful degradation** — ``fcntl`` is POSIX-only and the sidecar may
  live on a filesystem that rejects lock files (remote mounts). In both
  cases the context manager degrades to a no-op with a debug log line,
  preserving today's behavior instead of failing the write.
"""

from __future__ import annotations

import contextlib
import os
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


class FileLockTimeoutError(TimeoutError):
    """Raised when an advisory file lock cannot be acquired in time."""


@contextlib.contextmanager
def file_lock(
    lock_path: Path,
    *,
    timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Hold an exclusive advisory lock on *lock_path* for the ``with`` body.

    The sidecar lock file is created on demand (never deleted — deleting
    a lock file open in another process would break ``flock`` semantics).

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


__all__ = ["DEFAULT_LOCK_TIMEOUT_SECONDS", "FileLockTimeoutError", "file_lock"]
